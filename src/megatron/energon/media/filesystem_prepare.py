# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Callable, Iterator

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.aggregator_pool import AggregatorPool
from megatron.energon.flavors.webdataset.config import INDEX_SQLITE_FILENAME, MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.prepare import (
    IndexAggregatable,
    IndexMediaMetadata,
    SqliteIndexWriterAggregator,
)
from megatron.energon.flavors.webdataset.structs import ShardInfo
from megatron.energon.media.extractor import MediaFilterConfig
from megatron.energon.media.metadata import serialize_media_metadata


def prepare_filesystem_dataset(
    root_path: EPath,
    media_filter: MediaFilterConfig,
    *,
    progress: bool,
    workers: int = 16,
) -> int:
    """Scan a filesystem dataset and materialize media metadata into SQLite.

    Args:
        root_path: Dataset root directory.
        media_filter: Media filtering configuration.
        progress: Whether to display a tqdm progress bar.

    Returns:
        Number of metadata entries written to the database.
    """

    # Only supporting local file system, because sqlite does not support remote file systems.
    # TODO: Implement remote file systems. Maybe create locally in tmp then upload?
    assert str(root_path).startswith("/"), (
        f"SQLite path must be absolute local file system path: {root_path}"
    )

    root = Path(str(root_path))
    assert root.is_dir(), f"Expected directory for filesystem dataset, got {root}"
    assert root.is_absolute(), f"Filesystem dataset path must be absolute: {root}"

    meta_dir = root / MAIN_FOLDER_NAME
    meta_dir.mkdir(exist_ok=True, parents=True)

    files = _collect_media_files(root=root, media_filter=media_filter, progress=progress)

    sqlite_path = EPath(meta_dir / INDEX_SQLITE_FILENAME)

    agg_progress_fn: Callable[[Iterator[int], int], Iterator[int]] | None = None
    if progress:
        from tqdm.auto import tqdm

        def agg_progress_fn(iterator: Iterator[int], total: int) -> Iterator[int]:
            with tqdm(iterator, total=total, unit="file", desc="Processing media files") as bar:
                yield from bar

    aggregator = SqliteIndexWriterAggregator(
        sqlite_path,
        total_tasks=len(files),
        progress_fn=agg_progress_fn,
        enable_media_metadata=True,
        media_filter=media_filter,
        reset_tables=False,
        enable_sample_tables=False,
        progress_on_media=progress,
    )

    pool = AggregatorPool[
        Path,
        IndexAggregatable,
        tuple[list[ShardInfo], set[str], bool, list[tuple[str, int]]],
    ](
        num_workers=min(workers, len(files)) or 1,
        user_produce_data=partial(
            _process_filesystem_entry,
            root=root,
            media_filter=media_filter,
        ),
        aggregator=aggregator,
    )

    for file_path in files:
        pool.submit_task(file_path)

    pool.process()

    return aggregator.media_metadata_written


def _collect_media_files(
    *, root: Path, media_filter: MediaFilterConfig, progress: bool = False
) -> list[Path]:
    """Return a sorted list of files to process based on the media filter."""

    consider_all = media_filter.should_consider_all()
    files: list[Path] = []

    progress_bar = None
    if progress:
        from tqdm.auto import tqdm

        progress_bar = tqdm(total=None, unit="file", desc="Collecting media files")

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current_dir = Path(dirpath)

        if current_dir.name == MAIN_FOLDER_NAME:
            dirnames[:] = []
            continue

        dirnames[:] = [d for d in dirnames if d != MAIN_FOLDER_NAME]

        for filename in filenames:
            if progress_bar is not None:
                progress_bar.update()

            if not consider_all and not media_filter.should_consider_media(filename):
                continue

            candidate = current_dir / filename
            if MAIN_FOLDER_NAME in candidate.parts:
                continue

            if candidate.is_file():
                files.append(candidate)

    if progress_bar is not None:
        progress_bar.close()

    files.sort()
    return files


def _process_filesystem_entry(
    file_path: Path | str,
    *,
    root: Path,
    media_filter: MediaFilterConfig,
) -> Iterator[IndexMediaMetadata]:
    file_path = Path(file_path)
    extracted_metadata = media_filter.extract_metadata(file_path)
    if extracted_metadata is None:
        return

    stored_type, metadata_json = serialize_media_metadata(extracted_metadata)
    entry_key = file_path.relative_to(root).as_posix()

    yield IndexMediaMetadata(
        entry_key=entry_key,
        metadata_type=stored_type.value,
        metadata_json=metadata_json,
    )
