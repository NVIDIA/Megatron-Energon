# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Iterable, Iterator

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.aggregator_pool import AggregatorPool
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.prepare import (
    IndexMediaMetadata,
    SqliteIndexWriterAggregator,
)
from megatron.energon.media.extractor import (
    MediaFilterConfig,
    MediaFilterStrategy,
    extract_metadata,
    should_consider_media,
)
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

    sqlite_path = EPath(meta_dir / "index.sqlite")

    aggregator = SqliteIndexWriterAggregator(
        sqlite_path,
        total_tasks=len(files),
        progress_fn=None,
        enable_media_metadata=True,
        media_filter=media_filter,
        reset_tables=False,
        enable_sample_tables=False,
    )

    pool = AggregatorPool[
        Path,
        IndexMediaMetadata,
        tuple,
    ](
        num_workers=min(workers, len(files)) or 1,
        user_produce_data=partial(
            _process_filesystem_entry,
            root=root,
            media_filter=media_filter,
        ),
        aggregator=aggregator,
    )

    if progress:
        from tqdm.auto import tqdm

        def progress_fn(
            iterable: Iterable[Path],
            length: int | None = None,
        ) -> Iterator[Path]:
            yield from tqdm(
                iterable,
                total=length,
                unit="file",
                desc="Scheduling media files",
            )
    else:

        def progress_fn(
            iterable: Iterable[Path],
            length: int | None = None,
        ) -> Iterator[Path]:
            yield from iterable

    for file_path in progress_fn(files, len(files) or None):
        pool.submit_task(file_path)

    pool.process()

    return aggregator.media_metadata_written


def _collect_media_files(
    *, root: Path, media_filter: MediaFilterConfig, progress: bool = False
) -> list[Path]:
    """Return a sorted list of files to process based on the media filter."""

    consider_all = media_filter.strategy == MediaFilterStrategy.TYPE
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

            if not consider_all and not should_consider_media(filename, media_filter):
                continue

            candidate = current_dir / filename
            if MAIN_FOLDER_NAME in candidate.parts:
                continue

            if candidate.is_file():
                files.append(candidate)

    if progress_bar is not None:
        progress_bar.close()

    return files


def _process_filesystem_entry(
    file_path: str,
    *,
    root: Path,
    media_filter: MediaFilterConfig,
) -> Iterator[IndexMediaMetadata]:
    file_path = Path(file_path)
    metadata_tuple = extract_metadata(file_path, media_filter)
    if metadata_tuple is None:
        return

    metadata_type, metadata_obj = metadata_tuple
    stored_type, metadata_json = serialize_media_metadata(metadata_obj)
    entry_key = file_path.relative_to(root).as_posix()

    yield IndexMediaMetadata(
        entry_key=entry_key,
        metadata_type=stored_type.value,
        metadata_json=metadata_json,
    )
