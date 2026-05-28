# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import shutil
import tempfile
from functools import partial
from pathlib import Path
from typing import Callable, Iterator

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.aggregator_pool import AggregatorPool
from megatron.energon.flavors.webdataset.config import (
    INDEX_BATCH_SIZE,
    INDEX_SQLITE_FILENAME,
    MAIN_FOLDER_NAME,
)
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
    index_sqlite_tmp_path: Path | None = None,
) -> int:
    """Scan a filesystem dataset and materialize media metadata into SQLite.

    Args:
        root_path: Dataset root directory.
        media_filter: Media filtering configuration.
        progress: Whether to display a tqdm progress bar.
        index_sqlite_tmp_path: When ``root_path`` is remote, temp file path used to build
            ``index.sqlite`` locally before upload. If omitted, a new directory under
            ``/tmp`` is created and removed after a successful run.

    Returns:
        Number of metadata entries written to the database.
    """

    assert not root_path.is_file(), f"Expected directory for filesystem dataset, got {root_path}"

    files = _collect_media_files(root=root_path, media_filter=media_filter, progress=progress)

    if len(files) == 0:
        raise ValueError("No media files found to process")

    owns_remote_sqlite_tmp = False
    remote_sqlite_tmp_dir: Path | None = None
    if not root_path.is_local():
        if index_sqlite_tmp_path is None:
            remote_sqlite_tmp_dir = Path(
                tempfile.mkdtemp(dir="/tmp", prefix="energon-prepare-media-")
            )
            index_sqlite_tmp_path = remote_sqlite_tmp_dir / INDEX_SQLITE_FILENAME
            owns_remote_sqlite_tmp = True
    else:
        index_sqlite_tmp_path = None

    agg_progress_fn: Callable[[Iterator[int], int], Iterator[int]] | None = None
    if progress:
        from tqdm.auto import tqdm

        def agg_progress_fn(iterator: Iterator[int], total: int) -> Iterator[int]:
            with tqdm(iterator, total=total, unit="file", desc="Processing media files") as bar:
                yield from bar

    sqlite_path = root_path / MAIN_FOLDER_NAME / INDEX_SQLITE_FILENAME

    try:
        aggregator = SqliteIndexWriterAggregator(
            sqlite_path,
            total_tasks=len(files),
            progress_fn=agg_progress_fn,
            enable_media_metadata=True,
            media_filter=media_filter,
            reset_tables=False,
            enable_sample_tables=False,
            progress_on_media=progress,
            sqlite_local_build_path=index_sqlite_tmp_path,
        )

        pool = AggregatorPool[
            EPath,
            IndexAggregatable,
            tuple[list[ShardInfo], set[str], bool, list[tuple[str, int]]],
        ](
            num_workers=min(workers, len(files)) or 1,
            user_produce_data=partial(
                _process_filesystem_entry,
                root=root_path,
                media_filter=media_filter,
            ),
            aggregator=aggregator,
            batch_size=INDEX_BATCH_SIZE,
        )

        for file_path in files:
            pool.submit_task(file_path)

        pool.process()

        if sqlite_path.is_local():
            try:
                meta_dir = (root_path / MAIN_FOLDER_NAME).local_path()
                # Copy group permissions from the parent dir
                meta_dir.chmod((root_path.local_path().stat().st_mode | 0o700))
                # Just read/write, no execute
                sqlite_path.local_path().chmod(
                    (root_path.local_path().stat().st_mode | 0o600) & 0o666
                )
            except OSError:
                pass

        return aggregator.media_metadata_written
    finally:
        if owns_remote_sqlite_tmp and remote_sqlite_tmp_dir is not None:
            shutil.rmtree(remote_sqlite_tmp_dir, ignore_errors=True)


def _collect_media_files(
    *, root: EPath, media_filter: MediaFilterConfig, progress: bool = False
) -> list[EPath]:
    """Return a sorted list of files to process based on the media filter."""

    consider_all = media_filter.should_consider_all()
    files: list[EPath] = []

    progress_bar = None
    if progress:
        from tqdm.auto import tqdm

        progress_bar = tqdm(total=None, unit="file", desc="Collecting media files")

    # if root.is_local() and not root.profile == "dss":
    #     paths = (
    #         EPath(path) / file
    #         for path, _dirs, files in os.walk(root.local_path(), followlinks=False)
    #         for file in files
    #     )
    # else:
    #     paths = root.glob("**/*")

    for file in root.walk():
        if progress_bar is not None:
            progress_bar.update()

        if ("/" + MAIN_FOLDER_NAME + "/") in file.url:
            continue

        if not consider_all and not media_filter.should_consider_media(file.name):
            continue

        files.append(file)

    if progress_bar is not None:
        progress_bar.close()

    files.sort()
    return files


def _process_filesystem_entry(
    file_path: EPath,
    *,
    root: EPath,
    media_filter: MediaFilterConfig,
) -> Iterator[IndexMediaMetadata]:
    extracted_metadata = media_filter.extract_metadata(file_path)
    if extracted_metadata is None:
        return

    stored_type, metadata_json = serialize_media_metadata(extracted_metadata)
    entry_key = file_path.relative_to(root)

    yield IndexMediaMetadata(
        entry_key=entry_key,
        metadata_type=stored_type.value,
        metadata_json=metadata_json,
    )
