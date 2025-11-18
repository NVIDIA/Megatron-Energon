# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from megatron.energon.media.extractor import MediaFilterConfig, extract_metadata
from megatron.energon.media.metadata import serialize_media_metadata


def prepare_filesystem_dataset(
    root_path: EPath,
    media_filter: MediaFilterConfig,
    *,
    progress: bool,
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

    meta_dir = root / MAIN_FOLDER_NAME
    meta_dir.mkdir(exist_ok=True, parents=True)

    sqlite_path = meta_dir / "index.sqlite"

    connection = sqlite3.connect(str(sqlite_path))
    try:
        connection.execute("PRAGMA busy_timeout = 5000;")
        connection.execute("PRAGMA locking_mode = EXCLUSIVE;")
        connection.execute("PRAGMA journal_mode = DELETE;")

        connection.execute("DROP TABLE IF EXISTS samples")
        connection.execute("DROP TABLE IF EXISTS sample_parts")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS media_metadata (
                entry_key TEXT PRIMARY KEY,
                metadata_type TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS media_filters (
                strategy TEXT NOT NULL,
                pattern TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(strategy, pattern)
            )
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO media_filters (strategy, pattern) VALUES (?, ?)",
            (media_filter.strategy.value, media_filter.pattern),
        )

        def iter_dataset_files() -> Iterator[Path]:
            for candidate in root.rglob("*"):
                if not candidate.is_file():
                    continue
                if MAIN_FOLDER_NAME in candidate.parts:
                    continue
                yield candidate

        files_iter = iter_dataset_files()
        iterator = None
        bar = None
        try:
            if progress:
                from tqdm.auto import tqdm

                bar = tqdm(files_iter, unit="file", desc="Scanning media")
                iterator = bar
            else:
                iterator = files_iter

            inserted = 0
            for file_path in iterator:
                metadata_tuple = extract_metadata(file_path, media_filter)
                if metadata_tuple is None:
                    continue

                metadata_type, metadata_obj = metadata_tuple
                stored_type, metadata_json = serialize_media_metadata(metadata_obj)
                assert stored_type == metadata_type, "Metadata type mismatch"

                entry_key = file_path.relative_to(root).as_posix()
                connection.execute(
                    "INSERT OR REPLACE INTO media_metadata (entry_key, metadata_type, metadata_json) VALUES (?, ?, ?)",
                    (entry_key, metadata_type.value, metadata_json),
                )
                inserted += 1
            connection.commit()
            return inserted
        finally:
            if bar is not None:
                bar.close()
    finally:
        connection.close()
