# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import shutil
from collections.abc import Generator, Sequence
from typing import Protocol, Tuple, runtime_checkable

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.config import (
    INDEX_LMDB_DIRNAME,
    INDEX_SQLITE_FILENAME,
    MAIN_FOLDER_NAME,
)
from megatron.energon.flavors.webdataset.itar import ITarRawSamplePartPointer, ITarSamplePointer
from megatron.energon.flavors.webdataset.lmdb_index import LmdbIndexReader, LmdbIndexWriter
from megatron.energon.flavors.webdataset.indexing import SqliteIndexReader, SqliteIndexWriter

if False:  # TYPE_CHECKING
    from megatron.energon.flavors.webdataset.prepare import IndexMediaMetadata, IndexSample, IndexSamplePart


@runtime_checkable
class IndexWriter(Protocol):
    def append_samples(self, rows: Sequence["IndexSample"]) -> None: ...

    def append_parts(self, rows: Sequence["IndexSamplePart"]) -> None: ...

    def append_media_metadata_batch(self, rows: Sequence["IndexMediaMetadata"]) -> None: ...

    def append_media_filter(self, *, strategy: str, patterns: str | None) -> None: ...

    def close(self) -> None: ...

    def __enter__(self) -> "IndexWriter": ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...


@runtime_checkable
class IndexReader(Protocol):
    index_path: EPath

    def db_has_sample_parts(self) -> bool: ...

    def db_has_media_metadata(self) -> bool: ...

    def list_all_samples(self) -> Generator[Tuple[str, int, int], None, None]: ...

    def list_all_sample_parts(self) -> Generator[Tuple[str, int, int], None, None]: ...

    def list_sample_parts(self, sample_key: str) -> Generator[Tuple[str, int, int], None, None]: ...

    def get_total_size(self) -> int: ...

    def get_sample_count(self) -> int: ...

    def get_sample_part(self, key: str, part_name: str) -> ITarRawSamplePartPointer: ...

    def get_sample_pointer_by_key(self, key: str) -> ITarSamplePointer: ...

    def get_media_metadata(self, entry_key: str) -> Tuple[str, str] | None: ...

    def close(self) -> None: ...

    def __enter__(self) -> "IndexReader": ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...


def meta_dir_path(dataset_or_meta_path: EPath) -> EPath:
    """Resolve `.nv-meta` directory from dataset root or meta path."""
    if dataset_or_meta_path.name == MAIN_FOLDER_NAME:
        return dataset_or_meta_path
    return dataset_or_meta_path / MAIN_FOLDER_NAME


def has_prepared_index(meta_dir: EPath) -> bool:
    return (meta_dir / INDEX_LMDB_DIRNAME).is_dir() or (
        meta_dir / INDEX_SQLITE_FILENAME
    ).is_file()


def index_lmdb_path(meta_dir: EPath) -> EPath:
    return meta_dir / INDEX_LMDB_DIRNAME


def index_sqlite_path(meta_dir: EPath) -> EPath:
    return meta_dir / INDEX_SQLITE_FILENAME


def remove_prepared_index(meta_dir: EPath) -> None:
    """Remove prepared index artifacts (LMDB directory and legacy SQLite file)."""
    lmdb_path = index_lmdb_path(meta_dir)
    if lmdb_path.is_dir():
        shutil.rmtree(lmdb_path.local_path(), ignore_errors=True)
    temp_lmdb = lmdb_path.local_path().with_name(lmdb_path.local_path().name + ".tmp")
    if temp_lmdb.exists():
        shutil.rmtree(temp_lmdb, ignore_errors=True)
    sqlite_path = index_sqlite_path(meta_dir)
    if sqlite_path.is_file():
        sqlite_path.unlink()


def open_index_writer(
    meta_dir: EPath,
    *,
    enable_sample_tables: bool = True,
    enable_media_metadata: bool = False,
    reset_tables: bool = True,
) -> IndexWriter:
    """Create a new LMDB index under ``meta_dir/index.lmdb``."""
    return LmdbIndexWriter(
        index_lmdb_path(meta_dir),
        enable_sample_tables=enable_sample_tables,
        enable_media_metadata=enable_media_metadata,
        reset_tables=reset_tables,
    )


def open_index_reader(dataset_path: EPath) -> IndexReader:
    """Open prepared index for a dataset (LMDB preferred, SQLite legacy fallback)."""
    meta_dir = meta_dir_path(dataset_path)
    lmdb_path = index_lmdb_path(meta_dir)
    if lmdb_path.is_dir():
        return LmdbIndexReader(lmdb_path)
    sqlite_path = index_sqlite_path(meta_dir)
    if sqlite_path.is_file():
        return SqliteIndexReader(sqlite_path)
    raise FileNotFoundError(
        f"No prepared index found under {meta_dir}. "
        f"Expected {INDEX_LMDB_DIRNAME}/ or {INDEX_SQLITE_FILENAME}. Run `energon prepare`."
    )
