# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import shutil
import struct
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import lmdb

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.config import INDEX_LMDB_MAP_SIZE
from megatron.energon.flavors.webdataset.index_codec import (
    pack_loc_key,
    pack_media_metadata_value,
    pack_part_key,
    pack_part_value,
    pack_sample_value,
    unpack_media_metadata_value,
    unpack_part_value,
    unpack_sample_value,
)
from megatron.energon.flavors.webdataset.indexing import DuplicateSampleKeyError
from megatron.energon.flavors.webdataset.itar import ITarRawSamplePartPointer, ITarSamplePointer

if TYPE_CHECKING:
    from megatron.energon.flavors.webdataset.prepare import (
        IndexMediaMetadata,
        IndexSample,
        IndexSamplePart,
    )

_META_DB = b"__meta__"
_META_VERSION_KEY = b"version"
_META_VERSION = b"1"
_META_SAMPLE_COUNT_KEY = b"sample_count"
_META_TOTAL_SIZE_KEY = b"total_byte_size"
_META_HAS_PARTS_KEY = b"has_sample_parts"
_META_HAS_MEDIA_KEY = b"has_media_metadata"

_DB_SAMPLES_BY_KEY = b"samples_by_key"
_DB_SAMPLES_BY_LOC = b"samples_by_loc"
_DB_PARTS = b"parts"
_DB_MEDIA = b"media_metadata"


def _compact_lmdb_env(source_path: Path, dest_path: Path) -> None:
    """Copy *source_path* to *dest_path* with LMDB compaction (drops free pages)."""
    if dest_path.exists():
        shutil.rmtree(dest_path)
    dest_path.mkdir(parents=True, exist_ok=False)
    env = lmdb.open(str(source_path), readonly=True, lock=False, max_dbs=8)
    try:
        env.copy(str(dest_path), compact=True)
    finally:
        env.close()


class LmdbIndexWriter:
    """LMDB-backed webdataset index writer for prepare-time bulk load."""

    index_path: EPath
    enable_sample_tables: bool
    enable_media_metadata: bool
    _final_path: Path
    _temp_path: Path
    _env: Optional[lmdb.Environment]
    _txn: Optional[lmdb.Transaction]
    _samples_by_key: Optional[lmdb._Database]
    _samples_by_loc: Optional[lmdb._Database]
    _parts: Optional[lmdb._Database]
    _media_metadata: Optional[lmdb._Database]
    _meta: Optional[lmdb._Database]
    _sample_count: int
    _total_byte_size: int
    _media_filters_seen: set[Tuple[str, Optional[str]]]

    def __init__(
        self,
        index_path: EPath,
        *,
        enable_sample_tables: bool = True,
        enable_media_metadata: bool = False,
        reset_tables: bool = True,
        map_size: int = INDEX_LMDB_MAP_SIZE,
    ) -> None:
        self.index_path = index_path
        self.enable_sample_tables = enable_sample_tables
        self.enable_media_metadata = enable_media_metadata
        self.reset_tables = reset_tables
        self._map_size = map_size
        self._env = None
        self._txn = None
        self._samples_by_key = None
        self._samples_by_loc = None
        self._parts = None
        self._media_metadata = None
        self._meta = None
        self._sample_count = 0
        self._total_byte_size = 0
        self._media_filters_seen = set()

        self._final_path = self.index_path.local_path()
        self._final_path.parent.mkdir(parents=True, exist_ok=True)
        self._temp_path = self._final_path.with_name(self._final_path.name + ".tmp")
        if self._temp_path.exists():
            shutil.rmtree(self._temp_path)

        self._env = lmdb.open(
            str(self._temp_path),
            map_size=self._map_size,
            max_dbs=8,
            writemap=True,
            sync=False,
            metasync=False,
        )
        self._meta = self._env.open_db(_META_DB, create=True)
        if self.enable_sample_tables:
            assert self.reset_tables, "Reset tables is required when enabling sample tables"
            self._samples_by_key = self._env.open_db(_DB_SAMPLES_BY_KEY, create=True)
            self._samples_by_loc = self._env.open_db(_DB_SAMPLES_BY_LOC, create=True)
            self._parts = self._env.open_db(_DB_PARTS, create=True)
        if self.enable_media_metadata:
            self._media_metadata = self._env.open_db(_DB_MEDIA, create=True)

        self._txn = self._env.begin(write=True)
        self._txn.put(_META_VERSION_KEY, _META_VERSION, db=self._meta)

    def append_samples(self, rows: Sequence["IndexSample"]) -> None:
        assert self._txn is not None, "Index is closed"
        assert self._samples_by_key is not None

        if len(rows) == 0:
            return

        duplicate_key = self._find_duplicate_in_batch(rows)
        if duplicate_key is not None:
            raise DuplicateSampleKeyError(duplicate_key)

        for row in rows:
            key = row.sample_key.encode("utf-8")
            value = pack_sample_value(
                tar_file_id=row.tar_file_id,
                sample_index=row.sample_index,
                byte_offset=row.byte_offset,
                byte_size=row.byte_size,
            )
            if not self._txn.put(key, value, db=self._samples_by_key, overwrite=False):
                raise DuplicateSampleKeyError(row.sample_key)
            loc_key = pack_loc_key(row.tar_file_id, row.sample_index)
            self._txn.put(loc_key, key, db=self._samples_by_loc)
            self._sample_count += 1
            self._total_byte_size += row.byte_size

    def append_parts(self, rows: Sequence["IndexSamplePart"]) -> None:
        assert self._txn is not None, "Index is closed"
        assert self._parts is not None

        for row in rows:
            key = pack_part_key(row.tar_file_id, row.sample_index, row.part_name)
            value = pack_part_value(row.content_byte_offset, row.content_byte_size)
            self._txn.put(key, value, db=self._parts)

    def append_media_metadata_batch(self, rows: Sequence["IndexMediaMetadata"]) -> None:
        assert self.enable_media_metadata
        assert self._txn is not None
        assert self._media_metadata is not None

        for row in rows:
            key = row.entry_key.encode("utf-8")
            value = pack_media_metadata_value(row.metadata_type, row.metadata_json)
            self._txn.put(key, value, db=self._media_metadata)

    def append_media_filter(self, *, strategy: str, patterns: str | None) -> None:
        assert self._txn is not None
        assert self._meta is not None
        key = (strategy + "\0" + (patterns or "")).encode("utf-8")
        filter_key = b"filter:" + key
        if filter_key in self._media_filters_seen:
            return
        self._media_filters_seen.add((strategy, patterns))
        self._txn.put(filter_key, b"1", db=self._meta)

    def close(self) -> None:
        assert self._txn is not None
        assert self._env is not None
        assert self._meta is not None

        if self.enable_sample_tables:
            self._txn.put(
                _META_SAMPLE_COUNT_KEY,
                struct.pack(">Q", self._sample_count),
                db=self._meta,
            )
            self._txn.put(
                _META_TOTAL_SIZE_KEY,
                struct.pack(">Q", self._total_byte_size),
                db=self._meta,
            )
            self._txn.put(_META_HAS_PARTS_KEY, b"1", db=self._meta)

        if self.enable_media_metadata:
            self._txn.put(_META_HAS_MEDIA_KEY, b"1", db=self._meta)

        self._txn.commit()
        self._txn = None
        self._env.sync(True)
        self._env.close()
        self._env = None

        compact_path = self._temp_path.with_name(self._temp_path.name + ".compact")
        try:
            _compact_lmdb_env(self._temp_path, compact_path)
        finally:
            if self._temp_path.exists():
                shutil.rmtree(self._temp_path)

        if self._final_path.exists():
            shutil.rmtree(self._final_path)
        shutil.move(str(compact_path), str(self._final_path))

    def _abort(self) -> None:
        if self._txn is not None:
            self._txn.abort()
            self._txn = None
        if self._env is not None:
            self._env.close()
            self._env = None
        if self._temp_path.exists():
            shutil.rmtree(self._temp_path)
        compact_path = self._temp_path.with_name(self._temp_path.name + ".compact")
        if compact_path.exists():
            shutil.rmtree(compact_path)

    def __enter__(self) -> "LmdbIndexWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self._abort()

    @staticmethod
    def _find_duplicate_in_batch(rows: Sequence["IndexSample"]) -> str | None:
        seen_in_batch: set[str] = set()
        for row in rows:
            if row.sample_key in seen_in_batch:
                return row.sample_key
            seen_in_batch.add(row.sample_key)
        return None


class LmdbIndexReader:
    """Read-only LMDB webdataset index.

    Uses one shared ``Environment``; read transactions are created per operation and
    may be used concurrently from multiple threads.
    """

    index_path: EPath
    _env: lmdb.Environment
    _meta: lmdb._Database
    _samples_by_key: Optional[lmdb._Database]
    _samples_by_loc: Optional[lmdb._Database]
    _parts: Optional[lmdb._Database]
    _media_metadata: Optional[lmdb._Database]
    _has_sample_tables: bool
    _has_media_metadata: bool
    _sample_count: int
    _total_byte_size: int

    def __init__(self, index_path: EPath) -> None:
        if not index_path.is_local():
            raise ValueError(f"LMDB index must be on local storage: {index_path}")
        self.index_path = index_path
        local_path = index_path.local_path()
        if not local_path.is_dir():
            raise FileNotFoundError(f"LMDB index directory not found: {local_path}")

        self._env = lmdb.open(
            str(local_path),
            readonly=True,
            lock=False,
            max_dbs=8,
            map_size=INDEX_LMDB_MAP_SIZE,
        )
        self._meta = self._env.open_db(_META_DB)
        with self._env.begin() as txn:
            version = txn.get(_META_VERSION_KEY, db=self._meta)
            if version != _META_VERSION:
                raise ValueError(f"Unsupported LMDB index version: {version!r}")

            has_parts = txn.get(_META_HAS_PARTS_KEY, db=self._meta)
            has_media = txn.get(_META_HAS_MEDIA_KEY, db=self._meta)
            self._has_sample_tables = has_parts is not None
            self._has_media_metadata = has_media is not None

            count_raw = txn.get(_META_SAMPLE_COUNT_KEY, db=self._meta)
            size_raw = txn.get(_META_TOTAL_SIZE_KEY, db=self._meta)
            self._sample_count = struct.unpack(">Q", count_raw)[0] if count_raw is not None else 0
            self._total_byte_size = struct.unpack(">Q", size_raw)[0] if size_raw is not None else 0

        self._samples_by_key = None
        self._samples_by_loc = None
        self._parts = None
        self._media_metadata = None
        if self._has_sample_tables:
            self._samples_by_key = self._env.open_db(_DB_SAMPLES_BY_KEY)
            self._samples_by_loc = self._env.open_db(_DB_SAMPLES_BY_LOC)
            self._parts = self._env.open_db(_DB_PARTS)
        if self._has_media_metadata:
            self._media_metadata = self._env.open_db(_DB_MEDIA)

    def db_has_sample_parts(self) -> bool:
        return self._has_sample_tables

    def db_has_media_metadata(self) -> bool:
        return self._has_media_metadata

    def list_all_samples(self) -> Generator[Tuple[str, int, int], None, None]:
        assert self._samples_by_key is not None
        with self._env.begin() as txn:
            cursor = txn.cursor(db=self._samples_by_key)
            for key, value in cursor:
                tar_file_id, _, _, byte_size = unpack_sample_value(value)
                yield key.decode("utf-8"), byte_size, tar_file_id

    def list_all_sample_parts(self) -> Generator[Tuple[str, int, int], None, None]:
        assert self._parts is not None
        assert self._samples_by_loc is not None
        with self._env.begin() as txn:
            cursor = txn.cursor(db=self._parts)
            for key, value in cursor:
                tar_file_id, sample_index = struct.unpack(">ii", key[:8])
                loc_key = pack_loc_key(tar_file_id, sample_index)
                sample_key_raw = txn.get(loc_key, db=self._samples_by_loc)
                if sample_key_raw is None:
                    continue
                part_name = key[8:].decode("utf-8")
                _, content_byte_size = unpack_part_value(value)
                full_key = f"{sample_key_raw.decode('utf-8')}.{part_name}"
                yield full_key, content_byte_size, tar_file_id

    def list_sample_parts(self, sample_key: str) -> Generator[Tuple[str, int, int], None, None]:
        tar_file_id, sample_index = self._resolve_sample_location(sample_key)
        prefix = struct.pack(">ii", tar_file_id, sample_index)
        assert self._parts is not None
        with self._env.begin() as txn:
            cursor = txn.cursor(db=self._parts)
            if not cursor.set_range(prefix):
                return
            for key, value in cursor:
                if not key.startswith(prefix):
                    break
                part_name = key[8:].decode("utf-8")
                _, content_byte_size = unpack_part_value(value)
                yield part_name, content_byte_size, tar_file_id

    def get_total_size(self) -> int:
        return self._total_byte_size

    def get_sample_count(self) -> int:
        return self._sample_count

    def _resolve_sample_location(self, sample_key: str) -> Tuple[int, int]:
        assert self._samples_by_key is not None
        with self._env.begin() as txn:
            raw = txn.get(sample_key.encode("utf-8"), db=self._samples_by_key)
        if raw is None:
            raise KeyError(f"Sample key not found: {sample_key}")
        tar_file_id, sample_index, _, _ = unpack_sample_value(raw)
        return tar_file_id, sample_index

    def get_sample_part(self, key: str, part_name: str) -> ITarRawSamplePartPointer:
        tar_file_id, sample_index = self._resolve_sample_location(key)
        part_key = pack_part_key(tar_file_id, sample_index, part_name)
        assert self._parts is not None
        with self._env.begin() as txn:
            raw = txn.get(part_key, db=self._parts)
        if raw is None:
            raise KeyError(
                f"Sample part not found: key={key}, part_name={part_name} in {self.index_path}"
            )
        content_byte_offset, content_byte_size = unpack_part_value(raw)
        return ITarRawSamplePartPointer(
            tar_file_id=tar_file_id,
            raw_byte_offset=content_byte_offset,
            raw_byte_size=content_byte_size,
        )

    def get_sample_pointer_by_key(self, key: str) -> ITarSamplePointer:
        assert self._samples_by_key is not None
        with self._env.begin() as txn:
            raw = txn.get(key.encode("utf-8"), db=self._samples_by_key)
        if raw is None:
            raise KeyError(f"Sample key not found: {key}")
        tar_file_id, _, byte_offset, byte_size = unpack_sample_value(raw)
        return ITarSamplePointer(
            tar_file_id=tar_file_id,
            byte_offset=byte_offset,
            byte_size=byte_size,
        )

    def get_media_metadata(self, entry_key: str) -> Tuple[str, str] | None:
        assert self._media_metadata is not None
        with self._env.begin() as txn:
            raw = txn.get(entry_key.encode("utf-8"), db=self._media_metadata)
        if raw is None:
            return None
        return unpack_media_metadata_value(raw)

    def close(self) -> None:
        self._env.close()

    def __enter__(self) -> "LmdbIndexReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
