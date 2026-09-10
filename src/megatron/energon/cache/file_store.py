# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import re
from typing import Any, Optional, Union

from megatron.energon.cache.base import FileStore, FileStoreDecoder, FileStoreWrapper
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.common.manifest.paths import INDEX_SQLITE_FILENAME, MAIN_FOLDER_NAME
from megatron.energon.flavors.common.sqlite import ThreadLocalSqlite
from megatron.energon.local_copy import ensure_local_copy
from megatron.energon.media.metadata import MediaMetadataBase, deserialize_media_metadata
from megatron.energon.source_info import SourceInfo, add_source_info


class DecodeFileStore(FileStoreWrapper[Any]):
    """Wrap a FileStore and decode data on access."""

    def __init__(self, inner: FileStore[bytes], *, decoder: FileStoreDecoder):
        super().__init__(inner)
        self.decoder = decoder

    def __getitem__(self, fname: str) -> tuple[Any, SourceInfo]:
        data, source_info = self._inner[fname]
        return self.decoder.decode(fname, data), source_info

    def _decode_raw(self, data: bytes, **kwargs) -> Any:
        fname = kwargs["fname"]
        return self.decoder.decode(fname, self._inner._decode_raw(data, **kwargs))

    def get_path(self) -> str:
        return self._inner.get_path()

    def __str__(self):
        return f"DecodeFileStore(inner={self._inner}, decoder={self.decoder})"

    def get_media_metadata(self, key: str) -> MediaMetadataBase:
        return self._inner.get_media_metadata(key)


class SystemFileStore(FileStore[bytes]):
    """A FileStore that reads files directly from the file system."""

    def __init__(self, base_dir: Optional[Union[EPath, str]] = None):
        self.base_dir = EPath(base_dir) if base_dir is not None else None
        self._media_metadata_reader: Optional[ThreadLocalSqlite] = None
        self._media_metadata_checked = False

    def __getitem__(self, key: str) -> tuple[bytes, SourceInfo]:
        file_path = EPath(key) if self.base_dir is None else self.base_dir / key
        with file_path.open("rb") as f:
            data = f.read()

        return data, SourceInfo(
            dataset_path=self.base_dir,
            index=key,
            shard_name=str(self.base_dir),
            file_names=(key,),
        )

    def get_path(self) -> str:
        return str(self.base_dir)

    def __str__(self):
        return f"SystemFileStore(base_dir={self.base_dir})"

    def get_media_metadata(self, key: str) -> MediaMetadataBase:
        if self.base_dir is None:
            raise RuntimeError("Media metadata requires a base directory for SystemFileStore")

        reader = self._ensure_media_metadata_reader()
        row = reader.select_one(
            "SELECT metadata_type, metadata_json FROM media_metadata WHERE entry_key = ?",
            (key,),
        )
        if row is None:
            file_path = self.base_dir / key
            if file_path.is_file():
                raise KeyError(
                    f"Media metadata missing for {key}. "
                    "Run `energon prepare --media-metadata-by-...` to regenerate it."
                )
            raise KeyError(f"File {file_path} not found")
        metadata_type, metadata_json = row
        return deserialize_media_metadata(metadata_type, metadata_json)

    def _ensure_media_metadata_reader(self) -> ThreadLocalSqlite:
        assert self.base_dir is not None
        if self._media_metadata_reader is None:
            sqlite_path = self.base_dir / MAIN_FOLDER_NAME / INDEX_SQLITE_FILENAME

            if not sqlite_path.is_file():
                raise RuntimeError(
                    f"Media metadata database missing at {sqlite_path}. "
                    "Run `energon prepare --media-metadata-by-...` for this dataset."
                )

            local_sqlite_path = ensure_local_copy(sqlite_path)
            db_uri = f"file:{str(local_sqlite_path)}?mode=ro&immutable=1"
            self._media_metadata_reader = ThreadLocalSqlite(db_uri, is_uri=True)

        if not self._media_metadata_checked:
            assert self._media_metadata_reader is not None
            exists = self._media_metadata_reader.select_one(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='media_metadata'"
            )
            if exists is None:
                self._media_metadata_reader.thread_close()
                self._media_metadata_reader = None
                raise RuntimeError(
                    "Media metadata table missing. Re-run `energon prepare --media-metadata-by-...`."
                )
            self._media_metadata_checked = True

        return self._media_metadata_reader


class ByteRangeStore(FileStore[bytes]):
    """A FileStore that reads byte ranges from files under a root path."""

    _KEY_SEPARATOR = "#bytes="
    _RE_KEY = re.compile(r"^(?P<path>.+)#bytes=(?P<offset>\d+):(?P<size>\d+)$")

    def __init__(self, root: Union[EPath, str]):
        self.root = EPath(root)

    def __getitem__(self, key: str) -> tuple[bytes, SourceInfo]:
        path, byte_offset, byte_size = self._parse_key(key)
        data, source_info = self._read_range_with_source(path, byte_offset, byte_size)
        return data, source_info

    def read_range(
        self,
        path: str,
        byte_offset: int,
        byte_size: int,
        *,
        sample: object | None = None,
    ) -> bytes:
        data, source_info = self._read_range_with_source(path, byte_offset, byte_size)
        if sample is not None:
            add_source_info(sample, source_info)
        return data

    def get_path(self) -> str:
        return str(self.root)

    def __str__(self):
        return f"ByteRangeStore(root={self.root})"

    @classmethod
    def _make_key(cls, path: str, byte_offset: int, byte_size: int) -> str:
        return f"{path}{cls._KEY_SEPARATOR}{byte_offset}:{byte_size}"

    @classmethod
    def _parse_key(cls, key: str) -> tuple[str, int, int]:
        match = cls._RE_KEY.match(key)
        if match is None:
            raise ValueError(f"Invalid byte-range key {key!r}. Expected 'path#bytes=offset:size'.")
        path = match.group("path")
        byte_offset = int(match.group("offset"))
        byte_size = int(match.group("size"))
        return path, byte_offset, byte_size

    def _read_range_with_source(
        self,
        path: str,
        byte_offset: int,
        byte_size: int,
    ) -> tuple[bytes, SourceInfo]:
        assert byte_offset >= 0, f"byte_offset must be non-negative, got {byte_offset}"
        assert byte_size >= 0, f"byte_size must be non-negative, got {byte_size}"
        file_path = self._resolve_path(path)
        if byte_size == 0:
            data = b""
        else:
            with file_path.open("rb") as f:
                f.seek(byte_offset)
                data = f.read(byte_size)

            if len(data) != byte_size:
                raise IOError(
                    f"Byte-range read from {file_path} returned {len(data)} bytes, "
                    f"expected {byte_size} bytes"
                )

        key = self._make_key(path, byte_offset, byte_size)
        return data, SourceInfo(
            dataset_path=self.root,
            index=key,
            shard_name=str(file_path),
            file_names=(key,),
        )

    def _resolve_path(self, path: str) -> EPath:
        file_path = self.root / path
        if file_path.profile != self.root.profile:
            raise ValueError(
                f"Byte-range path {path!r} resolves outside root {self.root}: {file_path}"
            )
        try:
            file_path.internal_path.relative_to(self.root.internal_path)
        except ValueError as exc:
            raise ValueError(
                f"Byte-range path {path!r} resolves outside root {self.root}: {file_path}"
            ) from exc
        return file_path


def __getattr__(name: str):
    if name == "WebdatasetFileStore":
        from megatron.energon.flavors.webdataset.file_store import WebdatasetFileStore

        return WebdatasetFileStore
    if name == "JsonlFileStore":
        from megatron.energon.flavors.jsonl.file_store import JsonlFileStore

        return JsonlFileStore
    if name == "BinIdxFileStore":
        from megatron.energon.flavors.binidx.file_store import BinIdxFileStore

        return BinIdxFileStore
    if name == "ParquetFileStore":
        from megatron.energon.flavors.parquet.file_store import ParquetFileStore

        return ParquetFileStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ByteRangeStore",
    "DecodeFileStore",
    "SystemFileStore",
]
