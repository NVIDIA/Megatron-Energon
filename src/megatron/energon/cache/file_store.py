# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import sqlite3
from typing import Any, Callable, Optional, Union

from megatron.energon.cache.base import FileStore, FileStoreDecoder, FileStoreWrapper
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.binidx.binidx_reader import BinIdxReader
from megatron.energon.flavors.jsonl.ijsonl_reader import IJsonlReader
from megatron.energon.flavors.parquet.reader import IParquetReader
from megatron.energon.flavors.webdataset.config import INDEX_SQLITE_FILENAME, MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.itar_reader import SqliteITarEntryReader
from megatron.energon.flavors.webdataset.thread_local_sqlite import ThreadLocalSqlite
from megatron.energon.local_copy import ensure_local_copy
from megatron.energon.media.metadata import MediaMetadataBase, deserialize_media_metadata
from megatron.energon.source_info import SourceInfo


class DecodeFileStore(FileStoreWrapper[Any]):
    """Used to wrap a FileStore and decode the data on access."""

    def __init__(
        self,
        inner: FileStore[bytes],
        *,
        decoder: FileStoreDecoder,
    ):
        """
        Args:
            inner: The FileStore to wrap.
            decoder: The decoder to apply to every item read from the FileStore.
        """

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
        """
        Args:
            base_dir: The base directory to use for relative paths. If None, you should only pass
                absolute paths to __getitem__.
        """

        self.base_dir = EPath(base_dir) if base_dir is not None else None
        self._media_metadata_reader: Optional[ThreadLocalSqlite] = None
        self._media_metadata_checked = False

    def __getitem__(self, key: str) -> tuple[bytes, SourceInfo]:
        # Construct the full path from the dataset path and the file key
        if self.base_dir is None:
            file_path = EPath(key)
        else:
            file_path = self.base_dir / key

        # Read and return the file contents as bytes
        with file_path.open("rb") as f:
            data = f.read()

        return data, SourceInfo(
            dataset_path=self.base_dir,
            index=key,
            shard_name=str(self.base_dir),
            file_names=(key,),
        )

    def get_path(self) -> str:
        """Returns the path to the dataset."""
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


class WebdatasetFileStore(SqliteITarEntryReader, FileStore[bytes]):
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset."""

    def __init__(
        self,
        dataset_path: EPath,
    ):
        super().__init__(
            base_path=dataset_path,
            key_is_full_entryname=True,
            disable_cache=True,
        )
        self._media_metadata_available: Optional[bool] = None

    def get_path(self) -> str:
        return str(self.base_path)

    def get_media_metadata(self, key: str) -> MediaMetadataBase:
        if self._media_metadata_available is None:
            try:
                self._media_metadata_available = self.sqlite_reader.db_has_media_metadata()
            except sqlite3.Error as exc:  # pragma: no cover - defensive
                self._media_metadata_available = False
                raise RuntimeError(
                    "Failed to inspect media metadata table. Re-run `energon prepare --media-metadata-by-...`."
                ) from exc

        if not self._media_metadata_available:
            raise RuntimeError(
                "Media metadata is not available for this dataset. "
                "Run `energon prepare --media-metadata-by-...` to generate it."
            )

        try:
            row = self.sqlite_reader.get_media_metadata(key)
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "Failed to load media metadata. Re-run `energon prepare --media-metadata-by-...`."
            ) from exc

        if row is None:
            raise KeyError(f"Sample {key!r} not found")

        metadata_type, metadata_json = row
        return deserialize_media_metadata(metadata_type, metadata_json)


class JsonlFileStore(IJsonlReader, FileStore[bytes]):
    """This dataset will directly read entries from a jsonl file."""

    def get_path(self) -> str:
        return str(self.jsonl_path)


class BinIdxFileStore(BinIdxReader, FileStore[bytes]):
    """This dataset will directly read entries from a bin-idx file pair."""

    def get_path(self) -> str:
        return str(self.bin_path)


class ParquetFileStore(IParquetReader, FileStore[Any]):
    """Random access to rows of a Parquet dataset directory (layout from Parquet footers).

    Per-column keys ``{row_index}.{column}`` return native pyarrow ``as_py()`` values, not bytes.
    """

    def __init__(
        self,
        dataset_root: EPath | str,
        *,
        part_filter: Callable[[str], bool] | None = None,
        parquet_file_cache_size: int = 5,
    ):
        from megatron.energon.flavors.parquet.prepare import (
            assert_layout_columns_subset,
            scan_parquet_dataset,
        )

        root = EPath(dataset_root)
        layout = scan_parquet_dataset(root)
        layout_cols = list(layout.columns)
        if part_filter is None:
            read_columns = layout_cols
        else:
            read_columns = [c for c in layout_cols if part_filter(c)]
        if not read_columns:
            raise ValueError(
                "part_filter excluded all Parquet columns; nothing to load. "
                f"Layout columns: {layout_cols}"
            )
        assert_layout_columns_subset(layout_cols, read_columns)
        super().__init__(
            root,
            layout,
            read_columns,
            parquet_file_cache_size=parquet_file_cache_size,
        )

    def get_path(self) -> str:
        return str(self.dataset_root)
