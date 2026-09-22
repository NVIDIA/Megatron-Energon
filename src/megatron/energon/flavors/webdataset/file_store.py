# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import sqlite3
from typing import Optional

from megatron.energon.cache.base import FileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.itar_reader import SqliteITarEntryReader
from megatron.energon.media.metadata import MediaMetadataBase, deserialize_media_metadata


class WebdatasetFileStore(SqliteITarEntryReader, FileStore[bytes]):
    """Random access to files inside a prepared WebDataset tar dataset."""

    def __init__(self, dataset_path: EPath):
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
