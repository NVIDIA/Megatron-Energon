# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional, Union

from megatron.energon.cache.base import FileStore, FileStoreDecoder
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.itar_reader import SqliteITarEntryReader
from megatron.energon.source_info import SourceInfo


class DecodeFileStore(FileStore[Any]):
    """Used to wrap a FileStore and decode the data on access."""

    def __init__(
        self,
        inner_reader: FileStore[bytes],
        *,
        decoder: FileStoreDecoder,
    ):
        """
        Args:
            inner_reader: The FileStore to wrap.
            decoder: The decoder to apply to every item read from the FileStore.
        """

        self.inner_reader = inner_reader
        self.decoder = decoder

    def __getitem__(self, fname: str) -> tuple[Any, SourceInfo]:
        data, source_info = self.inner_reader[fname]
        return self.decoder.decode(fname, data), source_info

    def get_path(self) -> str:
        return self.inner_reader.get_path()

    def __str__(self):
        return f"DecodeFileStore(inner_reader={self.inner_reader}, decoder={self.decoder})"


class SystemFileStore(FileStore[bytes]):
    """A FileStore that reads files directly from the file system."""

    def __init__(self, base_dir: Optional[Union[EPath, str]] = None):
        """
        Args:
            base_dir: The base directory to use for relative paths. If None, you should only pass
                absolute paths to __getitem__.
        """

        self.base_dir = EPath(base_dir) if base_dir is not None else None

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
            index=None,
            shard_name=None,
            file_names=(key,),
        )

    def get_path(self) -> str:
        """Returns the path to the dataset."""
        return str(self.base_dir)

    def __str__(self):
        return f"SystemFileStore(base_dir={self.base_dir})"


class WebdatasetFileStore(SqliteITarEntryReader, FileStore[bytes]):
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset."""

    def __init__(
        self,
        dataset_path: EPath,
    ):
        super().__init__(base_path=dataset_path, key_is_full_entryname=True)

    def get_path(self) -> str:
        return str(self.base_path)
