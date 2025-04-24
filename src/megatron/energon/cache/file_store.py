# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional, Union

from megatron.energon.cache.base import FileStore, FileStoreDecoder
from megatron.energon.epathlib import EPath


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

    def __getitem__(self, fname: str) -> Any:
        return self.decoder.decode(fname, self.inner_reader[fname])

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

    def __getitem__(self, key: str) -> bytes:
        # Construct the full path from the dataset path and the file key
        if self.base_dir is None:
            file_path = EPath(key)
        else:
            file_path = self.base_dir / key

        # Read and return the file contents as bytes
        with file_path.open("rb") as f:
            return f.read()

    def get_path(self) -> str:
        """Returns the path to the dataset."""
        return str(self.base_dir)

    def __str__(self):
        return f"SystemFileStore(base_dir={self.base_dir})"
