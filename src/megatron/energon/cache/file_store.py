# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from megatron.energon.cache.base import FileStore, FileStoreDecoder


class DecodeFileStore(FileStore[Any]):
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset and decode them."""

    def __init__(
        self,
        inner_reader: FileStore[bytes],
        *,
        decoder: FileStoreDecoder,
    ):
        self.inner_reader = inner_reader
        self.decoder = decoder

    def __getitem__(self, fname: str) -> Any:
        return self.decoder.decode(fname, self.inner_reader[fname])

    def get_path(self) -> str:
        return self.inner_reader.get_path()
