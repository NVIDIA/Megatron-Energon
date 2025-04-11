# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import FileStore
from megatron.energon.flavors.webdataset.itar_reader import SqliteITarEntryReader
from megatron.energon.flavors.webdataset.sample_decoder import SampleDecoder


class WebdatasetFileStore(SqliteITarEntryReader, FileStore[bytes]):
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset."""

    def __init__(
        self,
        dataset_path: EPath,
    ):
        super().__init__(base_path=dataset_path, key_is_full_entryname=True)

    def get_path(self) -> str:
        return str(self.base_path)


class DecodeFileStore(FileStore[Any]):
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset and decode them."""

    def __init__(
        self,
        inner_reader: FileStore[bytes],
        *,
        decoder: SampleDecoder,
    ):
        self.inner_reader = inner_reader
        self.decoder = decoder

    def __getitem__(self, fname: str) -> Any:
        return self.decoder.decode(fname, self.inner_reader[fname])

    def get_path(self) -> str:
        return self.inner_reader.get_path()
