# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import RandomAccessDataset
from megatron.energon.flavors.webdataset.config import split_name_re
from megatron.energon.flavors.webdataset.itar_reader import SqliteITarEntryReader
from megatron.energon.flavors.webdataset.sample_decoder import SampleDecoder

STATS_LOAD_QUEUE_SIZE = 0
STATS_DECODE_QUEUE_SIZE = 0
STATS_READ_TIME_NS = 0
STATS_DECODE_TIME_NS = 0


class RandomAccessWebdataset(SqliteITarEntryReader, RandomAccessDataset[bytes]):
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset."""

    def __init__(
        self,
        dataset_path: EPath,
    ):
        super().__init__(base_path=dataset_path, key_is_full_entryname=True)

    def get_path(self) -> str:
        return str(self.base_path)


class RandomAccessDecoderDataset(RandomAccessDataset[Any]):
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset and decode them."""

    def __init__(
        self,
        inner_reader: RandomAccessDataset[bytes],
        *,
        decoder: SampleDecoder,
    ):
        self._inner_reader = inner_reader
        self._decoder = decoder

    def __getitem__(self, fname: str) -> Any:
        m = split_name_re.match(fname)
        if not m:
            raise ValueError(f"Invalid file name: {fname}")
        cur_base_name, ext = m.groups()

        return self._decoder(
            {
                "__key__": cur_base_name,
                ext: self._inner_reader[fname],
            }
        )[ext]

    def get_path(self) -> str:
        return self._inner_reader.get_path()
