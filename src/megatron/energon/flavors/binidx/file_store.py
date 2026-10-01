# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from megatron.energon.cache.base import FileStore
from megatron.energon.flavors.binidx.binidx_reader import BinIdxReader


class BinIdxFileStore(BinIdxReader, FileStore[bytes]):
    """Random access to entries from a bin-idx file pair."""

    def get_path(self) -> str:
        return str(self.bin_path)
