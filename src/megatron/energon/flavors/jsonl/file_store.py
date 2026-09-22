# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from megatron.energon.cache.base import FileStore
from megatron.energon.flavors.jsonl.ijsonl_reader import IJsonlReader
from megatron.energon.flavors.jsonl.multi_jsonl_reader import MultiJsonlFileStore


class JsonlFileStore(IJsonlReader, FileStore[bytes]):
    """Random access to entries from a single JSONL file."""

    def get_path(self) -> str:
        return str(self.jsonl_path)


__all__ = ["JsonlFileStore", "MultiJsonlFileStore"]
