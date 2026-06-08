# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from bisect import bisect_right
from collections import OrderedDict
from typing import Any

from megatron.energon.cache import FileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.common.reader import IndexedSampleReader
from megatron.energon.flavors.common.sample_record import FilteredSample
from megatron.energon.flavors.jsonl.ijsonl import IJsonlIndexReader
from megatron.energon.source_info import SourceInfo


class MultiJsonlReader(IndexedSampleReader[FilteredSample]):
    """Random-access reader over a prepared logical dataset of JSONL shards."""

    def __init__(
        self,
        path: EPath,
        jsonl_paths: list[EPath],
        *,
        index_cache_size: int = 5,
        reader_cache_size: int = 16,
    ):
        assert jsonl_paths, f"No JSONL shards found for {path}"
        self.path = path
        self.jsonl_paths = jsonl_paths
        self.index_cache_size = index_cache_size
        self.reader_cache_size = reader_cache_size
        self.counts = [IJsonlIndexReader.count_samples(path) for path in jsonl_paths]
        self.cumulative_counts: list[int] = []
        running = 0
        for count in self.counts:
            running += count
            self.cumulative_counts.append(running)
        self._readers: OrderedDict[int, Any] = OrderedDict()

    def __len__(self) -> int:
        return self.cumulative_counts[-1] if self.cumulative_counts else 0

    def __str__(self) -> str:
        return f"MultiJsonlReader(path={self.path}, num_shards={len(self.jsonl_paths)})"

    def _reader_for_shard(self, shard_idx: int):
        from megatron.energon.flavors.jsonl.ijsonl_reader import IJsonlReader

        if shard_idx in self._readers:
            reader = self._readers.pop(shard_idx)
            self._readers[shard_idx] = reader
            return reader

        reader = IJsonlReader(
            self.jsonl_paths[shard_idx],
            index_cache_size=self.index_cache_size,
        )
        self._readers[shard_idx] = reader
        while len(self._readers) > self.reader_cache_size:
            _old_idx, old_reader = self._readers.popitem(last=False)
            old_reader.close()
        return reader

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")
        shard_idx = bisect_right(self.cumulative_counts, idx)
        prev_count = self.cumulative_counts[shard_idx - 1] if shard_idx > 0 else 0
        return shard_idx, idx - prev_count

    def _shard_name(self, shard_idx: int) -> str:
        shard_path = self.jsonl_paths[shard_idx]
        try:
            return shard_path.relative_to(self.path)
        except ValueError:
            return shard_path.name

    def _with_global_metadata(
        self,
        sample: FilteredSample,
        *,
        global_idx: int,
        shard_idx: int,
        local_idx: int,
    ) -> FilteredSample:
        shard_name = self._shard_name(shard_idx)
        sample = FilteredSample(sample)
        sample["__key__"] = str(global_idx)
        sample["__shard__"] = shard_name
        sample["__restore_key__"] = ("JsonlShardList", global_idx)
        sample["__sources__"] = (
            SourceInfo(
                dataset_path=str(self.path),
                index=global_idx,
                shard_name=shard_name,
                file_names=(f"{local_idx}.json",),
            ),
        )
        return sample

    def __getitem__(self, idx: int | str):
        full_entry_name = False
        if isinstance(idx, str):
            if idx.endswith(".json"):
                idx = idx.removesuffix(".json")
                full_entry_name = True
            idx = int(idx)
        shard_idx, local_idx = self._locate(idx)
        sample = self._reader_for_shard(shard_idx)[local_idx]
        if sample is None:
            return None
        sample = self._with_global_metadata(
            sample,
            global_idx=idx,
            shard_idx=shard_idx,
            local_idx=local_idx,
        )
        if full_entry_name:
            assert len(sample["__sources__"]) == 1
            return sample["json"], sample["__sources__"][0]
        return sample

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()


class MultiJsonlFileStore(FileStore[bytes]):
    """FileStore view over a prepared logical dataset of JSONL shards."""

    def __init__(
        self,
        path: EPath,
        jsonl_paths: list[EPath],
        *,
        reader_cache_size: int = 16,
    ):
        self.path = path
        self.reader = MultiJsonlReader(
            path,
            jsonl_paths,
            reader_cache_size=reader_cache_size,
        )

    def __getitem__(self, key: str) -> tuple[bytes, SourceInfo]:
        full_key = key if key.endswith(".json") else f"{key}.json"
        sample = self.reader[full_key]
        if sample is None:
            raise KeyError(key)
        return sample

    def get_path(self) -> str:
        return str(self.path)

    def close(self) -> None:
        self.reader.close()
