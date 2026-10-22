# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from bisect import bisect_right
from collections import OrderedDict
from typing import Any, Callable, Optional

from megatron.energon.cache import FileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.common.manifest.types import ShardInfo
from megatron.energon.flavors.common.reader import IndexedSampleReader
from megatron.energon.flavors.common.sample_record import FilteredSample
from megatron.energon.flavors.jsonl.ijsonl import IJsonlIndexReader
from megatron.energon.source_info import SourceInfo


class MultiJsonlReader(IndexedSampleReader[FilteredSample]):
    """Random-access reader over a prepared logical dataset of JSONL shards."""

    def __init__(
        self,
        path: EPath,
        shards: list[ShardInfo],
        *,
        index_cache_size: int = 5,
        reader_cache_size: int = 16,
        part_filter: Optional[Callable[[str], bool]] = None,
    ):
        assert shards, f"No JSONL shards found for {path}"
        self.path = path
        self.shards = shards
        self.index_cache_size = index_cache_size
        self.reader_cache_size = reader_cache_size
        self.part_filter = part_filter
        # Counts come from the manifest; shard indexes are only opened when a shard is first read.
        self.cumulative_counts: list[int] = []
        running = 0
        for shard in shards:
            running += shard.count
            self.cumulative_counts.append(running)
        self._validated_shards: set[int] = set()
        self._readers: OrderedDict[int, Any] = OrderedDict()

    def __len__(self) -> int:
        return self.cumulative_counts[-1] if self.cumulative_counts else 0

    def __str__(self) -> str:
        return f"MultiJsonlReader(path={self.path}, num_shards={len(self.shards)})"

    def _validate_shard(self, shard_idx: int) -> None:
        if shard_idx in self._validated_shards:
            return
        shard = self.shards[shard_idx]
        if (
            not IJsonlIndexReader.is_current(shard.path)
            or IJsonlIndexReader.count_samples(shard.path) != shard.count
        ):
            raise ValueError(
                f"The index of {shard.path} does not match the file or the dataset manifest. "
                "Regenerate the index with `energon prepare`."
            )
        self._validated_shards.add(shard_idx)

    def _reader_for_shard(self, shard_idx: int):
        from megatron.energon.flavors.jsonl.ijsonl_reader import IJsonlReader

        if shard_idx in self._readers:
            reader = self._readers.pop(shard_idx)
            self._readers[shard_idx] = reader
            return reader

        self._validate_shard(shard_idx)
        reader = IJsonlReader(
            self.shards[shard_idx].path,
            index_cache_size=self.index_cache_size,
            part_filter=self.part_filter,
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

    def _with_global_metadata(
        self,
        sample: FilteredSample,
        *,
        global_idx: int,
        shard_idx: int,
        local_idx: int,
    ) -> FilteredSample:
        shard_name = self.shards[shard_idx].name
        sample = FilteredSample(sample)
        sample["__key__"] = str(global_idx)
        sample["__shard__"] = shard_name
        # Must match the restore_key_kind of the DatasetSampler built by the factory.
        sample["__restore_key__"] = ("Webdataset", global_idx)
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
        shards: list[ShardInfo],
        *,
        reader_cache_size: int = 16,
    ):
        self.path = path
        self.reader = MultiJsonlReader(
            path,
            shards,
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
