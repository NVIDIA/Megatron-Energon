# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar

import numpy as np

from megatron.energon.cache import FileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import BaseCoreDatasetFactory, SavableDataset
from megatron.energon.flavors.common.dataset_sampler import DatasetSampler, RawSampleData
from megatron.energon.flavors.common.filter_index import FilterIndex, TranslatedIndexReader
from megatron.energon.flavors.common.manifest.sharder import Sharder
from megatron.energon.flavors.common.manifest.types import DatasetSubset, ShardInfo
from megatron.energon.flavors.common.reader import IndexedSampleReader
from megatron.energon.flavors.common.sample_record import SampleRecord
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.map_dataset import MapDataset

T_sample = TypeVar("T_sample", covariant=True)

DEBUG_SHARD_PRINT = os.getenv("ENERGON_DEBUG_SHARD_PRINT", "0") == "1"


def _print_shard_slices(
    worker_config: WorkerConfig, shards: list[ShardInfo], slice_offsets: Sequence[Sequence[int]]
) -> None:
    shard_starts = np.cumsum([0] + [shard.count for shard in shards])

    def shard_range_info(start: int, end: int) -> str:
        start_shard_idx = np.searchsorted(shard_starts, start, side="right") - 1
        end_shard_idx = np.searchsorted(shard_starts, end, side="left") - 1
        if start_shard_idx == end_shard_idx:
            shard = shards[start_shard_idx]
            start_str = "(start)" if start - shard_starts[start_shard_idx] == 0 else ""
            end_str = "(end)" if end - shard_starts[start_shard_idx] == shard.count else ""
            return (
                f"{shard.name}[{start - shard_starts[start_shard_idx]}{start_str}, "
                f"{end - shard_starts[start_shard_idx]}{end_str}]"
            )

        start_shard = shards[start_shard_idx]
        end_shard = shards[end_shard_idx]
        start_str = "(start)" if start - shard_starts[start_shard_idx] == 0 else ""
        end_str = "(end)" if end - shard_starts[end_shard_idx] == end_shard.count else ""
        return (
            f"{start_shard.name}[{start - shard_starts[start_shard_idx]}{start_str},]-"
            f"{end_shard.name}[,{end - shard_starts[end_shard_idx]}{end_str}]"
        )

    for worker_idx, sample_slice_offsets in enumerate(slice_offsets):
        start_idx = sample_slice_offsets[0]
        end_idx = sample_slice_offsets[-1]

        if len(sample_slice_offsets) > 6:
            indexes_str = (
                ", ".join(str(i) for i in sample_slice_offsets[:3])
                + ", ..., "
                + ", ".join(str(i) for i in sample_slice_offsets[-3:])
            )
        else:
            indexes_str = ", ".join(str(i) for i in sample_slice_offsets)
        print(
            f"rank={worker_config.rank}, worker={worker_idx}: "
            f"sample_range=[{start_idx}, {end_idx}] in {len(sample_slice_offsets) - 1} "
            f"slices, sum(count)={end_idx - start_idx}: indexes=[{indexes_str}] "
            f"slices=[{', '.join(shard_range_info(start, end) for start, end in zip(sample_slice_offsets, sample_slice_offsets[1:]))}]"
        )


class BaseIndexedDatasetFactory(
    BaseCoreDatasetFactory[T_sample],
    Sharder,
    Generic[T_sample],
    ABC,
):
    """Common runtime pipeline for indexed datasets over virtual shards."""

    path: EPath
    paths: list[EPath]
    shards: list[ShardInfo]
    training: bool
    worker_config: WorkerConfig
    shuffle_over_epochs: Optional[int]
    parallel_shard_iters: Optional[int]
    max_samples_per_sequence: Optional[int]
    subset: Optional[DatasetSubset]
    part_filter: Optional[Callable[[str], bool]]

    def __init__(
        self,
        path: EPath,
        *,
        shards: list[ShardInfo],
        sample_excludes: Optional[set[str]] = None,
        split_part_files: Optional[list[str]] = None,
        training: bool,
        worker_config: WorkerConfig,
        shuffle_over_epochs: Optional[int] = 1,
        parallel_shard_iters: Optional[int] = None,
        max_samples_per_sequence: Optional[int] = None,
        subset: Optional[DatasetSubset] = None,
        part_filter: Optional[Callable[[str], bool]] = None,
        filter_name: Optional[str] = None,
        restore_key_kind: str = "Webdataset",
    ):
        assert self.__sample_type__ is not None, f"Class {type(self)} must define __sample_type__"
        self.path = EPath(path)
        self.paths = [self.path]
        self.name = self.path.display_name
        self.shards = shards
        self.sample_excludes = sample_excludes or set()
        self.split_part_files = split_part_files or [shard.name for shard in shards]
        self.training = training
        self.worker_config = worker_config
        self.shuffle_over_epochs = shuffle_over_epochs
        self.parallel_shard_iters = parallel_shard_iters
        self.max_samples_per_sequence = max_samples_per_sequence
        self.subset = subset
        self.part_filter = part_filter
        self.filter_name = filter_name
        self.restore_key_kind = restore_key_kind
        self.filter_index = None
        if filter_name is not None:
            self.filter_index = FilterIndex(self.path, filter_name)

    def __len__(self) -> int:
        if self.filter_index:
            return len(self.filter_index)
        return sum(shard.count for shard in self.shards)

    def build(
        self, worker_rotation_offset: int = 0, part_filter: Callable[[str], bool] | None = None
    ) -> SavableDataset[T_sample]:
        parallel_shard_iters = self.parallel_shard_iters
        if parallel_shard_iters is None:
            parallel_shard_iters = 16 if self.training else 1

        part_filter = self._merge_part_filter(part_filter)
        if self.filter_index is None:
            active_shards = self.shards
        else:
            active_shards = self.filter_index.translate_shards(self.shards)

        workers_sample_slice_offsets = self.shard_workers(
            active_shards,
            worker_config=self.worker_config,
            max_samples_per_sequence=self.max_samples_per_sequence,
            rotation_offset=worker_rotation_offset,
            subset=self.subset,
        )
        self._print_shard_slices(workers_sample_slice_offsets, active_shards)

        reader = self._build_reader(
            parallel_shard_iters=parallel_shard_iters,
            part_filter=part_filter,
        )
        if self.filter_index is not None:
            reader = TranslatedIndexReader(reader, self.filter_index)

        dataset = DatasetSampler(
            join_readers=[reader],
            workers_sample_slice_offsets=workers_sample_slice_offsets,
            worker_config=self.worker_config,
            shuffle_over_epochs=self.shuffle_over_epochs if self.training else None,
            parallel_slice_iters=parallel_shard_iters,
            restore_key_kind=self.restore_key_kind,
        )
        return MapDataset(
            dataset,
            self._load_fn(part_filter),
            stateless_map_fn=True,
            map_fn_config=self.config,
            worker_config=self.worker_config,
        )

    def _merge_part_filter(
        self, part_filter: Callable[[str], bool] | None
    ) -> Callable[[str], bool] | None:
        if self.part_filter is None:
            return part_filter
        if part_filter is None:
            return self.part_filter
        inner_pf, outer_pf = part_filter, self.part_filter
        return lambda p, _i=inner_pf, _o=outer_pf: _o(p) and _i(p)

    def _print_shard_slices(
        self, slice_offsets: Sequence[Sequence[int]], shards: list[ShardInfo]
    ) -> None:
        if DEBUG_SHARD_PRINT:
            _print_shard_slices(self.worker_config, shards, slice_offsets)

    def _load_fn(
        self, part_filter: Callable[[str], bool] | None
    ) -> Callable[[RawSampleData], T_sample]:
        return self._load_sample_raw

    def _load_sample_raw(self, raw_sample: RawSampleData) -> T_sample:
        assert len(raw_sample.data) == 1 and raw_sample.data[0] is not None
        return self.load_sample(raw_sample.data[0])

    @abstractmethod
    def _build_reader(
        self,
        *,
        parallel_shard_iters: int,
        part_filter: Callable[[str], bool] | None,
    ) -> IndexedSampleReader: ...

    @abstractmethod
    def load_sample(self, raw_data: SampleRecord) -> T_sample:
        """Loads the sample from the dataset."""
        ...

    @abstractmethod
    def as_file_store(self) -> FileStore: ...

    def config(self) -> dict[str, Any]:
        return dict(
            type=type(self).__qualname__,
            training=self.training,
            _path=str(self.path),
            shards=[
                dict(
                    name=shard.name,
                    count=shard.count,
                    _path=str(shard.path),
                )
                for shard in self.shards
            ],
            sample_excludes=list(self.sample_excludes),
            shuffle_over_epochs=self.shuffle_over_epochs,
            parallel_shard_iters=self.parallel_shard_iters,
            max_samples_per_sequence=self.max_samples_per_sequence,
            subset=self.subset.config() if self.subset is not None else None,
            filter_name=self.filter_name,
        )
