# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, TypeVar

import numpy as np

from megatron.energon.cache import FileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import (
    BaseCoreDatasetFactory,
    SavableDataset,
    legacy_handler,
)
from megatron.energon.flavors.webdataset.error_handler import ErrorHandler
from megatron.energon.flavors.webdataset.metadata import WebdatasetMeta
from megatron.energon.flavors.webdataset.prepare import WebdatasetPreparator
from megatron.energon.flavors.webdataset.sample_loader import (
    RawSampleData,
    WebdatasetSampleLoaderDataset,
)
from megatron.energon.flavors.webdataset.sharder import Sharder
from megatron.energon.flavors.webdataset.structs import FilteredSample, ShardInfo, reraise_exception
from megatron.energon.source_info import SourceInfo
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.map_dataset import MapDataset

T_sample = TypeVar("T_sample", covariant=True)
T = TypeVar("T", covariant=True)

logger = logging.getLogger(__name__)


class BaseWebdatasetFactory(
    BaseCoreDatasetFactory[T_sample],
    WebdatasetPreparator,
    Sharder,
    ErrorHandler,
    Generic[T_sample],
    ABC,
):
    """
    Base class for all webdataset sample loader factories. Applies proper sharding across workers.
    """

    path: EPath

    training: bool
    worker_config: WorkerConfig

    shards: List[ShardInfo]
    rank_shards: List[List[Sequence[ShardInfo]]]

    def __init__(
        self,
        path: EPath,
        *,
        split_part: str,
        training: bool,
        worker_config: WorkerConfig,
        shuffle_over_epochs: Optional[int] = 1,
        parallel_shard_iters: Optional[int] = None,
        max_samples_per_sequence: Optional[int] = None,
        split_config: str = "split.yaml",
        part_filter: Optional[Callable[[str], bool]] = None,
        handler: Callable[
            [Exception, Optional[str], Optional[list[SourceInfo]]], None
        ] = reraise_exception,
    ):
        """
        Base factory for the webdataset sample loader.

        Args:
            path: Path to the dataset.
            split_part: Which part to load (e.g. 'train', 'val', 'test').
            training: If true, apply shuffling and loop the dataset.
            worker_config: Configuration for the workers.
            shuffle_over_epochs: Only effective if training=True.
                How many epochs to shuffle over if training.
                If = 1, every sample is seen exactly once per epoch.
                If > 1, samples (or rather shard slices) are shuffled within this number of epochs
                (i.e. randomly selected without replacement).
                If -1, the shards are effectively shuffle over infinite epochs (i.e. shard slices
                are drawn with replacement).
            parallel_shard_iters: Number of parallel opened shards per worker, shuffling between.
            max_samples_per_sequence: Maximum number of samples per sequence (=how many samples
                    will be sequentially iterated).
            split_config: Config file to use for shard split definitions.
            part_filter: (internal) Function for filtering tar files by dict keys
            handler: Exception handler. Args: (exception, key).
        """
        assert self.__sample_type__ is not None, f"Class {type(self)} must define __sample_type__"
        wds_meta = WebdatasetMeta.from_config(
            path=path, split_part=split_part, split_config=split_config
        )
        self.path = path
        self.paths = [path]
        self.shards = wds_meta.shards
        self.sample_excludes = wds_meta.sample_excludes
        self.split_part_files = wds_meta.split_part_files
        self.training = training
        self.worker_config = worker_config
        self.shuffle_over_epochs = shuffle_over_epochs
        self.parallel_shard_iters = parallel_shard_iters
        self.max_samples_per_sequence = max_samples_per_sequence
        self.part_filter = part_filter
        self.handler = legacy_handler(handler)

    def __len__(self) -> int:
        return sum(shard.count for shard in self.shards)

    def build(self, worker_rotation_offset: int = 0) -> SavableDataset[T_sample]:
        from megatron.energon.flavors.webdataset.itar_reader import ShardInfosITarReader

        if self.parallel_shard_iters is None:
            if self.training:
                # 16 seems to be a good choice since we don't want too many file handles open
                parallel_shard_iters = 16
            else:
                parallel_shard_iters = 1
        else:
            parallel_shard_iters = self.parallel_shard_iters

        workers_sample_slice_offsets = self.shard_workers(
            self.shards,
            worker_config=self.worker_config,
            max_samples_per_sequence=self.max_samples_per_sequence,
            rotation_offset=worker_rotation_offset,
        )
        _print_shard_slices(self.worker_config, self.shards, workers_sample_slice_offsets)

        itar_reader = ShardInfosITarReader(
            self.path,
            self.shards,
            part_filter=self.part_filter,
            sample_filter=self.sample_filter,
            itar_cache_size=parallel_shard_iters,
        )

        dataset = WebdatasetSampleLoaderDataset(
            join_readers=[itar_reader],
            workers_sample_slice_offsets=workers_sample_slice_offsets,
            worker_config=self.worker_config,
            shuffle_over_epochs=self.shuffle_over_epochs if self.training else None,
            parallel_slice_iters=parallel_shard_iters,
        )
        return MapDataset(
            dataset,
            self._load_sample_raw,
            error_handler=self.error_handler,
            stateless_map_fn=True,
            map_fn_config=self.config,
            worker_config=self.worker_config,
        )

    def as_file_store(self) -> "FileStore":
        from megatron.energon.cache.file_store import WebdatasetFileStore

        return WebdatasetFileStore(self.path)

    def sample_filter(self, key: str) -> bool:
        return key not in self.sample_excludes

    def _load_sample_raw(self, raw_sample: RawSampleData) -> T_sample:
        # Just a wrapper for the inner tuple. Tuple should be of length 1.
        assert len(raw_sample.data) == 1 and raw_sample.data[0] is not None
        return self.load_sample(raw_sample.data[0])

    @abstractmethod
    def load_sample(self, raw_data: FilteredSample) -> T_sample:
        """Loads the sample from the dataset."""
        ...

    def config(self) -> Dict[str, Any]:
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
        )

    def __str__(self):
        return f"{type(self).__name__}(path={self.path})"


def _print_shard_slices(
    worker_config: WorkerConfig, shards: List[ShardInfo], slice_offsets: Sequence[Sequence[int]]
):
    shard_starts = np.cumsum([0] + [shard.count for shard in shards])

    def shard_range_info(start: int, end: int) -> str:
        start_shard_idx = np.searchsorted(shard_starts, start, side="right") - 1
        end_shard_idx = np.searchsorted(shard_starts, end, side="left") - 1
        if start_shard_idx == end_shard_idx:
            shard = shards[start_shard_idx]
            if start - shard_starts[start_shard_idx] == 0:
                start_str = "(start)"
            else:
                start_str = ""
            if end - shard_starts[start_shard_idx] == shard.count:
                end_str = "(end)"
            else:
                end_str = ""
            return f"{shard.name}[{start - shard_starts[start_shard_idx]}{start_str}, {end - shard_starts[start_shard_idx]}{end_str}]"
        else:
            start_shard = shards[start_shard_idx]
            end_shard = shards[end_shard_idx]
            if start - shard_starts[start_shard_idx] == 0:
                start_str = "(start)"
            else:
                start_str = ""
            if end - shard_starts[end_shard_idx] == end_shard.count:
                end_str = "(end)"
            else:
                end_str = ""
            return f"{start_shard.name}[{start - shard_starts[start_shard_idx]}{start_str},]-{end_shard.name}[,{end - shard_starts[end_shard_idx]}{end_str}]"

    for worker_idx, sample_slice_offsets in enumerate(slice_offsets):
        start_idx = sample_slice_offsets[0]
        end_idx = sample_slice_offsets[-1]

        if len(sample_slice_offsets) > 6:
            offset_str = f"{', '.join(str(o) for o in sample_slice_offsets[:3])} ...<{len(sample_slice_offsets) - 6}> {', '.join(str(o) for o in sample_slice_offsets[-3:])}"
        else:
            offset_str = ", ".join(str(o) for o in sample_slice_offsets)
        if len(sample_slice_offsets) > 6:
            slices_str = (
                ", ".join(
                    shard_range_info(start, end)
                    for start, end in zip(sample_slice_offsets[:3], sample_slice_offsets[1:4])
                )
                + f" ...<{len(sample_slice_offsets) - 6}> "
                + ", ".join(
                    shard_range_info(start, end)
                    for start, end in zip(sample_slice_offsets[-4:-1], sample_slice_offsets[-3:])
                )
            )
        else:
            slices_str = ", ".join(
                shard_range_info(start, end)
                for start, end in zip(sample_slice_offsets[:-1], sample_slice_offsets[1:])
            )

        print(
            f"rank={worker_config.rank}, worker={worker_idx}: sample_range=[{start_idx}, {end_idx}] in {len(sample_slice_offsets) - 1} slices, "
            f"sum(count)={end_idx - start_idx}: indexes=[{offset_str}] slices=[{slices_str}]"
        )
