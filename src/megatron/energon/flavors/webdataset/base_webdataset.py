# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, TypeVar

from megatron.energon.cache import FileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_indexed_dataset import (
    DEBUG_SHARD_PRINT,
)
from megatron.energon.flavors.base_manifest_dataset import BaseManifestShardListDatasetFactory
from megatron.energon.flavors.common.manifest.types import DatasetSubset, ManifestSplits
from megatron.energon.flavors.common.sample_record import SampleRecord
from megatron.energon.flavors.webdataset.prepare import WebdatasetPreparator
from megatron.energon.worker import WorkerConfig

T_sample = TypeVar("T_sample", covariant=True)
T = TypeVar("T", covariant=True)

logger = logging.getLogger(__name__)


class BaseWebdatasetFactory(
    BaseManifestShardListDatasetFactory[T_sample],
    WebdatasetPreparator,
    Generic[T_sample],
    ABC,
):
    """
    Base class for all webdataset sample loader factories. Applies proper sharding across workers.
    """

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
        subset: Optional[DatasetSubset] = None,
        split_config: str | ManifestSplits | None = None,
        part_filter: Optional[Callable[[str], bool]] = None,
        filter_name: Optional[str] = None,
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
            subset: If specified, the dataset will be subsetted.
            split_config: Config file to use for shard split definitions.
            part_filter: (internal) Function for filtering tar files by dict keys
        """
        super().__init__(
            path,
            split_part=split_part,
            split_config=split_config,
            training=training,
            worker_config=worker_config,
            shuffle_over_epochs=shuffle_over_epochs,
            parallel_shard_iters=parallel_shard_iters,
            max_samples_per_sequence=max_samples_per_sequence,
            subset=subset,
            part_filter=part_filter,
            filter_name=filter_name,
        )

    def _build_reader(
        self,
        *,
        parallel_shard_iters: int,
        part_filter: Callable[[str], bool] | None,
    ):
        from megatron.energon.flavors.webdataset.itar_reader import ShardInfosITarReader

        return ShardInfosITarReader(
            self.path,
            self.shards,
            part_filter=part_filter,
            sample_filter=self.sample_filter,
            itar_cache_size=parallel_shard_iters,
        )

    def as_file_store(self) -> "FileStore":
        from megatron.energon.flavors.webdataset.file_store import WebdatasetFileStore

        return WebdatasetFileStore(self.path)

    def sample_filter(self, key: str) -> bool:
        return key not in self.sample_excludes

    def _print_shard_slices(self, slice_offsets, shards) -> None:
        if DEBUG_SHARD_PRINT:
            super()._print_shard_slices(slice_offsets, shards)

    def config(self):
        return dict(
            **super().config(),
            filter_name=self.filter_name,
        )

    @abstractmethod
    def load_sample(self, raw_data: SampleRecord) -> T_sample:
        """Loads the sample from the dataset."""
        ...

    def __str__(self):
        return f"{type(self).__name__}(path={self.path})"
