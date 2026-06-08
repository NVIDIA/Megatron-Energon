# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable, Optional, TypeVar

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_indexed_dataset import BaseIndexedDatasetFactory
from megatron.energon.flavors.common.manifest.types import DatasetSubset, ShardInfo
from megatron.energon.worker import WorkerConfig

T_sample = TypeVar("T_sample", covariant=True)


class BaseFileCollectionDatasetFactory(BaseIndexedDatasetFactory[T_sample]):
    """Base for discovered file collections represented as virtual shards."""

    def __init__(
        self,
        path: EPath,
        *,
        shards: list[ShardInfo],
        training: bool,
        worker_config: WorkerConfig,
        shuffle_over_epochs: Optional[int] = 1,
        parallel_shard_iters: Optional[int] = None,
        max_samples_per_sequence: Optional[int] = None,
        subset: Optional[DatasetSubset] = None,
        part_filter: Optional[Callable[[str], bool]] = None,
        filter_name: Optional[str] = None,
    ):
        super().__init__(
            EPath(path),
            shards=shards,
            training=training,
            worker_config=worker_config,
            shuffle_over_epochs=shuffle_over_epochs,
            parallel_shard_iters=parallel_shard_iters,
            max_samples_per_sequence=max_samples_per_sequence,
            subset=subset,
            part_filter=part_filter,
            filter_name=filter_name,
        )
