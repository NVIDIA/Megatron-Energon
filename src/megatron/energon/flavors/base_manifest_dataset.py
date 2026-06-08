# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable, Generic, Optional, TypeVar

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_indexed_dataset import BaseIndexedDatasetFactory
from megatron.energon.flavors.common.manifest.io import ShardListMeta
from megatron.energon.flavors.common.manifest.types import DatasetSubset, ManifestSplits
from megatron.energon.worker import WorkerConfig

T_sample = TypeVar("T_sample", covariant=True)


class BaseManifestShardListDatasetFactory(
    BaseIndexedDatasetFactory[T_sample],
    Generic[T_sample],
):
    """Base for datasets whose virtual shards are read from `.nv-meta` manifests."""

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
        meta = ShardListMeta.from_config(
            path=EPath(path),
            split_part=split_part,
            split_config=split_config,
        )
        self._validate_manifest_meta(meta)
        super().__init__(
            path,
            shards=meta.shards,
            sample_excludes=meta.sample_excludes,
            split_part_files=meta.split_part_files,
            training=training,
            worker_config=worker_config,
            shuffle_over_epochs=shuffle_over_epochs,
            parallel_shard_iters=parallel_shard_iters,
            max_samples_per_sequence=max_samples_per_sequence,
            subset=subset,
            part_filter=part_filter,
            filter_name=filter_name,
        )

    def _validate_manifest_meta(self, meta: ShardListMeta) -> None:
        pass
