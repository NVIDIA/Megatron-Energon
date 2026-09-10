# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Generic, Optional, Type, TypeVar

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import BaseCoreDatasetFactory
from megatron.energon.flavors.base_indexed_dataset import BaseIndexedDatasetFactory
from megatron.energon.flavors.common.manifest.io import ShardListMeta, check_dataset_info_present
from megatron.energon.flavors.common.manifest.paths import MAIN_FOLDER_NAME
from megatron.energon.flavors.common.manifest.types import DatasetSubset, ManifestSplits
from megatron.energon.flavors.dataset_factory_resolver import (
    PRIORITY_MANIFEST,
    register_dataset_factory_provider,
)
from megatron.energon.flavors.dataset_type import EnergonDatasetType
from megatron.energon.worker import WorkerConfig

T_sample = TypeVar("T_sample", covariant=True)


@register_dataset_factory_provider(priority=PRIORITY_MANIFEST)
class BaseManifestDatasetFactory(
    BaseIndexedDatasetFactory[T_sample],
    Generic[T_sample],
):
    @classmethod
    def detect_path(cls, path: EPath) -> EnergonDatasetType | None:
        if check_dataset_info_present(path):
            return EnergonDatasetType.MANIFEST_DATASET
        return None

    @classmethod
    def from_path(
        cls,
        path: EPath,
        *,
        dataset_config: str | None = None,
        split_config: str | None = None,
        split_part: str | None = None,
        sample_type: Type[T_sample] | None = None,
        **kwargs: Any,
    ) -> BaseCoreDatasetFactory:
        from megatron.energon.dataset_config import load_config
        from megatron.energon.flavors.webdataset.standard_webdataset import (
            StandardWebdatasetFactory,
        )

        if dataset_config is None:
            dataset_config = "dataset.yaml"
        if split_config is None:
            split_config = "split.yaml"
        if split_part is None:
            split_part = "train"
        dataset = load_config(
            path / MAIN_FOLDER_NAME / dataset_config,
            default_kwargs=dict(
                path=path,
                split_config=split_config,
                split_part=split_part,
                **kwargs,
            ),
            default_type=StandardWebdatasetFactory,
        )
        if sample_type is not None:
            assert issubclass(dataset.__sample_type__, sample_type), (
                f"Sample of type {dataset.__sample_type__} is not a subclass of {sample_type}."
            )
        return dataset


class BaseManifestShardListDatasetFactory(
    BaseManifestDatasetFactory[T_sample],
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
