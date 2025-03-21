# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from megatron.energon.dataclass_slots import dataclass_slots
from megatron.energon.dataset_config import load_config
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import BaseCoreDatasetFactory
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from megatron.energon.metadataset.dataset_loader import DatasetLoader
from megatron.energon.metadataset.loader_interface import DatasetBlendMode, DatasetLoaderInterface
from megatron.energon.worker import WorkerConfig


@dataclass_slots
class DatasetReference:
    path: Union[str, EPath]
    split_part: Optional[str] = None
    subflavor: Optional[str] = None
    subflavors: Optional[Dict[str, Any]] = None
    shuffle_over_epochs_multiplier: Optional[int] = 1
    dataset_config: str = "dataset.yaml"
    split_config: str = "split.yaml"

    weight: float = 1.0

    _dataset: Optional[DatasetLoaderInterface] = None

    def post_initialize(self, mds_path: Optional[EPath] = None):
        assert mds_path is not None
        if not isinstance(self.path, EPath):
            self.path = mds_path.parent / self.path
        if self.path.is_file():
            assert self.dataset_config == "dataset.yaml", "Must not set dataset_config"
            assert self.split_config == "split.yaml", "Must not set split_config"
            self._dataset = load_config(
                self.path,
                default_type=Metadataset,
                default_kwargs=dict(path=self.path),
            )
            self._dataset.post_initialize()
        elif (self.path / MAIN_FOLDER_NAME / ".info.yaml").is_file():
            self._dataset = DatasetLoader(
                path=self.path,
                split_part=self.split_part,
                subflavor=self.subflavor,
                subflavors=self.subflavors,
                shuffle_over_epochs_multiplier=self.shuffle_over_epochs_multiplier,
                dataset_config=self.dataset_config,
                split_config=self.split_config,
            )
            self._dataset.post_initialize()
        else:
            raise FileNotFoundError(self.path)

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavor: Optional[str] = None,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        **kwargs,
    ) -> Tuple[DatasetBlendMode, List[Tuple[BaseCoreDatasetFactory, Union[float, int, None]]]]:
        if self.subflavors is not None:
            subflavors = {**self.subflavors, **(subflavors or {})}
        assert self._dataset is not None
        return self._dataset.get_datasets(
            training=training,
            split_part=self.split_part or split_part,
            worker_config=worker_config,
            subflavor=subflavor or self.subflavor,
            subflavors=subflavors,
            shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier
            * self.shuffle_over_epochs_multiplier,
            **kwargs,
        )


@dataclass_slots
class MetadatasetBlender:
    """Internal blending of the dataset."""

    datasets: List[DatasetReference]

    def post_initialize(self, mds_path: Optional[EPath] = None):
        assert mds_path is not None
        for dataset in self.datasets:
            dataset.post_initialize(mds_path)

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavor: Optional[str] = None,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        **kwargs,
    ) -> Tuple[DatasetBlendMode, List[Tuple[BaseCoreDatasetFactory, Union[float, int, None]]]]:
        sum_weight = sum(dataset.weight for dataset in self.datasets)
        datasets = []
        for dataset in self.datasets:
            inner_blend_mode, inner_datasets = dataset.get_datasets(
                training=training,
                split_part=split_part,
                worker_config=worker_config,
                subflavor=subflavor,
                subflavors=subflavors,
                shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
                **kwargs,
            )
            if inner_blend_mode not in (DatasetBlendMode.NONE, DatasetBlendMode.DATASET_WEIGHT):
                raise ValueError(
                    "Can only blend datasets which are of the same blend mode. Cannot mix blend with blend_epochized."
                )
            for loaded_dataset, weight in inner_datasets:
                if inner_blend_mode == DatasetBlendMode.DATASET_WEIGHT:
                    assert isinstance(weight, float)
                else:
                    assert weight is None
                    weight = 1.0
                datasets.append((loaded_dataset, weight * dataset.weight / sum_weight))
        return DatasetBlendMode.DATASET_WEIGHT, datasets


class Metadataset(DatasetLoaderInterface):
    """Main entry for metadataset."""

    _path: EPath
    _splits: Dict[str, MetadatasetBlender]

    def __init__(
        self,
        path: Union[EPath, str],
        splits: Dict[str, MetadatasetBlender],
    ):
        """Create the metadataset"""
        self._path = EPath(path)
        self._splits = splits

    def post_initialize(self, mds_path: Optional[EPath] = None):
        assert mds_path is None
        for split in self._splits.values():
            split.post_initialize(self._path)

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavor: Optional[str] = None,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        **kwargs,
    ) -> Tuple[DatasetBlendMode, List[Tuple[BaseCoreDatasetFactory, Union[float, int, None]]]]:
        return self._splits[split_part].get_datasets(
            training=training,
            split_part=split_part,
            worker_config=worker_config,
            subflavor=subflavor,
            subflavors=subflavors,
            shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
            **kwargs,
        )
