# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

from megatron.energon.dataset_config import MAIN_FOLDER_NAME, load_config
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import BaseCoreDataset, Sample
from megatron.energon.metadataset.dataset_loader import DatasetLoader
from megatron.energon.metadataset.join_dataset_loader import JoinDatasetLoader
from megatron.energon.metadataset.loader_interface import DatasetLoaderInterface
from megatron.energon.worker import WorkerConfig


@dataclass
class JoinDatasetReference:
    path: Union[str, EPath]

    split_part: Optional[str] = None
    subflavor: Optional[str] = None
    subflavors: Optional[Dict[str, Any]] = None
    shuffle_over_epochs_multiplier: int = 1
    dataset_config: str = "dataset.yaml"
    split_config: str = "split.yaml"

    def prepare(self, parent_path: EPath) -> DatasetLoader:
        self.path = parent_path.absolute() / self.path
        if (self.path / MAIN_FOLDER_NAME / ".info.yaml").is_file():
            return DatasetLoader(
                path=self.path,
                split_part=self.split_part,
                subflavor=self.subflavor,
                subflavors=self.subflavors,
                shuffle_over_epochs_multiplier=self.shuffle_over_epochs_multiplier,
                dataset_config=self.dataset_config,
                split_config=self.split_config,
            )
        else:
            raise FileNotFoundError(self.path)


@dataclass
class DatasetReference:
    path: Optional[Union[str, EPath]] = None
    join: Union[List[JoinDatasetReference], Dict[str, JoinDatasetReference], None] = None
    join_method: Literal["inner_match", "inner", "left"] = "inner_match"
    joiner: Optional[Union[Type[Sample], Callable[..., Sample]]] = None
    split_part: Optional[str] = None
    subflavor: Optional[str] = None
    subflavors: Optional[Dict[str, Any]] = None
    shuffle_over_epochs_multiplier: int = 1
    dataset_config: str = "dataset.yaml"
    split_config: str = "split.yaml"

    weight: float = 1.0

    _dataset: Optional[DatasetLoaderInterface] = None

    def prepare(self, parent_path: EPath):
        assert (self.path is None) != (self.join is None), "Must set path or join key"
        if self.path is not None:
            assert self.join is None
            assert self.joiner is None, "Must not set joiner for single datasets"
            assert self.join_method == "inner_match", "Must not set join_method for single datasets"
            self.path = parent_path.absolute() / self.path
            if self.path.is_file():
                assert self.dataset_config == "dataset.yaml", "Must not set dataset_config"
                assert self.split_config == "split.yaml", "Must not set split_config"
                self._dataset = load_config(
                    self.path,
                    default_type=Metadataset,
                    strict=True,
                    default_kwargs=dict(parent_path=self.path.parent),
                )
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
            else:
                raise FileNotFoundError(self.path)
        else:
            assert self.join is not None
            assert self.joiner is not None, "Must set joiner for joining datasets"
            assert (
                self.dataset_config == "dataset.yaml"
            ), "Cannot set dataset_config for joining datasets"
            if isinstance(self.join, list):
                inner_loaders = [join.prepare(parent_path) for join in self.join]
            elif isinstance(self.join, dict):
                inner_loaders = {key: join.prepare(parent_path) for key, join in self.join.items()}
            else:
                raise ValueError("Invalid join type")
            self._dataset = JoinDatasetLoader(
                datasets=inner_loaders,
                join_method=self.join_method,
                joiner=self.joiner,
                split_part=self.split_part,
                subflavor=self.subflavor,
                subflavors=self.subflavors,
                shuffle_over_epochs_multiplier=self.shuffle_over_epochs_multiplier,
                split_config=self.split_config,
            )

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavor: Optional[str] = None,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: int = 1,
        **kwargs,
    ) -> List[Tuple[BaseCoreDataset, float]]:
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


@dataclass
class MetadatasetMixer:
    """Internal mixer for the dataset."""

    datasets: List[DatasetReference]

    def prepare(self, parent_path: EPath):
        parent_path = parent_path.absolute()
        for dataset in self.datasets:
            dataset.prepare(parent_path)

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavor: Optional[str] = None,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: int = 1,
        **kwargs,
    ) -> List[Tuple[BaseCoreDataset, float]]:
        sum_weight = sum(dataset.weight for dataset in self.datasets)
        return [
            (loaded_dataset, weight * dataset.weight / sum_weight)
            for dataset in self.datasets
            for loaded_dataset, weight in dataset.get_datasets(
                training=training,
                split_part=split_part,
                worker_config=worker_config,
                subflavor=subflavor,
                subflavors=subflavors,
                shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
                **kwargs,
            )
        ]


class Metadataset(DatasetLoaderInterface):
    """Main entry for metadataset."""

    _splits: Dict[str, MetadatasetMixer]

    def __init__(
        self,
        parent_path: Union[EPath, str],
        splits: Dict[str, MetadatasetMixer],
    ):
        """Create the metadataset"""
        parent_path = EPath(parent_path).absolute()
        self._splits = splits
        # Fix paths
        for split in splits.values():
            split.prepare(parent_path)

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavor: Optional[str] = None,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: int = 1,
        **kwargs,
    ) -> List[Tuple[BaseCoreDataset, float]]:
        return self._splits[split_part].get_datasets(
            training=training,
            split_part=split_part,
            worker_config=worker_config,
            subflavor=subflavor,
            subflavors=subflavors,
            shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
            **kwargs,
        )
