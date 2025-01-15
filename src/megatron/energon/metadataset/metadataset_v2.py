# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

from megatron.energon.dataset_config import load_config
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseCoreDatasetFactory, Sample
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from megatron.energon.metadataset.dataset_loader import DatasetLoader
from megatron.energon.metadataset.join_dataset_loader import JoinDatasetLoader
from megatron.energon.metadataset.loader_interface import DatasetBlendMode, DatasetLoaderInterface
from megatron.energon.worker import WorkerConfig


@dataclass
class DatasetReference(DatasetLoaderInterface):
    path: Union[str, EPath]

    split_part: Optional[str] = None
    subflavor: Optional[str] = None
    subflavors: Optional[Dict[str, Any]] = None
    shuffle_over_epochs_multiplier: int = 1
    dataset_config: str = "dataset.yaml"
    split_config: str = "split.yaml"

    _dataset: Optional[DatasetLoaderInterface] = None

    def prepare(self, parent_path: EPath) -> None:
        self.path = parent_path.absolute() / self.path
        if self.path.is_file():
            assert self.dataset_config == "dataset.yaml", "Must not set dataset_config"
            assert self.split_config == "split.yaml", "Must not set split_config"
            self._dataset = load_config(
                self.path,
                default_type=MetadatasetV2,
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


@dataclass
class JoinDatasetReference(DatasetReference):
    def prepare(self, parent_path: EPath) -> DatasetLoader:
        # Override and disable another metadataset reference, only allow direct dataset references.
        # Do not store the loader, the parent MetadatasetJoin will do that.
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

    def get_datasets(
        self,
        **kwargs,
    ) -> Tuple[DatasetBlendMode, List[Tuple[BaseCoreDatasetFactory, Union[float, int, None]]]]:
        assert (
            False
        ), "JoinDatasetReference should not be used directly, but only by MetadatasetJoin"


@dataclass
class MetadatasetJoin(DatasetLoaderInterface):
    join: Union[List[JoinDatasetReference], Dict[str, JoinDatasetReference]]
    joiner: Union[Type[Sample], Callable[..., Sample]]
    join_method: Literal["inner_match", "inner", "left"] = "inner_match"

    split_part: Optional[str] = None
    subflavor: Optional[str] = None
    subflavors: Optional[Dict[str, Any]] = None
    shuffle_over_epochs_multiplier: int = 1
    dataset_config: str = "dataset.yaml"
    split_config: str = "split.yaml"

    _dataset: Optional[DatasetLoaderInterface] = None

    def prepare(self, parent_path: EPath):
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
    ) -> Tuple[DatasetBlendMode, List[Tuple[BaseCoreDatasetFactory, Union[float, int, None]]]]:
        assert self._dataset is not None, "Not prepared."
        return self._dataset.get_datasets(
            training=training,
            split_part=split_part,
            worker_config=worker_config,
            subflavor=subflavor,
            subflavors=subflavors,
            shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
            **kwargs,
        )


@dataclass
class BlendWeightMixin:
    weight: float = 1.0


@dataclass
class BlendDatasetReference(BlendWeightMixin, DatasetReference):
    pass


@dataclass
class BlendJoinDatasetReference(BlendWeightMixin, MetadatasetJoin):
    pass


@dataclass
class MetadatasetBlend(DatasetLoaderInterface):
    """Blending of datasets by specifying the sampling weight for the inner datasets."""

    blend: List[Union[BlendDatasetReference, BlendJoinDatasetReference]]

    def prepare(self, parent_path: EPath):
        parent_path = parent_path.absolute()
        for dataset in self.blend:
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
    ) -> Tuple[DatasetBlendMode, List[Tuple[BaseCoreDatasetFactory, Union[float, int, None]]]]:
        sum_weight = sum(dataset.weight for dataset in self.blend)
        datasets = []
        for dataset in self.blend:
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


@dataclass
class BlendRepetitionsMixin:
    repetitions: int = 1


@dataclass
class BlendEpochizedDatasetReference(BlendRepetitionsMixin, DatasetReference):
    pass


@dataclass
class BlendEpochizedJoinDatasetReference(BlendRepetitionsMixin, MetadatasetJoin):
    pass


@dataclass
class MetadatasetBlendEpochized(DatasetLoaderInterface):
    """Blending of datasets, by specifying the number of repetitions for samples from the inner
    datasets. Ensures that the constraint, that samples are seen exactly this many times before
    repeating the "epoch" (i.e. one epoch contains the total number of repetitions for each inner
    dataset)."""

    blend_epochized: List[Union[BlendEpochizedDatasetReference, BlendEpochizedJoinDatasetReference]]

    def prepare(self, parent_path: EPath):
        parent_path = parent_path.absolute()
        for dataset in self.blend_epochized:
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
    ) -> Tuple[DatasetBlendMode, List[Tuple[BaseCoreDatasetFactory, Union[float, int, None]]]]:
        datasets = []
        for dataset in self.blend_epochized:
            inner_blend_mode, inner_datasets = dataset.get_datasets(
                training=training,
                split_part=split_part,
                worker_config=worker_config,
                subflavor=subflavor,
                subflavors=subflavors,
                shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
                **kwargs,
            )
            if inner_blend_mode not in (DatasetBlendMode.NONE, DatasetBlendMode.SAMPLE_REPETITIONS):
                raise ValueError(
                    "Can only blend datasets which are of the same blend mode. Cannot mix blend with blend_epochized."
                )
            for loaded_dataset, repetitions in inner_datasets:
                if inner_blend_mode == DatasetBlendMode.SAMPLE_REPETITIONS:
                    assert isinstance(repetitions, int)
                else:
                    assert repetitions is None
                    repetitions = 1
                datasets.append((loaded_dataset, dataset.repetitions * repetitions))
        return DatasetBlendMode.SAMPLE_REPETITIONS, datasets


@dataclass
class MetadatasetV2(DatasetLoaderInterface):
    parent_path: Union[EPath, str]
    splits: Dict[
        str, Union[MetadatasetBlend, MetadatasetBlendEpochized, MetadatasetJoin, DatasetReference]
    ]

    def __post_init__(self):
        """Post-initialization to fix paths."""
        self.parent_path = EPath(self.parent_path).absolute()
        # Fix paths
        for split in self.splits.values():
            split.prepare(self.parent_path)

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
    ) -> Tuple[DatasetBlendMode, List[Tuple[BaseCoreDatasetFactory, Union[float, int, None]]]]:
        return self.splits[split_part].get_datasets(
            training=training,
            split_part=split_part,
            worker_config=worker_config,
            subflavor=subflavor,
            subflavors=subflavors,
            shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
            **kwargs,
        )
