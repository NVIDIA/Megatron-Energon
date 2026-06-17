# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, List, Literal, Optional, Union

from megatron.energon.dataset_config import load_config
from megatron.energon.deprecation import warn_deprecated
from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.metadata import check_dataset_info_present
from megatron.energon.flavors.webdataset.structs import DatasetSubset
from megatron.energon.metadataset.dataset_loader import DatasetLoader
from megatron.energon.metadataset.loader_interface import (
    DatasetBlendMode,
    DatasetLoaderInterface,
    LoadedDatasetList,
    TraversedDatasetReference,
)
from megatron.energon.worker import WorkerConfig


@edataclass
class DatasetReference:
    path: Union[str, EPath]
    split_part: Optional[str] = None
    # Note: subflavor is only for legacy compatibility.
    subflavor: Optional[str] = None
    subflavors: Optional[Dict[str, Any]] = None
    shuffle_over_epochs_multiplier: Optional[int] = 1
    dataset_config: Optional[str] = None
    split_config: Optional[str] = None

    weight: float = 1.0

    _dataset: Optional[DatasetLoaderInterface] = None

    def __post_init__(self):
        if self.subflavor is not None:
            warn_deprecated(
                "subflavor is deprecated, use subflavors instead. This will be removed in a future release."
            )
            if self.subflavors is None:
                self.subflavors = {"__subflavor__": self.subflavor}
            elif "__subflavor__" not in self.subflavors:
                self.subflavors = {"__subflavor__": self.subflavor, **(self.subflavors or {})}
            self.subflavor = None

    def _resolve_path(self, mds_path: Optional[EPath]) -> EPath:
        assert mds_path is not None
        if not isinstance(self.path, EPath):
            self.path = mds_path.parent / self.path
        return self.path

    def _load_nested_metadataset(self) -> DatasetLoaderInterface:
        assert isinstance(self.path, EPath)
        assert self.dataset_config is None, "Must not set dataset_config"
        assert self.split_config is None, "Must not set split_config"
        return load_config(
            self.path,
            default_type=Metadataset,
            default_kwargs=dict(path=self.path),
        )

    def _merge_traversed_subflavors(
        self, inherited_subflavors: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge this reference's subflavors with the inherited traversal subflavors.

        The merge order mirrors `get_datasets(...)`: this reference contributes the base mapping,
        and inherited outer-hierarchy subflavors override on key conflicts.

        Args:
            inherited_subflavors: Effective subflavors accumulated from outer metadataset
                references during traversal.

        Returns:
            The effective subflavor mapping for this reference, after applying outer-overrides-inner
            merge semantics.
        """
        if self.subflavors is not None:
            return {**self.subflavors, **(inherited_subflavors or {})}
        return dict(inherited_subflavors or {})

    def _merge_shuffle_over_epochs_multiplier(
        self, inherited_shuffle_over_epochs_multiplier: Optional[int]
    ) -> Optional[int]:
        """Same semantics as Metadataset V2 ``ShuffleOverEpochsMultiplierMixin`` / ``get_datasets``."""
        if (
            inherited_shuffle_over_epochs_multiplier is None
            or self.shuffle_over_epochs_multiplier is None
        ):
            return None
        if (
            inherited_shuffle_over_epochs_multiplier == -1
            or self.shuffle_over_epochs_multiplier == -1
        ):
            return -1
        return inherited_shuffle_over_epochs_multiplier * self.shuffle_over_epochs_multiplier

    def post_initialize(self, mds_path: Optional[EPath] = None):
        self._resolve_path(mds_path)
        if self.path.is_file():
            self._dataset = self._load_nested_metadataset()
            self._dataset.post_initialize()
        elif check_dataset_info_present(self.path):
            self._dataset = DatasetLoader(
                path=self.path,
                split_config=self.split_config,
                dataset_config=self.dataset_config,
            )
            self._dataset.post_initialize()
        else:
            raise FileNotFoundError(self.path)

    def traverse(
        self,
        mds_path: Optional[EPath] = None,
        *,
        split_part: Union[Literal["train", "val", "test"], str],
        _group: Optional[str] = None,
        _shuffle_over_epochs_multiplier: Optional[int] = 1,
        _subflavors: Optional[Dict[str, Any]] = None,
    ) -> List[TraversedDatasetReference]:
        self._resolve_path(mds_path)
        _subflavors = self._merge_traversed_subflavors(_subflavors)
        _shuffle_over_epochs_multiplier = self._merge_shuffle_over_epochs_multiplier(
            _shuffle_over_epochs_multiplier
        )
        if self.path.is_file():
            return self._load_nested_metadataset().traverse(
                split_part=self.split_part or split_part,
                _group=_group,
                _shuffle_over_epochs_multiplier=_shuffle_over_epochs_multiplier,
                _subflavors=_subflavors,
            )
        return [
            TraversedDatasetReference(
                path=self.path,
                split_part=self.split_part or split_part,
                aux={},
                subflavors=_subflavors,
                group=_group,
                shuffle_over_epochs_multiplier=_shuffle_over_epochs_multiplier,
            )
        ]

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        subset: Optional[DatasetSubset] = None,
        group: Optional[str] = None,
        **kwargs,
    ) -> LoadedDatasetList:
        if self.subflavors is not None:
            subflavors = {**self.subflavors, **(subflavors or {})}
        assert self._dataset is not None

        if shuffle_over_epochs_multiplier is None or self.shuffle_over_epochs_multiplier is None:
            # If no shuffling is requested, this has override priority.
            new_shuffle_over_epochs_multiplier = None
        elif shuffle_over_epochs_multiplier == -1 or self.shuffle_over_epochs_multiplier == -1:
            # Next priority is sampling without replacement.
            new_shuffle_over_epochs_multiplier = -1
        else:
            # Otherwise, multiply the shuffle over epochs multiplier.
            new_shuffle_over_epochs_multiplier = (
                shuffle_over_epochs_multiplier * self.shuffle_over_epochs_multiplier
            )

        return self._dataset.get_datasets(
            training=training,
            split_part=self.split_part or split_part,
            worker_config=worker_config,
            subflavors=subflavors,
            shuffle_over_epochs_multiplier=new_shuffle_over_epochs_multiplier,
            subset=subset,
            group=group,
            **kwargs,
        )


@edataclass
class MetadatasetBlender:
    """Internal blending of the dataset."""

    datasets: List[DatasetReference]

    def post_initialize(self, mds_path: Optional[EPath] = None):
        assert mds_path is not None
        for dataset in self.datasets:
            dataset.post_initialize(mds_path)

    def traverse(
        self,
        mds_path: Optional[EPath] = None,
        *,
        split_part: Union[Literal["train", "val", "test"], str],
        _group: Optional[str] = None,
        _shuffle_over_epochs_multiplier: Optional[int] = 1,
        _subflavors: Optional[Dict[str, Any]] = None,
    ) -> List[TraversedDatasetReference]:
        assert mds_path is not None
        flattened: List[TraversedDatasetReference] = []
        for dataset in self.datasets:
            flattened.extend(
                dataset.traverse(
                    mds_path,
                    split_part=split_part,
                    _group=_group,
                    _shuffle_over_epochs_multiplier=_shuffle_over_epochs_multiplier,
                    _subflavors=_subflavors,
                )
            )
        return flattened

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        subset: Optional[DatasetSubset] = None,
        group: Optional[str] = None,
        **kwargs,
    ) -> LoadedDatasetList:
        sum_weight = sum(dataset.weight for dataset in self.datasets)
        datasets = []
        for dataset in self.datasets:
            inner_result = dataset.get_datasets(
                training=training,
                split_part=split_part,
                worker_config=worker_config,
                subflavors=subflavors,
                shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
                subset=subset,
                group=group,
                **kwargs,
            )
            if inner_result.blend_mode not in (
                DatasetBlendMode.NONE,
                DatasetBlendMode.DATASET_WEIGHT,
            ):
                raise ValueError(
                    "Can only blend datasets which are of the same blend mode. Cannot mix blend with blend_epochized."
                )
            for loaded_dataset in inner_result.datasets:
                if inner_result.blend_mode == DatasetBlendMode.DATASET_WEIGHT:
                    assert isinstance(loaded_dataset.weight, float)
                else:
                    assert loaded_dataset.weight is None
                    loaded_dataset.weight = 1.0
                loaded_dataset.weight = loaded_dataset.weight * dataset.weight / sum_weight
                datasets.append(loaded_dataset)
        return LoadedDatasetList(
            blend_mode=DatasetBlendMode.DATASET_WEIGHT,
            datasets=datasets,
        )


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

    def traverse(
        self,
        mds_path: Optional[EPath] = None,
        *,
        split_part: Union[Literal["train", "val", "test"], str],
        _group: Optional[str] = None,
        _shuffle_over_epochs_multiplier: Optional[int] = 1,
        _subflavors: Optional[Dict[str, Any]] = None,
    ) -> List[TraversedDatasetReference]:
        assert mds_path is None
        return self._splits[split_part].traverse(
            self._path,
            split_part=split_part,
            _group=_group,
            _shuffle_over_epochs_multiplier=_shuffle_over_epochs_multiplier,
            _subflavors=_subflavors,
        )

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        subset: Optional[DatasetSubset] = None,
        group: Optional[str] = None,
        **kwargs,
    ) -> LoadedDatasetList:
        return self._splits[split_part].get_datasets(
            training=training,
            split_part=split_part,
            worker_config=worker_config,
            subflavors=subflavors,
            shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
            subset=subset,
            group=group,
            **kwargs,
        )
