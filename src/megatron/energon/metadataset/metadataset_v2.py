# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Type, Union

from megatron.energon.cache import FileStore
from megatron.energon.cache.file_store import (
    SystemFileStore,
    WebdatasetFileStore,
)
from megatron.energon.dataset_config import load_config
from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import Sample
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.metadata import check_dataset_info_present
from megatron.energon.metadataset.dataset_loader import DatasetLoader
from megatron.energon.metadataset.join_dataset_loader import JoinDatasetLoader, JoinedDatasetInfo
from megatron.energon.metadataset.loader_interface import (
    DatasetBlendMode,
    DatasetLoaderInterface,
    LoadedDatasetList,
)
from megatron.energon.metadataset.metadataset import Metadataset
from megatron.energon.worker import WorkerConfig

# Regex for any URL-like string (any protocol)
url_regex = re.compile(r"^(?P<protocol>[a-z][a-z0-9+.-]*)://(?P<path>.*)", re.IGNORECASE)


@edataclass
class AuxDatasetReference:
    path: Union[str, EPath]

    def post_initialize(self, mds_path: Optional[EPath] = None) -> None:
        assert mds_path is not None
        if not isinstance(self.path, EPath):
            self.path = mds_path.parent / self.path
        assert not self.path.is_file(), (
            "Auxiliary datasets must not be metadataset, but direct dataset references"
        )
        assert (self.path / MAIN_FOLDER_NAME / "index.sqlite").is_file(), (
            "Auxiliary datasets must be prepared Energon dataset"
        )

    def get_file_store(self) -> FileStore:
        assert isinstance(self.path, EPath), "Missing call to post_initialize"
        return WebdatasetFileStore(self.path)


@edataclass
class AuxFilesystemReference:
    fs_path: Union[str, EPath]

    def post_initialize(self, mds_path: Optional[EPath] = None) -> None:
        assert mds_path is not None
        if not isinstance(self.fs_path, EPath):
            self.fs_path = mds_path.parent / self.fs_path

    def get_file_store(self) -> FileStore:
        assert isinstance(self.fs_path, EPath), "Missing call to post_initialize"
        return SystemFileStore(self.fs_path)


@edataclass
class DatasetReference(DatasetLoaderInterface):
    path: Union[str, EPath]

    split_part: Optional[str] = None
    subflavors: Optional[Dict[str, Any]] = None
    shuffle_over_epochs_multiplier: Optional[int] = 1
    dataset_config: str = "dataset.yaml"
    split_config: str = "split.yaml"

    #: Auxiliary datasets. May only be specified for crude datasets for cooking. Cooking will get
    # these references to load data from. If specified as string, it will be interpreted as a
    # dataset path.
    aux: Optional[Dict[str, str]] = None

    _dataset: Optional[DatasetLoaderInterface] = None

    def post_initialize(self, mds_path: Optional[EPath] = None) -> None:
        assert mds_path is not None
        if not isinstance(self.path, EPath):
            self.path = mds_path.parent / self.path
        if self.path.is_file():
            assert self.aux is None, "Cannot specify auxiliary datasets for crude datasets"
            assert self.dataset_config == "dataset.yaml", "Must not set dataset_config"
            assert self.split_config == "split.yaml", "Must not set split_config"
            # Note: For backwards compatibility, the type must be Metadataset (V1).
            self._dataset = load_config(
                self.path,
                default_type=Metadataset,
                default_kwargs=dict(path=self.path),
            )
            self._dataset.post_initialize()
        elif check_dataset_info_present(self.path):
            self._dataset = DatasetLoader(path=self.path)
            self._dataset.post_initialize()
            if self.aux is not None:
                new_aux = {}
                for k, v in self.aux.items():
                    if m := url_regex.match(v):
                        if m.group("protocol") == "filesystem":
                            new_aux[k] = AuxFilesystemReference(fs_path=m.group("path"))
                        else:
                            raise ValueError(f"Unsupported protocol: {m.group('protocol')}")
                    else:
                        new_aux[k] = AuxDatasetReference(path=v)

                    new_aux[k].post_initialize(mds_path)
                self.aux = new_aux
        else:
            raise FileNotFoundError(self.path)

    def prepare(self, split_part: Optional[str] = None) -> Sequence[EPath]:
        assert self._dataset is not None
        return self._dataset.prepare(split_part=split_part)

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
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
        result = self._dataset.get_datasets(
            training=training,
            split_part=self.split_part or split_part,
            worker_config=worker_config,
            subflavors=subflavors,
            shuffle_over_epochs_multiplier=new_shuffle_over_epochs_multiplier,
            **kwargs,
        )
        if self.aux is not None:
            aux = {k: v.get_file_store() for k, v in self.aux.items()}
            for loaded_dataset in result.datasets:
                if loaded_dataset.aux is None:
                    loaded_dataset.aux = aux
                else:
                    loaded_dataset.aux.update(aux)
        return result


@edataclass
class JoinDatasetReference(DatasetReference):
    nonmatch: Literal["skip", "none", "error"] = "error"

    def post_initialize(self, mds_path: Optional[EPath] = None) -> DatasetLoader:
        assert mds_path is not None
        # Override and disable another metadataset reference, only allow direct dataset references.
        # Do not store the loader, the parent MetadatasetJoin will do that.
        if not isinstance(self.path, EPath):
            self.path = mds_path.parent / self.path
        if check_dataset_info_present(self.path):
            return DatasetLoader(
                path=self.path,
                split_part=self.split_part,
                subflavors=self.subflavors,
                shuffle_over_epochs_multiplier=self.shuffle_over_epochs_multiplier,
                dataset_config=self.dataset_config,
                split_config=self.split_config,
            )
        else:
            raise FileNotFoundError(self.path)

    def prepare(self, split_part: Optional[str] = None):
        assert False, (
            "JoinDatasetReference should not be used directly, but only by MetadatasetJoin"
        )

    def get_datasets(
        self,
        **kwargs,
    ) -> LoadedDatasetList:
        assert False, (
            "JoinDatasetReference should not be used directly, but only by MetadatasetJoin"
        )


@edataclass
class MetadatasetJoin(DatasetLoaderInterface):
    join: Union[List[JoinDatasetReference], Dict[str, JoinDatasetReference]]
    joiner: Union[Type[Sample], Callable[..., Sample]]

    split_part: Optional[str] = None
    subflavors: Optional[Dict[str, Any]] = None
    shuffle_over_epochs_multiplier: Optional[int] = 1
    dataset_config: str = "dataset.yaml"
    split_config: str = "split.yaml"

    _dataset: Optional[JoinDatasetLoader] = None

    def post_initialize(self, mds_path: Optional[EPath] = None):
        assert mds_path is not None
        assert self.join is not None
        assert self.joiner is not None, "Must set joiner for joining datasets"
        assert self.dataset_config == "dataset.yaml", (
            "Cannot set dataset_config for joining datasets"
        )
        if isinstance(self.join, list):
            inner_loaders = [
                JoinedDatasetInfo(
                    dataset=join.post_initialize(mds_path),
                    nonmatch=join.nonmatch,
                )
                for join in self.join
            ]
        elif isinstance(self.join, dict):
            inner_loaders = {
                key: JoinedDatasetInfo(
                    dataset=join.post_initialize(mds_path),
                    nonmatch=join.nonmatch,
                )
                for key, join in self.join.items()
            }
        else:
            raise ValueError("Invalid join type")

        self._dataset = JoinDatasetLoader(
            datasets=inner_loaders,
            joiner=self.joiner,
            split_part=self.split_part,
            subflavors=self.subflavors,
            shuffle_over_epochs_multiplier=self.shuffle_over_epochs_multiplier,
            split_config=self.split_config,
        )
        self._dataset.post_initialize(mds_path)

    def prepare(self, split_part: Optional[str] = None) -> Sequence[EPath]:
        assert self._dataset is not None, "Missing post_initialize call."
        return self._dataset.prepare(split_part=split_part)

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        **kwargs,
    ) -> LoadedDatasetList:
        assert self._dataset is not None, "Missing post_initialize call."
        return self._dataset.get_datasets(
            training=training,
            split_part=split_part,
            worker_config=worker_config,
            subflavors=subflavors,
            shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
            **kwargs,
        )


@dataclass
class BlendWeightMixin:
    weight: float = 1.0


@edataclass
class BlendDatasetReference(BlendWeightMixin, DatasetReference):
    pass


@edataclass
class BlendJoinDatasetReference(BlendWeightMixin, MetadatasetJoin):
    pass


@edataclass
class MetadatasetBlend(DatasetLoaderInterface):
    """Blending of datasets by specifying the sampling weight for the inner datasets."""

    blend: List[Union[BlendDatasetReference, BlendJoinDatasetReference]]

    def post_initialize(self, mds_path: Optional[EPath] = None):
        assert mds_path is not None
        for dataset in self.blend:
            dataset.post_initialize(mds_path)

    def prepare(self, split_part: Optional[str] = None) -> Sequence[EPath]:
        files = []
        for dataset in self.blend:
            files.extend(dataset.prepare(split_part=split_part))
        return files

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        **kwargs,
    ) -> LoadedDatasetList:
        sum_weight = sum(dataset.weight for dataset in self.blend)
        datasets = []
        for dataset in self.blend:
            inner_result = dataset.get_datasets(
                training=training,
                split_part=split_part,
                worker_config=worker_config,
                subflavors=subflavors,
                shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
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
                    assert inner_result.blend_mode == DatasetBlendMode.NONE
                    assert loaded_dataset.weight is None
                    assert loaded_dataset.repetitions is None
                    loaded_dataset.weight = 1.0
                loaded_dataset.weight = loaded_dataset.weight * dataset.weight / sum_weight
                datasets.append(loaded_dataset)
        return LoadedDatasetList(
            blend_mode=DatasetBlendMode.DATASET_WEIGHT,
            datasets=datasets,
        )


@dataclass
class BlendRepetitionsMixin:
    repetitions: Union[int, float] = 1


@edataclass
class BlendEpochizedDatasetReference(BlendRepetitionsMixin, DatasetReference):
    pass


@edataclass
class BlendEpochizedJoinDatasetReference(BlendRepetitionsMixin, MetadatasetJoin):
    pass


@edataclass
class MetadatasetBlendEpochized(DatasetLoaderInterface):
    """Blending of datasets, by specifying the number of repetitions for samples from the inner
    datasets. Ensures that the constraint, that samples are seen exactly this many times before
    repeating the "epoch" (i.e. one epoch contains the total number of repetitions for each inner
    dataset)."""

    blend_epochized: List[Union[BlendEpochizedDatasetReference, BlendEpochizedJoinDatasetReference]]

    def post_initialize(self, mds_path: Optional[EPath] = None):
        assert mds_path is not None
        for dataset in self.blend_epochized:
            dataset.post_initialize(mds_path)

    def prepare(self, split_part: Optional[str] = None) -> Sequence[EPath]:
        files = []
        for dataset in self.blend_epochized:
            files.extend(dataset.prepare(split_part=split_part))
        return files

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        **kwargs,
    ) -> LoadedDatasetList:
        datasets = []
        for dataset in self.blend_epochized:
            inner_result = dataset.get_datasets(
                training=training,
                split_part=split_part,
                worker_config=worker_config,
                subflavors=subflavors,
                shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
                **kwargs,
            )
            if inner_result.blend_mode not in (
                DatasetBlendMode.NONE,
                DatasetBlendMode.SAMPLE_REPETITIONS,
            ):
                raise ValueError(
                    "Can only blend datasets which are of the same blend mode. Cannot mix blend with blend_epochized."
                )
            for loaded_dataset in inner_result.datasets:
                if inner_result.blend_mode == DatasetBlendMode.SAMPLE_REPETITIONS:
                    assert isinstance(loaded_dataset.repetitions, (int, float))
                else:
                    assert loaded_dataset.weight is None
                    assert loaded_dataset.repetitions is None
                    loaded_dataset.repetitions = 1
                loaded_dataset.repetitions = dataset.repetitions * loaded_dataset.repetitions
                datasets.append(loaded_dataset)
        return LoadedDatasetList(
            blend_mode=DatasetBlendMode.SAMPLE_REPETITIONS,
            datasets=datasets,
        )


@edataclass
class MetadatasetV2(DatasetLoaderInterface):
    path: EPath
    splits: Dict[
        str, Union[MetadatasetBlend, MetadatasetBlendEpochized, MetadatasetJoin, DatasetReference]
    ]

    def post_initialize(self, mds_path: Optional[EPath] = None):
        assert mds_path is None
        for split in self.splits.values():
            split.post_initialize(self.path)

    def prepare(self, split_part: Optional[str] = None) -> Sequence[EPath]:
        # In the case of prepare for MetadatasetV2, we ignore the passed cache_path
        # and instead use the own path.
        # If someone runs energon prepare on a metadataset that refers to another metadataset,
        # any actions concerning the inner metadataset will be done on the inner metadataset's path.

        if split_part is None:
            files = []
            for split_part, split in self.splits.items():
                files.extend(split.prepare(split_part=split_part))
        else:
            files = self.splits[split_part].prepare(split_part=split_part)
        # Cleanup paths here
        cache_path = EPath(self.path.parent / f"{self.path.name}.cache")
        remove_files = set(cache_path.glob("*")) - set(files)
        for file in remove_files:
            print(f"Cleanup cache file {file}")
            file.unlink()
        # Do not return paths. This is a barrier
        return ()

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        **kwargs,
    ) -> LoadedDatasetList:
        return self.splits[split_part].get_datasets(
            training=training,
            split_part=split_part,
            worker_config=worker_config,
            subflavors=subflavors,
            shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
            **kwargs,
        )
