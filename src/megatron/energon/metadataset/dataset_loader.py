# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Literal, Optional, Union

from megatron.energon.dataset_config import get_dataset_from_config
from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseCoreDatasetFactory
from megatron.energon.metadataset.loader_interface import (
    DatasetBlendMode,
    DatasetLoaderInterface,
    LoadedDataset,
    LoadedDatasetList,
)
from megatron.energon.worker import WorkerConfig


@edataclass
class DatasetLoader(DatasetLoaderInterface):
    """Loads a dataset from a path."""

    path: Union[str, EPath]
    split_part: Optional[str] = None
    subflavors: Optional[Dict[str, Any]] = None
    shuffle_over_epochs_multiplier: Optional[int] = 1
    dataset_config: str = "dataset.yaml"
    split_config: str = "split.yaml"

    def post_initialize(self, mds_path: Optional[EPath] = None):
        pass

    def get_dataset(
        self,
        *,
        training: bool,
        split_part: Optional[str] = None,
        worker_config: WorkerConfig,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs: Optional[int] = 1,
        split_config: Optional[str] = None,
        dataset_config: Optional[str] = None,
        **kwargs,
    ) -> BaseCoreDatasetFactory:
        """
        Args:
            training: If true, apply training randomization.
            split_part: Default split part to use.
            worker_config: Worker configuration.
            shuffle_buffer_size: Size of the sample shuffle buffer (before task encoding).
            subflavors: Subflavors to use, might be overridden by inner datasets.
            shuffle_over_epochs: Shuffle the dataset over this many epochs.
            **kwargs: Additional arguments to the dataset constructor.

        Returns:
            The loaded dataset
        """
        if self.split_part is not None:
            split_part = self.split_part
        if split_part is None:
            raise ValueError("Missing split part")
        if self.subflavors is not None:
            subflavors = {**self.subflavors, **(subflavors or {})}
        if split_config is None:
            split_config = self.split_config
        if dataset_config is None:
            dataset_config = self.dataset_config
        return get_dataset_from_config(
            self.path,
            training=training,
            split_part=split_part,
            worker_config=worker_config,
            subflavors=subflavors,
            dataset_config=dataset_config,
            split_config=split_config,
            shuffle_over_epochs=shuffle_over_epochs,
            **kwargs,
        )

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
        return LoadedDatasetList(
            blend_mode=DatasetBlendMode.NONE,
            datasets=[
                LoadedDataset(
                    dataset=self.get_dataset(
                        training=training,
                        split_part=split_part,
                        worker_config=worker_config,
                        subflavors=subflavors,
                        shuffle_over_epochs=shuffle_over_epochs_multiplier,
                        **kwargs,
                    ),
                    weight=None,
                )
            ],
        )
