# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

from megatron.energon.flavors import (
    BaseCoreDatasetFactory,
    BaseWebdatasetFactory,
    JoinedWebdatasetFactory,
    Sample,
)
from megatron.energon.metadataset.dataset_loader import DatasetLoader
from megatron.energon.metadataset.loader_interface import DatasetBlendMode, DatasetLoaderInterface
from megatron.energon.worker import WorkerConfig


@dataclass
class JoinDatasetLoader(DatasetLoaderInterface):
    """Loads a joined dataset from a path."""

    datasets: Union[List[DatasetLoader], Dict[str, DatasetLoader]]
    joiner: Union[Type[Sample], Callable[..., Sample]]
    join_method: Literal["inner_match", "inner", "left"] = "inner_match"

    split_part: Optional[str] = None
    split_config: Optional[str] = None
    subflavor: Optional[str] = None
    subflavors: Optional[Dict[str, Any]] = None
    shuffle_over_epochs_multiplier: int = 1

    def get_dataset(
        self,
        *,
        training: bool,
        split_part: Optional[str] = None,
        worker_config: WorkerConfig,
        subflavor: Optional[str] = None,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs: int = 1,
        split_config: Optional[str] = None,
        **kwargs,
    ) -> BaseCoreDatasetFactory:
        """
        Args:
            training: If true, apply training randomization.
            split_part: Default split part to use.
            worker_config: Worker configuration.
            shuffle_buffer_size: Size of the sample shuffle buffer (before task encoding).
            subflavor: Subflavor to use, might be overridden by inner datasets.
            subflavors: Subflavors to use, might be overridden by inner datasets.
            shuffle_over_epochs: Shuffle the dataset over this many epochs.
            **kwargs: Additional arguments to the dataset constructor.

        Returns:
            The loaded dataset
        """
        if self.split_config is not None:
            split_config = self.split_config
        if self.split_part is not None:
            split_part = self.split_part
        if split_part is None:
            raise ValueError("Missing split part")
        if subflavor is None:
            subflavor = self.subflavor
        if self.subflavors is not None:
            subflavors = {**self.subflavors, **(subflavors or {})}
        if isinstance(self.datasets, list):
            inner_datasets = [
                dataset.get_dataset(
                    training=training,
                    split_part=split_part,
                    worker_config=worker_config,
                    subflavor=subflavor,
                    subflavors=subflavors,
                    shuffle_over_epochs=shuffle_over_epochs,
                    split_config=split_config,
                    **kwargs,
                )
                for dataset in self.datasets
            ]
            assert all(
                isinstance(d, BaseWebdatasetFactory) for d in inner_datasets
            ), "Can only merge webdatasets efficiently"
        elif isinstance(self.datasets, dict):
            inner_datasets = {
                key: dataset.get_dataset(
                    training=training,
                    split_part=split_part,
                    worker_config=worker_config,
                    subflavor=subflavor,
                    subflavors=subflavors,
                    shuffle_over_epochs=shuffle_over_epochs,
                    split_config=split_config,
                    **kwargs,
                )
                for key, dataset in self.datasets.items()
            }
            assert all(
                isinstance(d, BaseWebdatasetFactory) for d in inner_datasets.values()
            ), "Can only merge webdatasets efficiently"
        else:
            raise ValueError("Invalid join type")
        return JoinedWebdatasetFactory(
            inner_datasets=inner_datasets,
            training=training,
            worker_config=worker_config,
            shuffle_over_epochs=shuffle_over_epochs,
            join_method=self.join_method,
            joiner=self.joiner,
            **kwargs,
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
        return DatasetBlendMode.NONE, [
            (
                self.get_dataset(
                    training=training,
                    split_part=split_part,
                    worker_config=worker_config,
                    subflavor=subflavor,
                    subflavors=subflavors,
                    shuffle_over_epochs=shuffle_over_epochs_multiplier,
                    **kwargs,
                ),
                None,
            )
        ]
