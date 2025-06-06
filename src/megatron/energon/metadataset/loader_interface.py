# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from megatron.energon.cache import FileStore
from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import BaseCoreDatasetFactory
from megatron.energon.worker import WorkerConfig


class DatasetBlendMode(Enum):
    """Determines how the the datasets are to be blended. Either by using the associated number as
    the weight for sampling from that dataset, or alternatively by using the number as the number
    of repetitions for samples in that dataset in one epoch (effectively, that corresponds to the
    weight for samples)."""

    NONE = "none"
    DATASET_WEIGHT = "dataset_weight"
    SAMPLE_REPETITIONS = "sample_repetitions"


@edataclass
class LoadedDataset:
    dataset: BaseCoreDatasetFactory
    weight: Union[float, int, None] = None
    repetitions: Union[float, int, None] = None
    aux: Optional[Dict[str, FileStore]] = None


@edataclass
class LoadedDatasetList:
    datasets: List[LoadedDataset]
    blend_mode: DatasetBlendMode = DatasetBlendMode.NONE


class DatasetLoaderInterface(ABC):
    """General interface for a dataset loader."""

    @abstractmethod
    def post_initialize(self, mds_path: Optional[EPath] = None):
        """Called to finally initialize the dataset."""
        ...

    @abstractmethod
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
        """
        Calls :func:`megatron.energon.dataset_config.get_dataset_from_config` (loads the raw dataset)
        for all innermost datasets and resolves their relative weights to absolute weights.

        Args:
            training: If true, apply training randomization.
            split_part: Default split part to use.
            worker_config: Worker configuration to use.
            subflavors: Set the default subflavors for all datasets.
            shuffle_over_epochs_multiplier: Multiply the inner datasets
                `shuffle_over_epochs(_multiplier)` by this factor. E.g. if the inner dataset
                has `shuffle_over_epochs_multiplier=2` and this function has
                `shuffle_over_epochs_multiplier=3`, the inner dataset will be shuffled
                over 6 epochs. Shuffling over `n` epochs guarantees that each sample is seen
                exactly `n` times in `n` epochs of the inner dataset. Use -1 for shuffling over
                an infinite number of epochs (effectively, this will draw shard slices with
                replacement).
            **kwargs: Additional arguments to the dataset constructor.

        Returns:
            The dataset blending mode and the instantiated core datasets with their weights/repetitions.
        """
        ...

    def prepare(self, split_part: Optional[str] = None) -> Sequence[EPath]:
        """
        Prepares the loader by creating caches and other necessary structures on disk.

        Args:
            split_part: Name of the split to load.

        Returns:
            List of paths to the cache paths. This is used for cleanup.
        """
        return ()
