# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from megatron.energon.cache import FileStore
from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import BaseCoreDatasetFactory
from megatron.energon.flavors.webdataset.structs import DatasetSubset
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
    #: The dataset factory.
    dataset: BaseCoreDatasetFactory
    #: Sampling weight when using dataset-weight blending.
    weight: Union[float, int, None] = None
    #: Epochized repetition count when using repetition-based blending.
    repetitions: Union[float, int, None] = None
    #: Packing group key for train pipelines with ``packing_buffer_size`` (see Metadataset V2).
    #: ``None`` is the default group.
    packing_group: Optional[str] = None
    #: Auxiliary datasets for crude cooking.
    aux: Optional[Dict[str, FileStore]] = None


@edataclass
class LoadedDatasetList:
    datasets: List[LoadedDataset]
    blend_mode: DatasetBlendMode = DatasetBlendMode.NONE


@dataclass
class TraversedDatasetReference:
    """Flattened leaf dataset reference produced by metadataset traversal.

    Attributes:
        path: Resolved path to the referenced leaf dataset.
        split_part: Effective split part to use when loading the leaf dataset.
        aux: Resolved auxiliary dataset or filesystem references keyed by auxiliary name.
        subflavors: Effective subflavors implied by the traversed metadataset hierarchy.
        packing_group: Optional packing group name from metadataset references (for tooling).
        shuffle_over_epochs_multiplier: Effective shuffle over epochs multiplier from metadataset references.
    """

    path: EPath
    split_part: str
    aux: dict[str, EPath]
    subflavors: dict[str, Any]
    packing_group: Optional[str] = None
    shuffle_over_epochs_multiplier: Optional[int] = 1


class DatasetLoaderInterface(ABC):
    """General interface for a dataset loader."""

    @abstractmethod
    def post_initialize(self, mds_path: Optional[EPath] = None):
        """Called to finally initialize the dataset."""
        ...

    def traverse(
        self,
        mds_path: Optional[EPath] = None,
        *,
        split_part: Union[Literal["train", "val", "test"], str],
        _packing_group: Optional[str] = None,
        _shuffle_over_epochs_multiplier: Optional[int] = 1,
        _subflavors: Optional[Dict[str, Any]] = None,
    ) -> List[TraversedDatasetReference]:
        """Traverse a metadataset subtree and collect flattened leaf dataset references.

        This method is the traversal-side counterpart to `get_datasets()`. Instead of
        instantiating dataset loaders for leaf datasets, it walks the hierarchy, resolves nested
        metadataset references, and returns the final leaf dataset references for a single split.

        Args:
            mds_path: Parent metadataset path used internally to resolve relative dataset and
                auxiliary paths. Must be set for nested references and inner traversal nodes;
                use None only for top-level metadatasets.
            split_part: Split to traverse, such as `\"train\"`, `\"val\"`, or `\"test\"`. Nested
                references may override this with their own configured split.
            _packing_group: Effective packing group from metadataset references (for tooling).
            _shuffle_over_epochs_multiplier: Inherited shuffle multiplier (merged per node like
                ``get_datasets``); default ``1``.
            _subflavors: Effective subflavors implied by the traversed metadataset hierarchy.

        Returns:
            A flattened list of `TraversedDatasetReference` values for all leaf datasets reached
            during the traversal.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement traverse()")

    @abstractmethod
    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        subset: Optional[DatasetSubset] = None,
        packing_group: Optional[str] = None,
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
            subset: If specified, the inner dataset(s) will be subsetted.
            packing_group: When set, this leaf belongs to a packing group: samples from datasets sharing the same key
                are blended and shuffled together, then packed in one :class:`~megatron.energon.PackingDataset`
                buffer before blending across groups. ``None`` means the default group.
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
