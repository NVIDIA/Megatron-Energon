# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, Generic, Iterator, Tuple, TypeVar, Union

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.wrappers.base import BaseSingleWrapperDataset

T_sample = TypeVar("T_sample")


class FilterDataset(BaseSingleWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """This dataset wrapper applies a custom filter function to each sample and does not yield
    filtered samples."""

    filter_fn: Callable[[T_sample], bool]

    def __init__(self, dataset: SavableDataset[T_sample], filter_fn: Callable[[T_sample], bool]):
        """Construct a MapDataset.

        Args:
            dataset: The input dataset to wrap
            filter_fn: The function to apply to each sample. If it returns `True`, the sample is
               accepted.
        """
        super().__init__(dataset)
        self.filter_fn = filter_fn

    def __len__(self):
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_sample]:
        for sample in self.dataset:
            if self.filter_fn(sample):
                yield sample

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "filter_fn": self._function_config(self.filter_fn),
        }

    def __str__(self):
        return f"FilterDataset(filter_fn={self.filter_fn}, dataset={self.dataset})"
