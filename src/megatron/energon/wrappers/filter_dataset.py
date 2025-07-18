# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, Generic, Iterator, Optional, TypeVar, Union

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset, SampleIndex

T_sample = TypeVar("T_sample")


class FilterDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """This dataset wrapper applies a custom filter function to each sample and does not yield
    filtered samples."""

    filter_fn: Callable[[T_sample], bool]
    filter_fn_config: Optional[Union[Dict[str, Any], Callable[[], Dict[str, Any]]]]
    _sample_index: SampleIndex

    _savable_fields = ("_sample_index",)

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        *,
        filter_fn: Callable[[T_sample], bool],
        filter_fn_config: Optional[Union[Dict[str, Any], Callable[[], Dict[str, Any]]]] = None,
        worker_config: WorkerConfig,
    ):
        """Construct a MapDataset.

        Args:
            dataset: The input dataset to wrap
            filter_fn: The function to apply to each sample. If it returns `True`, the sample is
               accepted.
            filter_fn_config: Configuration for the filter function. If callable, it should return the
                configuration. Defaults to None.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset, worker_config=worker_config)
        self.filter_fn = filter_fn
        self.filter_fn_config = filter_fn_config

        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._sample_index = SampleIndex(self.worker_config, src=self)

    def len_worker(self, worker_idx: int | None = None) -> int:
        return self.dataset.len_worker(worker_idx)

    def __iter__(self) -> Iterator[T_sample]:
        for sample in self.dataset:
            with self._sample_index.ctx():
                filter_res = self.filter_fn(sample)
            if filter_res:
                yield sample

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "filter_fn": self._function_config(self.filter_fn),
            **(
                {
                    "filter_fn_config": (
                        self.filter_fn_config()
                        if callable(self.filter_fn_config)
                        else self.filter_fn_config
                    )
                }
                if self.filter_fn_config
                else {}
            ),
        }

    def __str__(self):
        return f"FilterDataset(filter_fn={self.filter_fn}, dataset={self.dataset})"
