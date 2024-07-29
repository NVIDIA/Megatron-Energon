# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar, Union

from megatron.energon.flavors.base_dataset import SavableDataset, add_sample_restore_key
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import (
    BaseSingleWrapperDataset,
    BaseSingleWrapperMergedState,
    BaseSingleWrapperState,
    wrap_worker_sample_index,
    wrap_worker_sample_index_ctx,
)

T_sample = TypeVar("T_sample")


@dataclass
class FilterState(BaseSingleWrapperState):
    sample_index: int


@dataclass
class FilterMergedState(BaseSingleWrapperMergedState):
    sample_indexes: List[int]


class FilterDataset(BaseSingleWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """This dataset wrapper applies a custom filter function to each sample and does not yield
    filtered samples."""

    filter_fn: Callable[[T_sample], bool]
    _sample_index: List[int]

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        filter_fn: Callable[[T_sample], bool],
        worker_config: WorkerConfig,
    ):
        """Construct a MapDataset.

        Args:
            dataset: The input dataset to wrap
            filter_fn: The function to apply to each sample. If it returns `True`, the sample is
               accepted.
        """
        super().__init__(dataset)
        self.filter_fn = filter_fn
        self.worker_config = worker_config
        self._sample_index = [0] * max(self.worker_config.num_workers, 1)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_sample]:
        worker_index = self.worker_config.rank_worker_id()
        for sample_idx, sample in wrap_worker_sample_index(
            self.dataset, self._sample_index, worker_index
        ):
            if self.filter_fn(sample):
                yield add_sample_restore_key(sample, sample_idx, src=self)

    def save_state(self) -> FilterState:
        return FilterState.extend(
            super().save_state(),
            sample_index=self._sample_index[self.worker_config.rank_worker_id()],
        )

    def merge_states(self, states: List[FilterState]) -> FilterMergedState:
        assert all(s is None or isinstance(s, FilterState) for s in states)
        return FilterMergedState.extend(
            super().merge_states(states),
            sample_indexes=[0 if state is None else state.sample_index for state in states],
        )

    def restore_state(self, state: Optional[FilterMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._sample_index = [0] * max(self.worker_config.num_workers, 1)
        else:
            assert isinstance(state, FilterMergedState)
            self._sample_index = state.sample_indexes

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        id, sample_idx = index[:2]
        assert id == type(self).__name__
        index = index[2:]
        with wrap_worker_sample_index_ctx(sample_idx):
            return self.dataset.restore_sample(index)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "filter_fn": self._function_config(self.filter_fn),
        }

    def __str__(self):
        return f"FilterDataset(filter_fn={self.filter_fn}, dataset={self.dataset})"
