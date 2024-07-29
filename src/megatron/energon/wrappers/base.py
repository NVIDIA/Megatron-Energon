# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar, Union

from megatron.energon.flavors.base_dataset import MergedState, Sample, SavableDataset, State
from megatron.energon.worker import WorkerConfig

T = TypeVar("T")
T_sample = TypeVar("T_sample")
T_sample_out = TypeVar("T_sample_out")
T_sample_in = TypeVar("T_sample_in")


@dataclass
class BaseSingleWrapperState(State):
    """Base class for dataset states."""

    #: The class name of the dataset saving this state for assertion
    dataset_type: str

    #: State of the inner dataset
    dataset_state: State


@dataclass
class BaseSingleWrapperMergedState(MergedState):
    """Base class for dataset states."""

    #: The class name of the dataset saving/merging this state for assertion
    dataset_type: str

    #: State of the inner dataset
    dataset_state: MergedState


class BaseWrapperDataset(SavableDataset[T_sample], Generic[T_sample]):
    """Base class for dataset wrappers. All dataset wrappers should derive from this. A dataset
    wrapper takes one dataset and modifies its samples to make a new dataset. This can be for
    shuffling samples or applying custom functions to the data. Some wrappers only modify the
    length of the dataset or how it's repeated."""


class BaseSingleWrapperDataset(
    BaseWrapperDataset[T_sample_out], Generic[T_sample_in, T_sample_out]
):
    """Base class for dataset wrappers that wrap a single dataset. Provides default implementations
    for saving and restoring the dataset state."""

    dataset: SavableDataset[T_sample_in]

    def __init__(self, dataset: SavableDataset[T_sample_in]):
        super().__init__()
        self.dataset = dataset

    def save_state(self) -> BaseSingleWrapperState:
        return BaseSingleWrapperState(
            dataset_type=type(self).__name__,
            dataset_state=self.dataset.save_state(),
        )

    def merge_states(
        self, states: List[Optional[BaseSingleWrapperState]]
    ) -> BaseSingleWrapperMergedState:
        assert all(s is None or isinstance(s, BaseSingleWrapperState) for s in states)
        assert all(s is None or s.dataset_type == type(self).__name__ for s in states)
        return BaseSingleWrapperMergedState(
            dataset_type=type(self).__name__,
            dataset_state=self.dataset.merge_states(
                [None if s is None else s.dataset_state for s in states]
            ),
        )

    def restore_state(self, state: Optional[BaseSingleWrapperMergedState]) -> None:
        if state is None:
            self.dataset.restore_state(None)
        else:
            assert isinstance(state, BaseSingleWrapperMergedState)
            assert state.dataset_type == type(self).__name__
            self.dataset.restore_state(state.dataset_state)

    def can_restore_sample(self) -> bool:
        return self.dataset.can_restore_sample()

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample_out:
        return self.dataset.restore_sample(index)

    def verify_worker_config(self, worker_config: WorkerConfig) -> None:
        super().verify_worker_config(worker_config)
        self.dataset.verify_worker_config(worker_config)


def wrap_worker_sample_index(
    src: Iterable[T], index_src: List[int], index_idx: int
) -> Iterable[Tuple[int, T]]:
    """
    Wraps the given iterable to push and pop the sample index for the given worker config.
    """
    WorkerConfig.active_worker_config.worker_push_sample_index(index_src[index_idx])
    index_active = True
    try:
        for sample in src:
            WorkerConfig.active_worker_config.worker_pop_sample_index()
            index_active = False
            sample_idx = index_src[index_idx]
            index_src[index_idx] += 1
            yield sample_idx, sample
            WorkerConfig.active_worker_config.worker_push_sample_index(index_src[index_idx])
            index_active = True
    finally:
        if index_active:
            WorkerConfig.active_worker_config.worker_pop_sample_index()


@contextmanager
def wrap_worker_sample_index_ctx(sample_idx: int):
    """
    Wraps the given iterable to push and pop the sample index for the given worker config.
    """
    WorkerConfig.active_worker_config.worker_push_sample_index(sample_idx)
    try:
        yield
    finally:
        WorkerConfig.active_worker_config.worker_pop_sample_index()


def get_sample_restore_key(sample: Any) -> Optional[Union[str, int]]:
    """Gets the restore key from an arbitrary sample."""
    if isinstance(sample, Sample):
        return sample.__restore_key__
    elif isinstance(sample, dict) and "__restore_key__" in sample:
        return sample["__restore_key__"]
    else:
        return None
