# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar, Union

from megatron.energon.flavors.base_dataset import (
    MergedState,
    SavableDataset,
    State,
    add_sample_restore_key,
)
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import (
    BaseWrapperDataset,
    wrap_worker_sample_index,
    wrap_worker_sample_index_ctx,
)

T_sample = TypeVar("T_sample")


@dataclass
class ConcatState(State):
    #: State of the inner datasets
    dataset_states: List[State]
    #: The indexes of the samples of each dataset
    sample_indexes: List[int]


@dataclass
class ConcatMergedState(MergedState):
    #: State of the inner datasets
    dataset_states: List[MergedState]
    #: The indexes of the samples in the datasets [dataset_index][worker_index]
    sample_indexes: List[List[int]]


class ConcatDataset(BaseWrapperDataset[T_sample], Generic[T_sample]):
    """
    This dataset wrapper concatenates multiple iterable datasets together. The datasets must be
    finite, otherwise not all datasets can be sampled. This is only useful for validation / test
    datasets.
    """

    datasets: Tuple[SavableDataset[T_sample], ...]
    # Note that this is inversed: [dataset_index][worker_index]
    _sample_indexes: List[List[int]]

    def __init__(
        self,
        *datasets: SavableDataset[T_sample],
        worker_config: WorkerConfig,
    ):
        """Construct a concatenated dataset."""
        super().__init__()
        self.worker_config = worker_config
        self.datasets = datasets
        assert len(self) >= 0, "Datasets must be finite."
        self._sample_indexes = [
            [0] * max(self.worker_config.num_workers, 1) for _ in range(len(datasets))
        ]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __iter__(self) -> Iterator[T_sample]:
        worker_idx = self.worker_config.rank_worker_id()
        for ds_idx, dataset in enumerate(self.datasets):
            for sample_idx, sample in wrap_worker_sample_index(
                dataset,
                self._sample_indexes[ds_idx],
                worker_idx,
            ):
                yield add_sample_restore_key(
                    sample,
                    ds_idx,
                    sample_idx,
                    src=self,
                )

    def save_state(self) -> ConcatState:
        return ConcatState(
            dataset_states=[dataset.save_state() for dataset in self.datasets],
            sample_indexes=[
                idxs[self.worker_config.rank_worker_id()] for idxs in self._sample_indexes
            ],
        )

    def merge_states(self, states: List[ConcatState]) -> ConcatMergedState:
        assert all(s is None or isinstance(s, ConcatState) for s in states)
        assert all(s is None or len(s.dataset_states) == len(self.datasets) for s in states)
        return ConcatMergedState(
            dataset_states=[
                dataset.merge_states(
                    [None if s is None else s.dataset_states[ds_idx] for s in states]
                )
                for ds_idx, dataset in enumerate(self.datasets)
            ],
            sample_indexes=[
                [
                    0 if states[s_idx] is None else states[s_idx].dataset_states[ds_idx]
                    for s_idx in range(len(states))
                ]
                for ds_idx in range(len(self.datasets))
            ],
        )

    def restore_state(self, state: Optional[ConcatMergedState]) -> None:
        if state is None:
            for dataset in self.datasets:
                dataset.restore_state(None)
            self._sample_indexes = [
                [0] * len(self.datasets) for _ in range(max(self.worker_config.num_workers, 1))
            ]
        else:
            assert isinstance(state, ConcatMergedState)
            assert len(self.datasets) == len(state.dataset_states)
            for dataset, dstate in zip(self.datasets, state.dataset_states):
                dataset.restore_state(dstate)
            self._sample_indexes = state.sample_indexes

    def can_restore_sample(self) -> bool:
        return all(dataset.can_restore_sample() for dataset in self.datasets)

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        id, ds_idx, sample_idx = index[:3]
        assert id == type(self).__name__
        index = index[3:]
        assert isinstance(ds_idx, int)
        with wrap_worker_sample_index_ctx(sample_idx):
            return add_sample_restore_key(
                self.datasets[ds_idx].restore_sample(index),
                ds_idx,
                sample_idx,
                src=self,
            )

    def verify_worker_config(self, worker_config: WorkerConfig) -> None:
        super().verify_worker_config(worker_config)
        for dataset in self.datasets:
            dataset.verify_worker_config(worker_config)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "datasets": [dataset.config() for dataset in self.datasets],
        }

    def __str__(self):
        return f"ConcatDataset(datasets={self.datasets})"
