# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar, Union

from megatron.energon.flavors.base_dataset import Sample, SavableDataset
from megatron.energon.rng import WorkerRng, WorkerRngMergedState, WorkerRngState
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import (
    BaseSingleWrapperDataset,
    BaseSingleWrapperMergedState,
    BaseSingleWrapperState,
    get_sample_restore_key,
    wrap_worker_sample_index,
)

T_sample = TypeVar("T_sample")


@dataclass
class ShuffleBufferState(BaseSingleWrapperState):
    buffer: List[Tuple[Union[str, int], ...]]
    rng: WorkerRngState
    sample_index: int


@dataclass
class ShuffleBufferMergedState(BaseSingleWrapperMergedState):
    buffer: List[List[Tuple[Union[str, int], ...]]]
    rng: WorkerRngMergedState
    sample_indexes: List[int]


class ShuffleBufferDataset(BaseSingleWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """Shuffle buffer for the dataset."""

    size: int
    worker_config: WorkerConfig
    _worker_rng: WorkerRng

    _active_buffer: List[List[T_sample]]
    _active_buffer_restore_keys: List[List[Tuple[Union[str, int], ...]]]
    _restore_pending: bool
    _sample_index: List[int]

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        size: int,
        *,
        worker_config: WorkerConfig,
    ):
        """Create a shuffle buffer for the dataset."""
        super().__init__(dataset)
        self.size = size
        self.worker_config = worker_config
        self._worker_rng = WorkerRng(self.worker_config)
        self._active_buffer = [[] for _ in range(max(self.worker_config.num_workers, 1))]
        self._active_buffer_restore_keys = [
            [] for _ in range(max(self.worker_config.num_workers, 1))
        ]
        self._restore_pending = False
        self._sample_index = [0] * max(self.worker_config.num_workers, 1)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_sample]:
        worker_idx = self.worker_config.rank_worker_id()
        # Cleanup other states
        for i in range(self.worker_config.num_workers):
            if i != worker_idx:
                self._active_buffer[i].clear()
                self._active_buffer_restore_keys[i].clear()
        active_buffer = self._active_buffer[worker_idx]
        active_buffer_restore_keys = self._active_buffer_restore_keys[worker_idx]
        if self._restore_pending:
            assert len(active_buffer) == 0
            self._restore_pending = False
            for restore_key in active_buffer_restore_keys:
                active_buffer.append(self.dataset.restore_sample(restore_key))
        assert len(active_buffer) == len(active_buffer_restore_keys)
        it = iter(wrap_worker_sample_index(self.dataset, self._sample_index, worker_idx))
        while True:
            if len(active_buffer) >= self.size:
                pop_idx = self._worker_rng.randbelow(len(active_buffer))
                active_buffer_restore_keys.pop(pop_idx)
                yield active_buffer.pop(pop_idx)
            else:
                try:
                    sample = next(it)
                except StopIteration:
                    break
                else:
                    active_buffer.append(sample)
                    restore_key = get_sample_restore_key(sample) or ()
                    active_buffer_restore_keys.append(restore_key)
        while len(active_buffer) > 0:
            pop_idx = self._worker_rng.randbelow(len(active_buffer))
            active_buffer_restore_keys.pop(pop_idx)
            yield active_buffer.pop(pop_idx)

    def save_state(self) -> ShuffleBufferState:
        from megatron.energon.flavors.crude import CrudeSample

        assert all(
            isinstance(el, (Sample, CrudeSample))
            for el in self._active_buffer[self.worker_config.rank_worker_id()]
        )
        assert (
            self.dataset.can_restore_sample()
        ), "Cannot restore sample from inner dataset, cannot save buffer state efficiently."
        return ShuffleBufferState.extend(
            super().save_state(),
            rng=self._worker_rng.save_state(),
            buffer=list(self._active_buffer_restore_keys[self.worker_config.rank_worker_id()]),
            sample_index=self._sample_index[self.worker_config.rank_worker_id()],
        )

    def merge_states(self, states: List[ShuffleBufferState]) -> ShuffleBufferMergedState:
        assert all(s is None or isinstance(s, ShuffleBufferState) for s in states)
        return ShuffleBufferMergedState.extend(
            super().merge_states(states),
            rng=self._worker_rng.merge_states([None if s is None else s.rng for s in states]),
            buffer=[[] if s is None else s.buffer for s in states],
            sample_indexes=[0 if state is None else state.sample_index for state in states],
        )

    def restore_state(self, state: Optional[ShuffleBufferMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._active_buffer = [[] for _ in range(max(self.worker_config.num_workers, 1))]
            self._active_buffer_restore_keys = [
                [] for _ in range(max(self.worker_config.num_workers, 1))
            ]
            self._restore_pending = False
            self._worker_rng.restore_state(None)
            self._sample_index = [0] * max(self.worker_config.num_workers, 1)
        else:
            assert isinstance(state, ShuffleBufferMergedState)
            self._active_buffer = [[] for _ in range(max(self.worker_config.num_workers, 1))]
            self._active_buffer_restore_keys = state.buffer
            self._restore_pending = True
            self._worker_rng.restore_state(state.rng)
            self._sample_index = state.sample_indexes

    def can_restore_sample(self) -> bool:
        return False

    def restore_sample(self, restore_key: Tuple[Union[str, int], ...]) -> T_sample:
        raise NotImplementedError("Cannot restore sample from shuffle buffer.")

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "size": self.size,
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"ShuffleBufferDataset(size={self.size}, dataset={self.dataset})"
