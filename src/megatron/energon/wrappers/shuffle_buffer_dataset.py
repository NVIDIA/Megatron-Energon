# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar, Union

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.rng import WorkerRng, WorkerRngMergedState, WorkerRngState
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import (
    BaseSingleWrapperDataset,
    BaseSingleWrapperMergedState,
    BaseSingleWrapperState,
)
from megatron.energon.wrappers.buffer import (
    SampleBufferMergedState,
    SampleBufferState,
    SavableSampleBuffer,
)

T_sample = TypeVar("T_sample")


@dataclass
class ShuffleBufferState(BaseSingleWrapperState):
    buffer: SampleBufferState
    rng: WorkerRngState
    sample_index: int


@dataclass
class ShuffleBufferMergedState(BaseSingleWrapperMergedState):
    buffer: SampleBufferMergedState
    rng: WorkerRngMergedState
    sample_indexes: List[int]


class ShuffleBufferDataset(BaseSingleWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """Shuffle buffer for the dataset."""

    size: int
    worker_config: WorkerConfig
    _worker_rng: WorkerRng

    _active_buffer: SavableSampleBuffer[T_sample]
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
        self._active_buffer = SavableSampleBuffer(dataset, worker_config)
        self._sample_index = [0] * max(self.worker_config.num_workers, 1)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_sample]:
        self._active_buffer.worker_start()
        it = iter(self._active_buffer.append_iter(self._sample_index))
        while True:
            if len(self._active_buffer) >= self.size:
                pop_idx = self._worker_rng.randbelow(len(self._active_buffer))
                yield self._active_buffer.pop(pop_idx)
            else:
                try:
                    next(it)
                except StopIteration:
                    break
        while len(self._active_buffer) > 0:
            pop_idx = self._worker_rng.randbelow(len(self._active_buffer))
            yield self._active_buffer.pop(pop_idx)

    def save_state(self) -> ShuffleBufferState:
        return ShuffleBufferState.extend(
            super().save_state(),
            rng=self._worker_rng.save_state(),
            buffer=self._active_buffer.save_state(),
            sample_index=self._sample_index[self.worker_config.rank_worker_id()],
        )

    def merge_states(self, states: List[Optional[ShuffleBufferState]]) -> ShuffleBufferMergedState:
        assert all(s is None or isinstance(s, ShuffleBufferState) for s in states)
        return ShuffleBufferMergedState.extend(
            super().merge_states(states),
            rng=self._worker_rng.merge_states([None if s is None else s.rng for s in states]),
            buffer=self._active_buffer.merge_states(
                [None if s is None else s.buffer for s in states]
            ),
            sample_indexes=[0 if state is None else state.sample_index for state in states],
        )

    def restore_state(self, state: Optional[ShuffleBufferMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._active_buffer.restore_state(None)
            self._worker_rng.restore_state(None)
            self._sample_index = [0] * max(self.worker_config.num_workers, 1)
        else:
            assert isinstance(state, ShuffleBufferMergedState)
            self._active_buffer.restore_state(state.buffer)
            self._worker_rng.restore_state(state.rng)
            self._sample_index = state.sample_indexes

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        return self._active_buffer.restore_sample(index)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "size": self.size,
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"ShuffleBufferDataset(size={self.size}, dataset={self.dataset})"
