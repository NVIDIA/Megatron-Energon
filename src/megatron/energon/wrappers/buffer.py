# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Dict, Generator, Generic, Iterator, List, Optional, Tuple, TypeVar, Union

from megatron.energon.flavors.base_dataset import Sample, SavableDataset, add_sample_restore_key
from megatron.energon.flavors.crude import CrudeSample
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import (
    BaseSingleWrapperDataset,
    BaseSingleWrapperMergedState,
    BaseSingleWrapperState,
    get_sample_restore_key,
    wrap_worker_sample_index,
    wrap_worker_sample_index_ctx,
)

T_sample = TypeVar("T_sample")


@dataclass
class SampleBufferState(BaseSingleWrapperState):
    buffer: List[Tuple[Union[str, int], ...]]


@dataclass
class SampleBufferMergedState(BaseSingleWrapperMergedState):
    buffer: List[List[Tuple[Union[str, int], ...]]]


class SavableSampleBuffer(BaseSingleWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """A buffer of samples, savable."""

    _buffer: List[List[T_sample]]
    _restore_keys: List[List[Tuple[Union[str, int, tuple], ...]]]

    _restore_pending: bool = False

    worker_config: WorkerConfig
    __rank_id: Optional[int] = None

    def __init__(self, dataset: SavableDataset[T_sample], worker_config: WorkerConfig):
        super().__init__(dataset)
        self.worker_config = worker_config
        self._buffer = [[] for _ in range(max(worker_config.num_workers, 1))]
        self._restore_keys = [[] for _ in range(max(worker_config.num_workers, 1))]

    @property
    def _rank_id(self) -> int:
        if self.__rank_id is None:
            self.__rank_id = self.worker_config.rank_worker_id()
        return self.__rank_id

    def worker_start(self) -> None:
        # Cleanup other states
        for i in range(self.worker_config.num_workers):
            if i != self._rank_id:
                self._buffer[i].clear()
                self._restore_keys[i].clear()
        if self._restore_pending:
            assert len(self._buffer[self._rank_id]) == 0
            self._restore_pending = False
            for restore_key in self._restore_keys[self._rank_id]:
                self._buffer[self._rank_id].append(self.restore_sample(restore_key))
        assert len(self._buffer[self._rank_id]) == len(self._restore_keys[self._rank_id])

    def append(self, sample_idx: int, sample: T_sample) -> T_sample:
        add_sample_restore_key(
            sample,
            sample_idx,
            src=self,
        )
        self._buffer[self._rank_id].append(sample)
        self._restore_keys[self._rank_id].append(get_sample_restore_key(sample) or ())
        return sample

    def append_iter(self, sample_index: List[int]) -> Generator[T_sample, None, None]:
        for sample_idx, sample in wrap_worker_sample_index(
            self.dataset,
            sample_index,
            self.worker_config.rank_worker_id(),
        ):
            yield self.append(sample_idx, sample)

    def pop(self, index: int) -> T_sample:
        self._restore_keys[self._rank_id].pop(index)
        return self._buffer[self._rank_id].pop(index)

    def __iter__(self) -> Iterator[T_sample]:
        return iter(self._buffer[self._rank_id])

    def __item__(self, index: int) -> T_sample:
        return self._buffer[self._rank_id][index]

    def __len__(self) -> int:
        return len(self._restore_keys[self._rank_id])

    def save_state(self) -> SampleBufferState:
        assert all(
            isinstance(el, (Sample, CrudeSample)) for el in self._buffer[self._rank_id]
        )
        assert (
            self.dataset.can_restore_sample()
        ), "Cannot restore sample from inner dataset, cannot save buffer state efficiently."
        return SampleBufferState.extend(
            super().save_state(),
            buffer=list(self._restore_keys[self._rank_id]),
        )

    def restore_key(self) -> Tuple[Union[str, int], ...]:
        return tuple(self._restore_keys[self._rank_id])

    def merge_states(self, states: List[Optional[SampleBufferState]]) -> SampleBufferMergedState:
        assert all(s is None or isinstance(s, SampleBufferState) for s in states)
        return SampleBufferMergedState.extend(
            super().merge_states(states),
            buffer=[[] if s is None else s.buffer for s in states],
        )

    def restore_state(self, state: Optional[SampleBufferMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._buffer = [[] for _ in range(max(self.worker_config.num_workers, 1))]
            self._restore_keys = [
                [] for _ in range(max(self.worker_config.num_workers, 1))
            ]
            self._restore_pending = False
        else:
            assert isinstance(state, SampleBufferMergedState)
            self._buffer = [[] for _ in range(max(self.worker_config.num_workers, 1))]
            self._restore_keys = state.buffer
            self._restore_pending = True

    def restore_samples(
        self, index: Tuple[Union[str, int, tuple], ...]
    ) -> Tuple[Tuple[Union[str, int, tuple], ...], List[T_sample]]:
        buffer = []
        restore_keys = []
        for sub_index in index:
            sample = self.restore_sample(sub_index)
            restore_keys.append(get_sample_restore_key(sample) or ())
            buffer.append(sample)
        return tuple(restore_keys), buffer

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        id, sample_idx = index[:2]
        assert id == type(self).__name__
        index = index[2:]
        with wrap_worker_sample_index_ctx(sample_idx):
            return add_sample_restore_key(
                self.dataset.restore_sample(index),
                sample_idx,
                src=self,
            )

    def clear(self) -> None:
        self._buffer.clear()
        self._restore_keys.clear()
    
    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"SavableSampleBuffer(size={len(self._buffer)})"
