# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar, Union

from megatron.energon.dataclass_slots import dataclass_slots
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import (
    BaseSingleWrapperDataset,
    BaseSingleWrapperMergedState,
    BaseSingleWrapperState,
)

T_sample = TypeVar("T_sample")


@dataclass_slots
class RepeatState(BaseSingleWrapperState):
    repetition: int
    index: int


@dataclass_slots
class RepeatMergedState(BaseSingleWrapperMergedState):
    repetition: List[int]
    index: List[int]


class RepeatDataset(BaseSingleWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """This dataset repeats the inner dataset indefinitely or a specific number of repeats."""

    repeats: Optional[Union[int, float]]
    _repetition: List[int]

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        *,
        repeats: Optional[Union[int, float]] = None,
        restart: bool = True,
        worker_config: WorkerConfig,
    ):
        """Construct a RepeatDataset.

        Args:
            dataset: The input dataset to repeat.
            repeats: Number of repeats, `None` for indefinitely repeating.
            restart: If true, restart the underlying dataset after iterating once through the
                repeats if repeats is set to an integer, but still stop iterating.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset, worker_config=worker_config)
        self.repeats = repeats
        self.restart = restart
        self._repetition = [0] * max(self.worker_config.num_workers, 1)
        self._index = [0] * max(self.worker_config.num_workers, 1)

    def __len__(self):
        if self.repeats is None:
            return len(self.dataset)
        return len(self.dataset) * self.repeats

    def __iter__(self) -> Iterator[T_sample]:
        worker_idx = self.worker_config.rank_worker_id()
        assert (
            self.repeats is not None or self.dataset.worker_has_samples()
        ), "Cannot repeat empty dataset indefinitely"

        ds_len = len(self.dataset)

        while self.repeats is None or self._repetition[worker_idx] < self.repeats:
            if self.repeats is not None and self._repetition[worker_idx] == math.floor(
                self.repeats
            ):
                # Last iteration, adjust the number of samples
                fraction = self.repeats - math.floor(self.repeats)
                stop_after = math.floor(ds_len * fraction)
                if self._index[worker_idx] >= stop_after:
                    # We restored an index and it is already past the stop_after
                    break
            else:
                stop_after = None

            for sample in self.dataset:
                self._index[worker_idx] += 1
                yield sample
                if stop_after is not None and self._index[worker_idx] >= stop_after:
                    break

            if self.worker_config.should_log(level=2):
                self.worker_config.worker_log(
                    {
                        "t": "RepeatDataset.repeat",
                        "r": self.worker_config.rank,
                        "w": self.worker_config.rank_worker_id(),
                        "offset": self._repetition[worker_idx],
                        "repeats": self.repeats,
                    }
                )
            self._repetition[worker_idx] += 1
            self._index[worker_idx] = 0

        if self.restart:
            self._repetition[worker_idx] = 0
        else:
            # No more repeats
            self._repetition[worker_idx] = math.ceil(self.repeats)

    def save_state(self) -> RepeatState:
        return RepeatState.extend(
            super().save_state(),
            repetition=self._repetition[self.worker_config.rank_worker_id()],
            index=self._index[self.worker_config.rank_worker_id()],
        )

    def merge_states(self, states: List[RepeatState]) -> RepeatMergedState:
        assert all(s is None or isinstance(s, RepeatState) for s in states)
        return RepeatMergedState.extend(
            super().merge_states(states),
            repetition=[0 if state is None else state.repetition for state in states],
            index=[0 if state is None else state.index for state in states],
        )

    def restore_state(self, state: Optional[RepeatMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._repetition = [0] * max(self.worker_config.num_workers, 1)
            self._index = [0] * max(self.worker_config.num_workers, 1)
        else:
            assert isinstance(state, RepeatMergedState)
            self._repetition = state.repetition
            self._index = state.index

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "repeats": self.repeats,
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"RepeatDataset(repeats={self.repeats}, dataset={self.dataset})"
