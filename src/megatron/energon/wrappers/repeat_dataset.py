# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Any, Dict, Generic, Iterator, Optional, TypeVar, Union

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.flavors.trace import TraceIter, trace_iter
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class RepeatDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """This dataset repeats the inner dataset indefinitely or a specific number of repeats."""

    repeats: Optional[Union[int, float]]
    _repetition: int
    _index: int

    _savable_fields = ("_repetition", "_index")

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

        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._repetition = 0
        self._index = 0

    def __len__(self):
        if self.repeats is None:
            return len(self.dataset)
        return int(len(self.dataset) * self.repeats)

    @trace_iter(
        next_args={
            "idx": lambda self: self._index,
        },
        call_args={
            "repetition": lambda self: self._repetition,
            "inner_len": lambda self: len(self.dataset),
            "config": lambda self: self._own_config(),
        },
    )
    def __iter__(self, trace_iter: TraceIter) -> Iterator[T_sample]:
        assert self.repeats is not None or self.dataset.worker_has_samples(), (
            "Cannot repeat empty dataset indefinitely"
        )

        @trace_iter.wrap_inner(
            call_args=lambda stop_after: {
                "repetition": self._repetition,
                "inner_len": len(self.dataset),
                "stop_after": stop_after,
            }
        )
        def repeat(stop_after: Optional[int]):
            for sample in self.dataset:
                self._index += 1
                yield sample

                if stop_after is not None and self._index >= stop_after:
                    break

        ds_len = len(self.dataset)

        while self.repeats is None or self._repetition < self.repeats:
            if self.repeats is not None and self._repetition == math.floor(self.repeats):
                # Last iteration, adjust the number of samples
                fraction = self.repeats - math.floor(self.repeats)
                stop_after = math.floor(ds_len * fraction)
                if self._index >= stop_after:
                    # We restored an index and it is already past the stop_after
                    break
            else:
                stop_after = None

            yield from repeat(stop_after)
            self._repetition += 1
            self._index = 0

        if self.restart:
            self._repetition = 0
        else:
            # No more repeats
            self._repetition = math.ceil(self.repeats)

    def _own_config(self) -> Dict[str, Any]:
        return {
            "repeats": self.repeats,
        }

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "repeats": self.repeats,
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"RepeatDataset(repeats={self.repeats}, dataset={self.dataset})"
