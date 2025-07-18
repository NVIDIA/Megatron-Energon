# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Any, Dict, Generic, Iterator, Optional, TypeVar, Union

from megatron.energon.flavors.base_dataset import SavableDataset
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

    def len_worker(self, worker_idx: int | None = None) -> int:
        if self.repeats is None:
            return self.dataset.len_worker(worker_idx)
        return int(self.dataset.len_worker(worker_idx) * self.repeats)

    def __iter__(self) -> Iterator[T_sample]:
        assert self.repeats is not None or self.dataset.worker_has_samples(), (
            "Cannot repeat empty dataset indefinitely"
        )

        # TODO: There is a small difference in the total sum of samples (across ranks) * repeats
        # and the sum(len_worker() for all workers across ranks).
        # This is due to the fact that the number of samples is not exactly divisible by the number of workers.

        # The dataset length is the size for the current rank. Need to divide by the number of workers
        ds_len = self.dataset.len_worker()

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

            for sample in self.dataset:
                self._index += 1
                yield sample
                if stop_after is not None and self._index >= stop_after:
                    break

            if self.worker_config.should_log(level=2):
                self.worker_config.worker_log(
                    {
                        "t": "RepeatDataset.repeat",
                        "r": self.worker_config.rank,
                        "w": self.worker_config.rank_worker_id(),
                        "offset": self._repetition,
                        "repeats": self.repeats,
                    }
                )
            self._repetition += 1
            self._index = 0

        if self.restart:
            self._repetition = 0
        else:
            # No more repeats
            self._repetition = math.ceil(self.repeats)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "repeats": self.repeats,
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"RepeatDataset(repeats={self.repeats}, dataset={self.dataset})"
