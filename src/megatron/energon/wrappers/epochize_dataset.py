# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Iterator, Optional, TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class EpochizeDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """
    Uses the base dataset, and creates one epoch, which has length samples. Keeps the underlying
    dataset iterator alive over epochs (i.e. if it is an infinite dataset, it will keep the state).
    Repeats the underlying dataset if the iterator is exhausted.
    """

    length: int
    _active_iter: Optional[Iterator[T_sample]]
    _offset: int

    _savable_fields = ("_offset",)

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        length: int,
        worker_config: WorkerConfig,
    ):
        """
        Create the epochized dataset.

        Args:
            dataset: The source dataset (possibly infinite)
            length: Number of samples to iterate before iteration stops (i.e. one epoch). When
                iteration continues, the original dataset iterator is resumed and does only restart
                if exhausted.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset, worker_config=worker_config)
        self.length = length
        self._active_iter = None

        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._offset = 0

    def __iter__(self) -> Iterator[T_sample]:
        # Compute the local length for this worker, i.e. all worker's lengths sum up to the total

        if self.worker_config.num_workers <= 1:
            local_length = self.length
        else:
            local_length = self.length // self.worker_config.num_workers
            if self.worker_config.rank_worker_id() < self.length % self.worker_config.num_workers:
                local_length += 1

        if self.worker_config.should_log(level=2):
            self.worker_config.worker_log(
                {
                    "t": "EpochizeDataset.epoch_start",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "offset": self._offset,
                    "local_length": local_length,
                    "length": self.length,
                }
            )

        offset_range = list(range(self._offset, local_length))

        # Only iterate if there are samples to iterate
        if len(offset_range) > 0:
            if self._active_iter is None:
                self._active_iter = iter(self.dataset)

            for idx in offset_range:
                self._offset = (idx + 1) % local_length
                try:
                    sample = next(self._active_iter)
                except StopIteration:
                    break
                yield sample

        if self.worker_config.should_log(level=2):
            self.worker_config.worker_log(
                {
                    "t": "EpochizeDataset.epoch_end",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "offset": self._offset,
                    "local_length": local_length,
                    "length": self.length,
                }
            )

    def len_worker(self, worker_idx: int | None = None) -> int:
        if worker_idx is None:
            self.worker_config.assert_worker()
            worker_idx = self.worker_config.rank_worker_id()
        if self.worker_config.num_workers <= 1:
            assert worker_idx == 0
            return self.length
        else:
            local_length = self.length // self.worker_config.num_workers
            if worker_idx < self.length % self.worker_config.num_workers:
                local_length += 1
            return local_length

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "length": self.length,
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"EpochizeDataset(length={self.length}, dataset={self.dataset})"
