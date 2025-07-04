# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Iterator, TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class LimitDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """Limits the length of the dataset."""

    length: int

    current_offset: int
    _savable_fields = ("current_offset",)

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        length: int,
        *,
        reset_after_epoch: bool = False,
        worker_config: WorkerConfig,
    ):
        """
        Limits the length of the dataset.

        Args:
            dataset: The dataset to limit
            length: The length to limit to
            reset_after_epoch: If true, reset the underlying dataset after one epoch.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset, worker_config=worker_config)
        self.length = length
        self.reset_after_epoch = reset_after_epoch
        self.reset_state_own()

    def reset_state_own(self) -> None:
        self.current_offset = 0

    def len_worker(self, worker_idx: int | None = None) -> int:
        if worker_idx is None:
            self.worker_config.assert_worker()
            worker_idx = self.worker_config.rank_worker_id()
        if self.worker_config.num_workers <= 1:
            return self.length
        else:
            local_limit = self.length // self.worker_config.num_workers
            if worker_idx < self.length % self.worker_config.num_workers:
                local_limit += 1
            return local_limit

    def len_rank(self) -> int:
        return min(self.length, self.dataset.len_rank())

    def __iter__(self) -> Iterator[T_sample]:
        worker_id = self.worker_config.rank_worker_id()

        # Compute the local limit for this worker, i.e. all worker's limits sum up to the total
        if self.worker_config.num_workers <= 1:
            local_limit = self.length
        else:
            local_limit = self.length // self.worker_config.num_workers
            if worker_id < self.length % self.worker_config.num_workers:
                local_limit += 1

        if self.worker_config.should_log(level=2):
            self.worker_config.worker_log(
                {
                    "t": "LimitDataset.start",
                    "r": self.worker_config.rank,
                    "w": worker_id,
                    "offset": self.current_offset,
                    "local_limit": local_limit,
                    "limit": self.length,
                }
            )

        offset_range = list(range(self.current_offset, local_limit))
        # Only iterate self.dataset if there are samples to iterate
        if len(offset_range) > 0:
            for sample, offset in zip(
                self.dataset,
                offset_range,
            ):
                self.current_offset = offset + 1
                yield sample

        if self.worker_config.should_log(level=2):
            self.worker_config.worker_log(
                {
                    "t": "LimitDataset.done",
                    "r": self.worker_config.rank,
                    "w": worker_id,
                    "offset": self.current_offset,
                    "local_limit": local_limit,
                    "limit": self.length,
                }
            )

        # Reset the inner dataset
        self.dataset.reset_state_deep()
        self.current_offset = 0
        if self.reset_after_epoch:
            self.dataset.reset_state_deep()

    def worker_has_samples(self) -> bool:
        return super().worker_has_samples() and self.length > 0

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "length": self.length,
            "reset_after_epoch": self.reset_after_epoch,
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"LimitDataset(length={self.length}, dataset={self.dataset})"
