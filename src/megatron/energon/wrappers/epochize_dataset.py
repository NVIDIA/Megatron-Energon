# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Iterator, Optional, TypeVar

from megatron.energon.flavors.base_dataset import (
    SavableDataset,
    add_sample_restore_key,
)
from megatron.energon.flavors.trace import TraceIter, trace_iter
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

    @trace_iter(
        name=lambda self: "EpochizeDataset",
        call_args={
            "config": lambda self: self._own_config(),
        },
    )
    def __iter__(self, trace_iter: TraceIter) -> Iterator[T_sample]:
        # Compute the local length for this worker, i.e. all worker's lengths sum up to the total
        if self.worker_config.num_workers <= 1:
            local_length = self.length
        else:
            local_length = self.length // self.worker_config.num_workers
            if self.worker_config.rank_worker_id() < self.length % self.worker_config.num_workers:
                local_length += 1

        while self._offset < local_length:
            try:
                if self._active_iter is None:
                    self._active_iter = iter(self.dataset)

                sample_offset = self._offset
                self._offset += 1
                try:
                    sample = next(self._active_iter)
                except StopIteration:
                    self._active_iter = None
                    break

                yield add_sample_restore_key(
                    sample,
                    sample_offset,
                    src=self,
                )
            except GeneratorExit:
                if self._active_iter is not None and hasattr(self._active_iter, "close"):
                    self._active_iter.close()
                    self._active_iter = None
                raise
        if self._offset >= local_length:
            self._offset = 0

    def __len__(self) -> int:
        return self.length

    def _own_config(self) -> Dict[str, Any]:
        return {
            "length": self.length,
        }

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "length": self.length,
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"EpochizeDataset(length={self.length}, dataset={self.dataset})"
