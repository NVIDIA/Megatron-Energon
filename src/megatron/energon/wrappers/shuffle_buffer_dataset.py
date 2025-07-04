# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Iterator, Tuple, TypeVar, Union

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.rng import WorkerRng
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset
from megatron.energon.wrappers.buffer import SavableSampleBuffer

T_sample = TypeVar("T_sample")


class ShuffleBufferDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """Shuffle buffer for the dataset."""

    size: int
    _worker_rng: WorkerRng
    _active_buffer: SavableSampleBuffer[T_sample]

    _savable_fields = ("_active_buffer", "_worker_rng")

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        size: int,
        *,
        worker_config: WorkerConfig,
    ):
        """Create a shuffle buffer for the dataset."""
        super().__init__(dataset, worker_config=worker_config)
        self.size = size
        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._worker_rng = WorkerRng(self.worker_config)
        self._active_buffer = SavableSampleBuffer(self.dataset, worker_config=self.worker_config)

    def len_worker(self, worker_idx: int | None = None) -> int:
        return self.dataset.len_worker(worker_idx)

    def __iter__(self) -> Iterator[T_sample]:
        self._active_buffer.worker_start()
        it = iter(self._active_buffer.append_iter())
        while True:
            if self._active_buffer.len_worker() >= self.size:
                pop_idx = self._worker_rng.randbelow(self._active_buffer.len_worker())
                yield self._active_buffer.pop(pop_idx)
            else:
                try:
                    next(it)
                except StopIteration:
                    break
        while self._active_buffer.len_worker() > 0:
            pop_idx = self._worker_rng.randbelow(self._active_buffer.len_worker())
            yield self._active_buffer.pop(pop_idx)

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        return self._active_buffer.restore_sample(restore_key)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "size": self.size,
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"ShuffleBufferDataset(size={self.size}, dataset={self.dataset})"
