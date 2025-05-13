# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Iterator, List, Tuple, TypeVar, Union

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
    _iterations: int
    _sample_creation: List[int]

    _savable_fields = ("_active_buffer", "_worker_rng", "_iterations", "_sample_creation")

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
        self._iterations = 0
        self._sample_creation = []

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_sample]:
        trace_span = self.worker_config.worker_trace_span()
        with trace_span.span(
            "ShuffleBufferDataset.__iter__", args={"config": self._own_config()}, level=1
        ):
            self._active_buffer.worker_start()
            it = iter(self._active_buffer.append_iter())
            while True:
                if len(self._active_buffer) >= self.size:
                    pop_idx = self._worker_rng.randbelow(len(self._active_buffer))
                    sample_creation = self._sample_creation.pop(pop_idx)
                    trace_span.instant(
                        "ShuffleBufferDataset.__iter__.yield",
                        args={
                            "idx": pop_idx,
                            "sample_creation": sample_creation,
                            "sample_age": self._iterations - sample_creation,
                        },
                        level=2,
                    )
                    yield self._active_buffer.pop(pop_idx)
                else:
                    try:
                        next(it)
                        self._sample_creation.append(self._iterations)
                        trace_span.instant(
                            "ShuffleBufferDataset.__iter__.append",
                            args={
                                "idx": len(self._sample_creation) - 1,
                                "sample_creation": self._iterations,
                            },
                            level=2,
                        )
                        self._iterations += 1
                    except StopIteration:
                        break
            with trace_span.span("ShuffleBufferDataset.__iter__.final_buffer", level=2):
                while len(self._active_buffer) > 0:
                    pop_idx = self._worker_rng.randbelow(len(self._active_buffer))
                    yield self._active_buffer.pop(pop_idx)

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        return self._active_buffer.restore_sample(index)

    def _own_config(self) -> Dict[str, Any]:
        return {
            "size": self.size,
        }

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "size": self.size,
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"ShuffleBufferDataset(size={self.size}, dataset={self.dataset})"
