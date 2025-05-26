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
        trace_span = self.worker_config.worker_trace_span()
        # Compute the local length for this worker, i.e. all worker's lengths sum up to the total
        if self.worker_config.num_workers <= 1:
            local_length = self.length
        else:
            local_length = self.length // self.worker_config.num_workers
            if self.worker_config.rank_worker_id() < self.length % self.worker_config.num_workers:
                local_length += 1

        with (
            trace_span.span(
                "EpochizeDataset.__iter__",
                args={
                    "offset": self._offset,
                    "local_length": local_length,
                    "config": self._own_config(),
                },
                level=1,
            ),
            self.worker_config.worker_trace_writer().generator(
                "EpochizeDataset.__iter__.next", level=2
            ) as trace_gen,
        ):
            try:
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
                        with trace_gen.yield_():
                            yield sample
                trace_span.instant("EpochizeDataset.__iter__.done", level=1)
            except GeneratorExit:
                if self._active_iter is not None and hasattr(self._active_iter, "close"):
                    self._active_iter.close()
                    self._active_iter = None
                raise

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
        }

    def __str__(self):
        return f"EpochizeDataset(length={self.length}, dataset={self.dataset})"
