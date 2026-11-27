# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Generic, Iterator, TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.state import FlexState
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T = TypeVar("T")


class BoundedDataset(BaseWrapperDataset[T, T], Generic[T]):
    """A loader-owned logical pass; epochization retains the source, limits reset it."""

    length: int
    reset_source: bool
    _offset: int
    _source: Iterator[T] | None
    _savable_fields = ("_offset",)
    _worker_local_fields = {"_source": lambda: None}

    def __init__(
        self,
        dataset: SavableDataset[T],
        *,
        length: int,
        reset_source: bool,
        worker_config: WorkerConfig,
    ):
        super().__init__(dataset, worker_config=worker_config)
        self.length = length
        self.reset_source = reset_source
        self._source = None
        self._offset = 0

    def reset_state_own(self) -> None:
        source, self._source = self._source, None
        if hasattr(source, "close"):
            source.close()
        self._offset = 0

    def reset_state_deep(self) -> None:
        self.reset_state_own()
        self.dataset.reset_state_deep()

    def worker_close(self) -> None:
        self.reset_state_own()
        super().worker_close()

    def len_worker(self, worker_idx: int | None = None) -> int:
        return self.worker_config.logical_worker_length(self.length, worker_idx)

    def __iter__(self) -> Iterator[T]:
        while self._offset < self.len_worker():
            if self._source is None:
                self._source = iter(self.dataset)
            try:
                sample = next(self._source)
            except StopIteration:
                self._source = None
                break
            self._offset += 1
            yield sample
        if self.reset_source:
            self.reset_state_deep()
        else:
            self._offset = 0

    @classmethod
    def remove_saved_boundary(cls, state: FlexState) -> FlexState:
        """Strip a loader boundary if the source loader configured one."""
        if state["__class__"] == cls.__name__:
            return cls._unwrap_child_state(state)
        return state

    def state_at_delivery(
        self, child: FlexState, iteration_index: int, worker_idx: int
    ) -> FlexState:
        offset = self.worker_config.logical_worker_length(iteration_index, worker_idx)
        return self._wrap_child_state(child, _offset=offset)

    def config(self) -> dict:
        return {
            "type": type(self).__name__,
            "length": self.length,
            "reset_source": self.reset_source,
            "dataset": self.dataset.config(),
        }
