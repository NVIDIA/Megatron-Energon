# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Generic, Iterator, TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.state import FlexState
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset
from megatron.energon.wrappers.bounded_dataset import BoundedDataset

T = TypeVar("T")


class StrideDataset(BaseWrapperDataset[T, T], Generic[T]):
    """Advance a complete logical pipeline and retain this physical member's outputs."""

    _cursor: int
    _started: bool
    #: A new assignment's first skip after installing a logical checkpoint.
    _restore_skip: int | None
    _savable_fields = ("_cursor", "_started")
    _worker_local_fields = {"_restore_skip": lambda: None}

    def __init__(self, dataset: SavableDataset[T], *, worker_config: WorkerConfig):
        super().__init__(dataset, worker_config=worker_config)
        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._cursor = 0
        self._started = False
        self._restore_skip = None

    def len_worker(self, worker_idx: int | None = None) -> int:
        assignment = self.worker_config.logical_assignment_for_physical(
            self.worker_config.physical_global_worker_id(worker_idx)
        )
        return max(
            0,
            (self.dataset.len_worker(worker_idx) - assignment.stride_offset + assignment.stride - 1)
            // assignment.stride,
        )

    def __iter__(self) -> Iterator[T]:
        assignment = self.worker_config.logical_assignment_for_physical(
            self.worker_config.physical_global_worker_id()
        )
        skip = assignment.stride - 1 if self._started else assignment.stride_offset
        if self._restore_skip is not None:
            skip, self._restore_skip = self._restore_skip, None
        source = iter(self.dataset)
        try:
            while True:
                for _ in range(skip):
                    self.dataset.set_skip_mode(True)
                    try:
                        next(source)
                        self._cursor += 1
                    finally:
                        self.dataset.set_skip_mode(False)
                sample = next(source)
                self._cursor += 1
                self._started = True
                yield sample
                skip = assignment.stride - 1
        except StopIteration:
            self.reset_state_own()
        finally:
            self.dataset.set_skip_mode(False)
            if hasattr(source, "close"):
                source.close()

    @classmethod
    def resize_saved_cursor(cls, state: FlexState, factor: int) -> FlexState:
        return cls._wrap_child_state(
            cls._unwrap_child_state(state),
            _cursor=state["_cursor"] * factor,
            _started=state["_started"],
        )

    def state_for_assignment(
        self,
        state: FlexState,
        iteration_index: int,
        worker_idx: int,
        *,
        boundary: BoundedDataset | None,
    ) -> tuple[FlexState, int]:
        """Adapt the saved boundary and compute the first skip for this worker."""
        config = self.worker_config
        physical = config.physical_global_worker_id(worker_idx)
        assignment = config.logical_assignment_for_physical(physical)
        skip = assignment.stride_offset
        if iteration_index:
            previous = config.logical_worker_last_physical_worker(
                assignment.logical_global_worker_id, iteration_index
            )
            skip = config.logical_worker_next_offset(physical, previous)

        child = BoundedDataset.remove_saved_boundary(self._unwrap_child_state(state))
        if boundary is not None:
            child = boundary.state_at_delivery(child, iteration_index, worker_idx)
        return self._wrap_child_state(
            child, _cursor=state["_cursor"], _started=state["_started"]
        ), skip

    def config(self) -> dict:
        return {"type": type(self).__name__, "dataset": self.dataset.config()}
