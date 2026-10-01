# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Iterator, Optional, TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.logical_worker import LogicalWorkerAssignment
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


def _stride_needed(worker_config: WorkerConfig) -> bool:
    return worker_config.physical_worker_count() > worker_config.logical_worker_count()


class StrideDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """Maps physical workers onto logical streams.

    When ``physical_workers > logical_workers``, consumes inner outputs via iteration
    (including discarded ones) so upstream stages still advance correctly. Discarded
    outputs are produced under skip mode so skip-safe wrappers can avoid heavy work.
    """

    _savable_fields = ()

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        *,
        worker_config: WorkerConfig,
        assignment: Optional[LogicalWorkerAssignment] = None,
    ):
        super().__init__(dataset, worker_config=worker_config)
        self._assignment = assignment
        self.reset_state_own()

    def reset_state_own(self) -> None:
        pass

    def len_worker(self, worker_idx: int | None = None) -> int:
        inner_len = self.dataset.len_worker(worker_idx)
        assignment = self._ensure_assignment()
        if inner_len <= assignment.stride_offset:
            return 0
        return (inner_len - assignment.stride_offset + assignment.stride - 1) // assignment.stride

    def _apply_active_logical_worker(self) -> None:
        assignment = self._ensure_assignment()
        WorkerConfig.set_active_logical_global_worker_id(assignment.logical_global_worker_id)

    def _ensure_assignment(self) -> LogicalWorkerAssignment:
        if self._assignment is None:
            self._assignment = self.worker_config.assignment_for_current_physical_worker()
        return self._assignment

    def _skip_inner(
        self,
        src_iter: Iterator[T_sample],
        count: int,
    ) -> None:
        for _ in range(count):
            self.dataset.set_skip_mode(True)
            try:
                next(src_iter)
            except StopIteration:
                return
            finally:
                self.dataset.set_skip_mode(False)

    def __iter__(self) -> Iterator[T_sample]:
        assignment = self._ensure_assignment()

        self._apply_active_logical_worker()
        src_iter = iter(self.dataset)
        try:
            while True:
                self._skip_inner(src_iter, assignment.stride_offset)
                self.dataset.set_skip_mode(False)
                try:
                    yield next(src_iter)
                except StopIteration:
                    break
                self._skip_inner(src_iter, assignment.stride - assignment.stride_offset - 1)
        finally:
            self.dataset.set_skip_mode(False)
            WorkerConfig.set_active_logical_global_worker_id(None)

    def worker_has_samples(self) -> bool:
        return self.dataset.worker_has_samples()

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "assignment": (
                None
                if self._assignment is None
                else {
                    "logical_global_worker_id": self._assignment.logical_global_worker_id,
                    "stride_offset": self._assignment.stride_offset,
                    "stride": self._assignment.stride,
                }
            ),
            "worker_config": self.worker_config.config(),
        }

    def __str__(self) -> str:
        return f"StrideDataset(dataset={self.dataset})"


def maybe_wrap_stride_dataset(
    dataset: SavableDataset[T_sample],
    *,
    worker_config: WorkerConfig,
    assignment: Optional[LogicalWorkerAssignment] = None,
) -> SavableDataset[T_sample]:
    """Wrap with :class:`StrideDataset` when logical and physical worker counts differ."""
    if _stride_needed(worker_config) or assignment is not None:
        return StrideDataset(dataset, worker_config=worker_config, assignment=assignment)
    return dataset
