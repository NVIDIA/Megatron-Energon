# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Iterator, Optional, TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.logical_worker import LogicalWorkerAssignment
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class StrideDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """Maps physical workers onto logical streams.

    When ``physical_workers > logical_workers``, consumes inner outputs via iteration
    (including discarded ones) so upstream stages still advance correctly. Discarded
    outputs are produced under skip mode so skip-safe wrappers can avoid heavy work.
    """

    #: Cached assignment, resolved lazily in the active physical worker.
    _assignment: Optional[LogicalWorkerAssignment]
    #: Whether the caller supplied a fixed assignment.
    _explicit_assignment: bool
    #: Next output position in the full logical stream, including discarded outputs.
    _logical_sample_index: int
    #: Whether the next iterator must replay and apply the restored starting offset.
    _restore_pending: bool
    #: Full logical outputs to replay before applying the current stride.
    _replay_outputs: int
    #: Previous contributor used to compute the first offset after restoring.
    _previous_worker_id: Optional[int]

    _savable_fields = ("_logical_sample_index",)

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        *,
        worker_config: WorkerConfig,
        assignment: Optional[LogicalWorkerAssignment] = None,
    ):
        super().__init__(dataset, worker_config=worker_config)
        self._assignment = assignment
        self._explicit_assignment = assignment is not None
        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._logical_sample_index = 0
        self._restore_pending = False
        self._replay_outputs = 0
        self._previous_worker_id = None

    def prepare_replay(self, outputs: int, *, previous_worker_id: int) -> None:
        """Apply transient replay instructions after ordinary dataset state restoration."""
        self._restore_pending = True
        self._replay_outputs = outputs
        self._previous_worker_id = previous_worker_id

    def len_worker(self, worker_idx: int | None = None) -> int:
        inner_len = self.dataset.len_worker(worker_idx)
        assignment = (
            self._ensure_assignment()
            if worker_idx is None or self._explicit_assignment
            else self.worker_config.logical_assignment_for_physical(
                self.worker_config.physical_global_worker_id(worker_idx)
            )
        )
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
                self._logical_sample_index += 1
            except StopIteration:
                return
            finally:
                self.dataset.set_skip_mode(False)

    def __iter__(self) -> Iterator[T_sample]:
        assignment = self._ensure_assignment()

        self._apply_active_logical_worker()
        physical_worker = self.worker_config.physical_global_worker_id()
        if self._explicit_assignment:
            initial_skip = assignment.stride_offset
            following_skip = assignment.stride - 1
        else:
            initial_skip = self.worker_config.logical_worker_next_offset(
                physical_worker,
                self._previous_worker_id if self._restore_pending else None,
            )
            following_skip = self.worker_config.logical_worker_next_offset(
                physical_worker, physical_worker
            )
        if self._restore_pending or self._logical_sample_index == 0:
            pending_skip = initial_skip
        else:
            # Ordinary physical checkpoints resume just after this member's last output.
            pending_skip = following_skip
        src_iter = iter(self.dataset)
        try:
            # Replay the full logical stream before applying the current worker's stride.
            self._skip_inner(src_iter, self._replay_outputs)
            self._replay_outputs = 0
            self._restore_pending = False
            self._previous_worker_id = None
            while True:
                self._skip_inner(src_iter, pending_skip)
                try:
                    sample = next(src_iter)
                except StopIteration:
                    break
                self._logical_sample_index += 1
                yield sample
                pending_skip = following_skip
            self.reset_state_own()
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
