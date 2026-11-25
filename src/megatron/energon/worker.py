# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import json
import multiprocessing
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional, TypeVar

import torch.distributed
import torch.utils.data

from megatron.energon.cache import CachePool
from megatron.energon.errors import log_exception, reraise_exception
from megatron.energon.logical_worker import LogicalWorkerAssignment
from megatron.energon.source_info import SourceInfo

__all__ = ("WorkerConfig",)

T = TypeVar("T")


class _classproperty:
    getter: Callable[[type], Any]

    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


@dataclass(slots=True, kw_only=True, eq=False)
class WorkerConfig:
    """
    Provides information about the current worker and the global configuration. This gives each
    data parallel rank its proper config. Every `rank` (up to `world_size-1`) must be used.
    If set wrong, the datasets might yield the same data or data might be missing, as data
    is split over the data parallel ranks with this config!
    You may set the same rank, if you need multiple ranks to retrieve the same data.
    """

    #: The data parallel rank/id of the current process.
    rank: int
    #: The total number of data parallel processes.
    world_size: int
    #: The number of workers per rank. May be 0 to disable worker processes.
    num_workers: int
    #: Stable global stream count; physical workers must be a multiple of this count.
    logical_workers: int | None = None

    #: If not using all ranks for data parallel, set this to the corresponding group.
    data_parallel_group: Optional[torch.distributed.ProcessGroup] = None

    #: The id offset of the current worker. e.g. the worker may live as `worker_info.id=0`, but
    # actually yield samples for id=1 (i.e. worker_id_offset=1). Required to support restoring the
    # worker state if last emitted sample was not for worker_id=0. Required by SavableDataLoader to
    # restore the worker state. Is only set to nonzero within a worker process.
    worker_id_offset: ClassVar[int] = 0

    #: The following seed_offset is used used at two points in the code.
    # 1. The seed_offset in the worker_config that is passed to the dataset initialization, is used
    #    to set the seed for the dataset shuffling and shuffled blending (All code that uses WorkerRng).
    # 2. The worker_config passed to the data loader initialization, is used to set the seed for the
    #    torch, numpy and random libraries. This does not affect the dataset shuffling, but only the
    #    user code (e.g. code in TaskEncoder).
    seed_offset: int = 0

    #: The path to the debug file for the current worker. Should contain "{worker_id}" and "{pid}"
    # to separate the workers.
    worker_debug_path: Optional[str] = None
    #: Log level for worker logging.
    worker_log_level: int = 0
    #: The opened file for the current worker. Should not be set from outside.
    _worker_debug_state: threading.local = field(
        default_factory=threading.local, init=False, repr=False
    )
    #: The global error handler to use for the worker during normal iteration.
    global_error_handler: Callable[
        [Exception, Any | list[Any], Optional[list[SourceInfo]]], None
    ] = log_exception
    #: The error handler to use when restoring samples. Defaults to reraising the exception.
    restore_error_handler: Callable[
        [Exception, Any | list[Any], Optional[list[SourceInfo]]], None
    ] = reraise_exception

    #: Worker context is local to the current thread.
    _active: ClassVar[threading.local] = threading.local()

    @_classproperty
    def active_worker_config(cls):
        return getattr(cls._active, "config", None)

    @property
    def safe_num_workers(self) -> int:
        return max(1, self.num_workers)

    @property
    def _cache_pool(self) -> CachePool | None:
        return getattr(self._active, "cache_pool", None)

    def __post_init__(self) -> None:
        physical = self.physical_worker_count()
        logical = self.logical_worker_count()
        if not isinstance(logical, int) or isinstance(logical, bool) or logical <= 0:
            raise ValueError(f"logical_workers ({logical}) must be greater than zero")
        if logical > physical:
            raise ValueError(
                f"logical_workers ({logical}) must be less than or equal to "
                f"physical_workers ({physical})"
            )
        if physical % logical != 0:
            raise ValueError(
                f"physical_workers ({physical}) must be divisible by logical_workers ({logical})"
            )

    def physical_workers_per_rank(self) -> int:
        """PyTorch DataLoader worker processes per rank (0 means main process only)."""
        return max(1, self.num_workers) if self.num_workers > 0 else 1

    def physical_worker_count(self) -> int:
        """Total physical workers across all ranks."""
        if self.num_workers == 0:
            return self.world_size
        return self.world_size * self.num_workers

    def logical_worker_count(self) -> int:
        """Total logical dataset partitions across all ranks."""
        if self.logical_workers is None:
            return self.physical_worker_count()
        return self.logical_workers

    def logical_assignment_for_physical(
        self, physical_global_worker_id: int
    ) -> LogicalWorkerAssignment:
        """Map a physical global worker id to logical partition and striding."""
        physical_count = self.physical_worker_count()
        logical_count = self.logical_worker_count()
        assert 0 <= physical_global_worker_id < physical_count

        fanout = physical_count // logical_count
        logical_id = physical_global_worker_id // fanout
        start = logical_id * fanout
        end = start + fanout
        workers_per_rank = self.physical_workers_per_rank()
        local_worker = physical_global_worker_id % workers_per_rank
        start_rank, start_local = divmod(start, workers_per_rank)
        end_rank, end_local = divmod(end, workers_per_rank)
        # Count group members with smaller local worker ids, then members with
        # the same local id on earlier ranks. Partial first/last ranks are included.
        stride_offset = (
            (end_rank - start_rank) * local_worker
            + min(end_local, local_worker)
            - min(start_local, local_worker)
            + (physical_global_worker_id - start) // workers_per_rank
        )
        return LogicalWorkerAssignment(
            logical_global_worker_id=logical_id,
            stride_offset=stride_offset,
            stride=fanout,
        )

    def logical_worker_last_physical_worker(
        self, logical_worker_id: int, iteration_index: int
    ) -> int:
        """Last contributor to a logical group after synchronized dataloader calls.

        iteration_index is the number of completed next() calls per rank. Only its phase
        modulo physical_workers_per_rank() matters, so the loader may pass next_worker_id.
        At zero calls, return the end-of-round member to select an initial checkpoint.
        Logical membership and physical ids remain unchanged.
        """
        assert iteration_index >= 0
        assert 0 <= logical_worker_id < self.logical_worker_count()
        fanout = self.physical_worker_count() // self.logical_worker_count()
        start = logical_worker_id * fanout
        last = start + fanout - 1
        workers_per_rank = self.physical_workers_per_rank()
        last_local_worker = (iteration_index - 1) % workers_per_rank
        # Last physical member with the requested local id. If that local id is
        # absent, the group's final physical member is the latest eligible one.
        candidate = last - (last - last_local_worker) % workers_per_rank
        return candidate if candidate >= start else last

    def logical_worker_next_offset(
        self, physical_global_worker_id: int, previous_worker_id: Optional[int] = None
    ) -> int:
        """Outputs to skip before this worker's next output in its logical stream.

        With no previous worker, count from the start of a logical round. Otherwise,
        count from just after previous_worker_id's output; both must belong to the same
        logical group. Passing the same worker gives the ordinary stride minus one.
        """
        assignment = self.logical_assignment_for_physical(physical_global_worker_id)
        if previous_worker_id is None:
            return assignment.stride_offset
        previous = self.logical_assignment_for_physical(previous_worker_id)
        if previous.logical_global_worker_id != assignment.logical_global_worker_id:
            raise ValueError("Previous worker must belong to the same logical group")
        return (assignment.stride_offset - previous.stride_offset - 1) % assignment.stride

    def logical_worker_length(self, length: int, worker_idx: Optional[int] = None) -> int:
        """Full logical outputs needed for a per-rank post-stride output budget.

        All physical replicas of a logical worker receive the same budget. Remainders
        follow the synchronized delivery order: local worker id first, then rank.
        """
        physical = self.physical_global_worker_id(worker_idx)
        assignment = self.logical_assignment_for_physical(physical)
        workers = self.physical_workers_per_rank()
        rounds, remainder = divmod(length, workers)
        start = assignment.logical_global_worker_id * assignment.stride
        end = start + assignment.stride
        # Count group members whose local worker id receives an extra output.
        extras = (
            end // workers * remainder
            + min(end % workers, remainder)
            - start // workers * remainder
            - min(start % workers, remainder)
        )
        return rounds * assignment.stride + extras

    def physical_global_worker_id(self, override_local_worker_id: int | None = None) -> int:
        return self.global_worker_id(override_local_worker_id)

    def logical_global_worker_id(self, override_local_worker_id: int | None = None) -> int:
        return self.logical_assignment_for_physical(
            self.physical_global_worker_id(override_local_worker_id)
        ).logical_global_worker_id

    def __getstate__(self) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        # Debug handles belong to an active thread, not to the configuration.
        attrs, slots = object.__getstate__(self)
        slots.pop("_worker_debug_state", None)
        return attrs, slots

    def __setstate__(self, state: tuple[dict[str, Any] | None, dict[str, Any]]) -> None:
        attrs, slots = state
        if attrs:
            self.__dict__.update(attrs)
        for name, value in slots.items():
            setattr(self, name, value)
        self._worker_debug_state = threading.local()

    def worker_activate(
        self,
        sample_index: int,
        override_global_rank: Optional[int] = None,
        cache_pool: "Optional[CachePool]" = None,
    ):
        """Activates the worker config for the current worker and sets it as actively iterating.
        Must be called before next() call on the datasets."""
        assert WorkerConfig.active_worker_config is None
        WorkerConfig._active.sample_index_stack = [sample_index]
        WorkerConfig._active.config = self
        WorkerConfig._active.override_global_rank = override_global_rank
        WorkerConfig._active.cache_pool = cache_pool

    def worker_push_sample_index(self, sample_index: int):
        """Pushes a new sample index to the sample index stack. Should be set by wrapping datasets
        before calling inners."""
        assert WorkerConfig.active_worker_config is not None
        WorkerConfig._active.sample_index_stack.append(sample_index)

    def worker_pop_sample_index(self):
        """Pushes a new sample index to the sample index stack. Should be set by wrapping datasets
        before calling inners."""
        assert WorkerConfig.active_worker_config is not None
        return WorkerConfig._active.sample_index_stack.pop()

    def worker_deactivate(self):
        """Deactivates the worker config for the current worker and deactivates it for iterating.
        Must be called after next() call on the datasets."""
        if WorkerConfig.active_worker_config is not None:
            assert len(WorkerConfig._active.sample_index_stack) == 1, (
                f"Sample index stack not empty: {WorkerConfig._active.sample_index_stack}"
            )
            WorkerConfig._active.sample_index_stack = None
            WorkerConfig._active.config = None
            WorkerConfig._active.override_global_rank = None
            WorkerConfig._active.cache_pool = None

    @property
    def active_worker_invocation_index(self) -> int:
        """Index within the worker, independent of how workers are grouped into ranks."""
        return WorkerConfig._active.sample_index_stack[-1]

    @property
    def active_worker_sample_index(self) -> int:
        """Returns the current sample index for the actively iterating worker."""
        if self.logical_workers is not None:
            return (
                WorkerConfig._active.sample_index_stack[-1] * self.logical_worker_count()
                + self.logical_global_worker_id()
            )
        # Internal sample index is for the local worker. If using multiple workers per rank, this
        # must be multiplied by the number of workers and offset by the local worker index.
        return (
            WorkerConfig._active.sample_index_stack[-1] * max(self.num_workers, 1)
            + self.rank_worker_id()
        )

    @property
    def active_worker_batch_index(self) -> int:
        """Returns the current batch index for the actively iterating worker."""
        # Internal batch index is for the local worker. If using multiple workers per rank, this
        # must be multiplied by the number of workers and offset by the local worker index.
        return (
            WorkerConfig._active.sample_index_stack[0] * max(self.num_workers, 1)
            + self.rank_worker_id()
        )

    def global_rank(self) -> int:
        """Returns the global rank of this worker config but as a global rank, not
        as a rank within the data parallel group."""

        if self.data_parallel_group is None:
            return self.rank

        return torch.distributed.get_global_rank(self.data_parallel_group, self.rank)

    def __eq__(self, other):
        """Do not compare everything to check for equal config"""
        if not isinstance(other, WorkerConfig):
            return NotImplementedError()
        return all(
            [
                self.rank == other.rank,
                self.world_size == other.world_size,
                self.num_workers == other.num_workers,
                self.logical_workers == other.logical_workers,
            ]
        )

    @staticmethod
    def default_worker_config(
        num_workers: int = 4, data_parallel_group: Optional[torch.distributed.ProcessGroup] = None
    ) -> "WorkerConfig":
        """Returns the default worker config using torch distributed if available.
        If torch distributed is not available, a single local rank is assumed."""

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank(data_parallel_group)
            world_size = torch.distributed.get_world_size(data_parallel_group)
        else:
            rank = 0
            world_size = 1
        return WorkerConfig(
            rank=rank,
            world_size=world_size,
            num_workers=num_workers,
            data_parallel_group=data_parallel_group,
        )

    def rank_worker_id(self, override_global_worker_id: int | None = None) -> int:
        """Returns the self worker id within the current rank."""
        if override_global_worker_id is not None:
            return override_global_worker_id - self.rank * self.safe_num_workers
        override = getattr(self._active, "override_global_rank", None)
        if override is not None:
            return override % self.safe_num_workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.worker_id_offset
        assert worker_info.num_workers == self.num_workers
        # Apply the worker_id_offset as a left rotation of the logical worker ids.
        # This ensures that after restoring a checkpoint the first physical
        # worker (id=0) corresponds to the logical worker that should emit the
        # next sample. For example, if `worker_id_offset` is 1, logical worker
        # 1 becomes the first to emit a sample, shifting the ordering forward.
        return (worker_info.id + self.worker_id_offset) % worker_info.num_workers

    def assert_worker(self):
        """Checks if the current process is a worker (if configured so), and that the workers are
        properly configured."""
        if self.num_workers <= 1:
            assert self.rank_worker_id() == 0
        else:
            worker_info = torch.utils.data.get_worker_info()
            assert worker_info is not None, "Cannot iterate out of worker context"
            assert worker_info.num_workers == self.num_workers, (
                f"Actual number of workers for this rank ({worker_info.num_workers}) does not "
                f"match the configured number of workers ({self.num_workers})"
            )

    def global_worker_id(self, override_local_worker_id: Optional[int] = None) -> int:
        """Returns the global worker index by multiplying the rank with the number of workers.
        Alternatively, you can override the local worker id.

        Args:
            override_local_worker_id (int, optional): The local worker id to override. None means
                the current worker, which is the default.
        """
        if getattr(self._active, "override_global_rank", None) is not None:
            assert override_local_worker_id is None
            return self._active.override_global_rank

        if override_local_worker_id is not None:
            return self.rank * self.safe_num_workers + override_local_worker_id
        else:
            self.assert_worker()
            return self.rank * self.safe_num_workers + self.rank_worker_id()

    def worker_seed(self, override_local_worker_id: Optional[int] = None) -> int:
        """Returns the seed for the current worker (or a specified worker).
        Base on the current worker id and the seed offset, compute a seed.
        Alternatively, you can override the local worker id with a fixed one to
        pregenerate seeds for multiple workers.

        Args:
            override_local_worker_id (int, optional): The local worker id to override. None means
                the current worker, which is the default.
        """

        global_worker_id = self.logical_global_worker_id(override_local_worker_id)

        seed_offset = self.seed_offset

        seed_hash = hashlib.sha1(f"{global_worker_id},{seed_offset}".encode("utf-8")).digest()

        return int.from_bytes(seed_hash, byteorder="big", signed=False) & 0xFFFFFFFF

    def config(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "num_workers": self.num_workers,
            "logical_workers": self.logical_workers,
            "data_parallel_group": (
                self.data_parallel_group.size() if self.data_parallel_group else None
            ),
        }

    def should_log(self, level: int) -> bool:
        return level <= self.worker_log_level

    def worker_log(self, data: dict) -> None:
        """Logs the given data to the worker debug file."""
        if self.worker_debug_path is None:
            print(json.dumps(data) + "\n", end="", flush=True)
        else:
            in_worker = torch.utils.data.get_worker_info() is not None
            # Additional "worker" with rank_worker_id=0 is the main process. All workers have +1
            # as their worker_id.
            worker_id = (
                self.rank * (self.num_workers + 1) + self.rank_worker_id() + (1 if in_worker else 0)
            )
            state = self._worker_debug_state
            if getattr(state, "file", None) is None or state.worker_id != worker_id:
                if getattr(state, "file", None) is not None:
                    state.file.close()
                path = Path(
                    self.worker_debug_path.format(
                        worker_id=worker_id, pid=multiprocessing.current_process().ident
                    )
                )
                path.parent.mkdir(exist_ok=True, parents=True)
                state.file = path.open("w")
                state.worker_id = worker_id
            state.file.write(json.dumps(data) + "\n")
            state.file.flush()
