# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gc
import multiprocessing
import queue
import threading
import time
import traceback
import weakref
from copy import deepcopy
from dataclasses import replace
from functools import partial
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch
import torch.distributed
import torch.multiprocessing
from torch.utils.data import DataLoader, IterableDataset

from megatron.energon.cache import CachePool
from megatron.energon.deprecation import deprecated
from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import (
    FlexState,
    SavableDataset,
    State,
    add_sample_restore_key,
)
from megatron.energon.rng import SystemRng, SystemRngState
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset
from megatron.energon.wrappers.batch_dataset import BatchDataset
from megatron.energon.wrappers.epochize_dataset import EpochizeDataset
from megatron.energon.wrappers.gc_dataset import GC_DEFAULT_EVERY_N_ITER, GcDataset, gc_init_worker
from megatron.energon.wrappers.limit_dataset import LimitDataset
from megatron.energon.wrappers.log_sample_dataset import LogSampleDataset, default_get_batch_keys
from megatron.energon.wrappers.stride_dataset import StrideDataset
from megatron.energon.wrappers.watchdog_dataset import WatchdogDataset

T = TypeVar("T")


def _close_data_loader(loader: DataLoader[Any]) -> None:
    """Shut down worker processes before closing the main-process dataset tree."""
    epoch_iterator = getattr(loader, "_epoch_iterator", None)
    if epoch_iterator is not None:
        close_iterator = getattr(epoch_iterator, "close", None)
        if close_iterator is not None:
            close_iterator()
        loader._epoch_iterator = None

    _stop_loader_stream(loader)

    dataset = getattr(loader, "dataset", None)
    close_dataset = getattr(dataset, "close", None)
    if close_dataset is not None:
        close_dataset()


def _stop_loader_stream(loader: DataLoader[Any]) -> None:
    stream = getattr(loader, "_stream_iterator", None)
    if stream is None:
        stream = getattr(loader, "_iterator", None)
    if stream is not None:
        shutdown = getattr(stream, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        else:
            fetcher = getattr(stream, "_dataset_fetcher", None)
            iterator = getattr(fetcher, "dataset_iter", None)
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
    loader._stream_iterator = None
    loader._iterator = None


def _init_worker(seed_per_worker: List[int], worker_id: int):
    """Initializes the the worker process.

    Sets the random seeds and prepare EPath for the forked worker process.
    """
    gc_init_worker(worker_id)

    worker_seed = seed_per_worker[worker_id]

    SystemRng.seed(worker_seed)


def _configure_worker_boundary(
    boundary: Optional[Tuple[Union[EpochizeDataset, LimitDataset], int]],
    worker_config: WorkerConfig,
    *,
    iteration_index: Optional[int] = None,
    replay_outputs: int = 0,
) -> None:
    """Supply a plain boundary wrapper with this logical replica's endpoint."""
    if boundary is None:
        return
    dataset, length = boundary
    if iteration_index is None:
        endpoint = worker_config.logical_worker_length(length)
    else:
        remaining = worker_config.logical_worker_length(
            max(iteration_index, length)
        ) - worker_config.logical_worker_length(iteration_index)
        offset = dataset.current_offset if isinstance(dataset, LimitDataset) else dataset._offset
        endpoint = offset + replay_outputs + remaining
    # EpochizeDataset divides length among physical workers. An exact multiple
    # supplies this replica's endpoint without another remainder distribution.
    dataset.length = endpoint * worker_config.physical_workers_per_rank()


class SimpleSavableDatasetWrapper(BaseWrapperDataset[T, Tuple[int, int, T]], Generic[T]):
    """Wrapper for non-multiprocessing savable datasets. Restarts the inner dataset. This class is
    not intended to be used directly."""

    #: The cache pool to use for the dataset.
    cache_pool: CachePool

    _state_restored: bool
    _sample_index: int
    #: Loader-owned boundary and its public per-rank budget; not checkpoint state.
    _boundary: Optional[Tuple[Union[EpochizeDataset, LimitDataset], int]]

    _savable_fields = ("_sample_index",)

    def __init__(
        self,
        dataset: SavableDataset[T],
        worker_config: WorkerConfig,
        cache_pool: CachePool,
        stride_dataset: StrideDataset[T],
        boundary: Optional[Tuple[Union[EpochizeDataset, LimitDataset], int]] = None,
    ):
        """
        Args:
            dataset: The dataset to wrap.
            worker_config: The worker config to use for the dataset.
            cache_pool: The cache pool to use for the dataset.
        """
        super().__init__(dataset, worker_config=worker_config)
        self.cache_pool = cache_pool
        self._stride_dataset = stride_dataset
        self._boundary = boundary

        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._sample_index = 0
        self._state_restored = False
        self._pending_checkpoint = None

    def get_checkpoint(self) -> SavableDatasetCheckpoint:
        if self._pending_checkpoint is not None:
            return self._pending_checkpoint
        return _save_worker_checkpoint(self._stride_dataset)

    def restore_checkpoint(
        self,
        checkpoint: SavableDatasetCheckpoint,
        iteration_index: int,
    ) -> None:
        self.reset_state_own()
        self._sample_index = _restore_worker_checkpoint(
            self.dataset,
            self._stride_dataset,
            checkpoint,
            self.worker_config,
            iteration_index,
            self._boundary,
        )
        self._pending_checkpoint = checkpoint

    def len_worker(self, worker_idx: int | None = None) -> int:
        return self.dataset.len_worker(worker_idx)

    @property
    def __len__(self):
        # Note: This disables hasattr(self, "__len__"), because that attr will
        raise AttributeError("Disabled direct length access to avoid DataLoader warnings.")

    def __iter__(self):
        if self._pending_checkpoint is None:
            _configure_worker_boundary(self._boundary, self.worker_config)
        self._state_restored = True
        worker_id = self.worker_config.rank_worker_id()
        logical_worker_id = self.worker_config.logical_global_worker_id()
        while self._state_restored:
            self._state_restored = False
            self.worker_config.worker_activate(self._sample_index, cache_pool=self.cache_pool)
            worker_active = True
            try:
                for src_data in self.dataset:
                    self.worker_config.worker_deactivate()
                    worker_active = False
                    self._pending_checkpoint = None
                    sample_index = self._sample_index
                    src_data = add_sample_restore_key(
                        src_data, logical_worker_id, sample_index, src=self
                    )
                    self._sample_index += 1
                    yield worker_id, sample_index, src_data
                    if self._state_restored:
                        # Restart iterator after restore
                        break
                    self.worker_config.worker_activate(
                        self._sample_index,
                        cache_pool=self.cache_pool,
                    )
                    worker_active = True
                else:
                    # EOF can finish replay without producing a new output.
                    self._pending_checkpoint = None
            finally:
                if worker_active:
                    self.worker_config.worker_deactivate()

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T:
        id, logical_worker_id, sample_idx = restore_key[:3]
        assert id == type(self).__name__
        restore_key = restore_key[3:]
        self.worker_config.worker_activate(
            sample_idx,
            override_logical_global_rank=logical_worker_id,
            cache_pool=self.cache_pool,
        )
        try:
            return add_sample_restore_key(
                self.dataset.restore_sample(restore_key),
                logical_worker_id,
                sample_idx,
                src=self,
            )
        finally:
            self.worker_config.worker_deactivate()

    def config(self) -> Dict[str, Any]:
        return self.dataset.config()

    def __str__(self):
        return f"SimpleSavableDatasetWrapper(dataset={self.dataset})"


@edataclass
class SavableDatasetState(State):
    """Logical pipeline snapshot, independent of its physical reader assignment."""

    dataset_state: FlexState
    rng: Optional[SystemRngState]


@edataclass
class SavableDatasetCheckpoint(State):
    """Pipeline snapshot plus the number of full logical outputs to replay."""

    state: Optional[SavableDatasetState]
    offset: int


@edataclass
class PhysicalWorkerCheckpoint(SavableDatasetCheckpoint):
    """Physical worker checkpoint retaining its resume phase for fixed-topology restore."""

    #: Global physical worker id defining the stride phase after checkpoint replay.
    previous_worker_id: int


@edataclass
class _CheckpointHistoryEntry:
    """Worker-local history entry used to locate snapshots despite output prefetching."""

    #: Pipeline snapshot and logical replay offset.
    checkpoint: SavableDatasetCheckpoint
    #: Monotonic creation time used to schedule the next snapshot.
    checkpoint_time: float
    #: Next physical output counter at this history boundary.
    sample_index: int


class SavableDatasetWrapper(IterableDataset[Tuple[int, int, T]], Generic[T]):
    """Internal class for wrapping a savable dataset for a worker process. Provides communication
    with the :class:`megatron.energon.SavableDataLoader`. This class is not intended to be used directly.
    See :class:`megatron.energon.SavableDataLoader` for more information."""

    #: The wrapped dataset
    dataset: SavableDataset[T]
    #: The configuration of the worker process
    worker_config: WorkerConfig
    #: The time interval in seconds to wait at minimum between two checkpoints
    checkpoint_every_sec: float
    #: The minimum number of samples to be emitted between two checkpoints. Should be `number of
    # workers * 2`.
    checkpoint_every_min_n_samples: int
    #: The number of checkpoints to keep in memory, before discarding. Should be 2.
    n_checkpoints: int
    #: The cache pool to use for the dataset.
    cache_pool: CachePool
    #: The queue of the worker process to receive commands from the `SavableDataLoader`.
    _cmd_queues: List[torch.multiprocessing.Queue]
    #: The queue of the worker process to send results to the `SavableDataLoader`.
    _result_queues: List[torch.multiprocessing.Queue]

    _sample_index: int = 0
    _worker_offset: int = 0
    _last_checkpoints: List[_CheckpointHistoryEntry]
    _final_checkpoint: Optional[SavableDatasetCheckpoint]

    _workers_restore_from: List[Optional[SavableDatasetCheckpoint]]
    #: Loader-owned boundary and its public per-rank budget; not checkpoint state.
    _boundary: Optional[Tuple[Union[EpochizeDataset, LimitDataset], int]]

    _running: bool = False
    _command_lock: Optional[threading.RLock] = None
    _cmd_thread: Optional[threading.Thread] = None

    def __init__(
        self,
        dataset: SavableDataset[T],
        worker_config: WorkerConfig,
        checkpoint_every_sec: float,
        checkpoint_every_min_n_samples: int,
        n_checkpoints: int = 2,
        *,
        cmd_queues: List[torch.multiprocessing.Queue],
        result_queues: List[torch.multiprocessing.Queue],
        cache_pool: CachePool,
        stride_dataset: StrideDataset[T],
        boundary: Optional[Tuple[Union[EpochizeDataset, LimitDataset], int]] = None,
    ):
        """
        Create the savable dataset wrapper for multiprocessing data loading.

        Args:
            dataset: The dataset to wrap
            worker_config: The worker config as used by all datasets
            checkpoint_every_sec: The time interval in seconds to wait at minimum between two
                checkpoints.
            checkpoint_every_min_n_samples: The minimum number of samples to be emitted between
                two checkpoints. Should be `number of workers * 2`.
            n_checkpoints: Number of checkpoints to keep.
            cmd_queues: The command queues for communicating with the worker processes.
            result_queues: The result queues for communicating with the worker processes.
            cache_pool: The cache pool to use for the dataset.
        """
        num_workers = max(worker_config.num_workers, 1)

        self.dataset = dataset
        self.worker_config = worker_config
        self.checkpoint_every_sec = checkpoint_every_sec
        self.checkpoint_every_min_n_samples = checkpoint_every_min_n_samples
        self.n_checkpoints = n_checkpoints
        self._last_checkpoints = [
            _CheckpointHistoryEntry(
                checkpoint=SavableDatasetCheckpoint(state=None, offset=0),
                checkpoint_time=time.perf_counter(),
                sample_index=0,
            )
        ]
        self._workers_restore_from = [None] * num_workers
        self._restore_iteration_index = 0
        self._final_checkpoint = None
        self._cmd_queues = cmd_queues
        self._result_queues = result_queues
        self.cache_pool = cache_pool
        self._stride_dataset = stride_dataset
        self._boundary = boundary

    @staticmethod
    def _command_thread(self: "SavableDatasetWrapper"):
        """The internal thread, which processes the command and result queues. This thread is
        static, because `self` is actually passed as weakref proxy, to avoid keeping the dataset
        alive via the thread.
        """
        # print(f"{id(self)}:{multiprocessing.current_process().ident} Worker command thread starting")
        assert self._command_lock is not None

        try:
            while self._running:
                try:
                    cmd_args = self._cmd_queues[self._worker_id].get(timeout=1)
                except queue.Empty:
                    continue
                # print(f"recv cmd {cmd_args}")
                with self._command_lock:
                    cmd = cmd_args[0]
                    if cmd is None:
                        break
                    try:
                        fn = getattr(self, cmd)
                        self._result_queues[self._worker_id].put(
                            {self._worker_id: fn(*cmd_args[1:])}
                        )
                        # print(f"result sent")
                    except Exception as e:
                        traceback.print_exc()
                        self._result_queues[self._worker_id].put({self._worker_id: e})
                        # print(f"exc sent")
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            pass
            # print(f"{id(self)}:{multiprocessing.current_process().ident} Worker command thread closing")

    def len_worker(self, worker_idx: int | None = None) -> int:
        return self.dataset.len_worker(worker_idx)

    def len_rank(self):
        return self.dataset.len_rank()

    @property
    def __len__(self):
        # Note: This disables hasattr(self, "__len__"), because that attr will
        raise AttributeError("Disabled direct length access to avoid DataLoader warnings.")

    def close(self) -> None:
        if self._cmd_thread is not None:
            # print(f"{id(self)}:{multiprocessing.current_process().ident} Closing cmd thread")
            self._running = False
            if self._cmd_thread is not threading.current_thread():
                self._cmd_thread.join()
            self._command_lock = None
            self._cmd_thread = None
            # print(f"{id(self)}:{multiprocessing.current_process().ident} Cmd thread closed")
        self.dataset.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Destructors may run during interpreter shutdown.
            pass

    def __iter__(self):
        # First: Set the worker offset globally for the current worker
        WorkerConfig.worker_id_offset = self._worker_offset
        self._worker_id = self.worker_config.rank_worker_id()
        logical_worker_id = self.worker_config.logical_global_worker_id()
        self._final_checkpoint = None
        if self._cmd_thread is None:
            self._running = True
            self._command_lock = threading.RLock()
            weakref_self = weakref.proxy(self)
            self._cmd_thread = threading.Thread(
                target=SavableDatasetWrapper._command_thread,
                name="command_thread",
                args=(weakref_self,),
                daemon=True,
            )
            self._cmd_thread.start()
            # atexit.register(lambda: weakref_self.__del__())
        try:
            assert self._command_lock is not None
            with self._command_lock:
                pending = self._workers_restore_from[self._worker_id]
                if pending is not None:
                    self._sample_index = _restore_worker_checkpoint(
                        self.dataset,
                        self._stride_dataset,
                        pending,
                        self.worker_config,
                        self._restore_iteration_index,
                        self._boundary,
                    )
                    self._last_checkpoints = [
                        _CheckpointHistoryEntry(
                            checkpoint=pending,
                            checkpoint_time=time.perf_counter(),
                            sample_index=self._sample_index,
                        )
                    ]
                    self._workers_restore_from[self._worker_id] = None
                else:
                    _configure_worker_boundary(self._boundary, self.worker_config)
                    # Store the initial state of the worker if we stop before the first sample
                    self._store_checkpoint(force=True)
                # If skipping, also restart the iterator to reach the start of the restored
                # checkpoint
                last_was_skip = True
                self.worker_config.worker_activate(
                    self._sample_index,
                    cache_pool=self.cache_pool,
                )
                worker_active = True
                try:
                    for src_data in self.dataset:
                        self.worker_config.worker_deactivate()
                        worker_active = False
                        sample_index = self._sample_index
                        add_sample_restore_key(src_data, logical_worker_id, sample_index, src=self)
                        self._sample_index += 1
                        # The first new output has an ordinary resumable stride phase.
                        # Do not snapshot the transient boundary immediately after replay.
                        self._store_checkpoint(force=last_was_skip)
                        last_was_skip = False
                        try:
                            self._command_lock.release()
                            # print(f"{id(self)}:{multiprocessing.current_process().ident} Lock released")
                            # Commands may be executed only when data was yielded, not during
                            # iteration fetching.
                            # print(f"Yield next data [{sample_index}:{self._worker_id}] {src_data}")
                            yield self._worker_id, sample_index, src_data
                        finally:
                            # print(f"{id(self)}:{multiprocessing.current_process().ident} Lock acquiring")
                            self._command_lock.acquire()
                            # print(f"{id(self)}:{multiprocessing.current_process().ident} Lock acquired")
                        self.worker_config.worker_activate(
                            self._sample_index,
                            cache_pool=self.cache_pool,
                        )
                        worker_active = True
                finally:
                    if worker_active:
                        self.worker_config.worker_deactivate()

        finally:
            # print(f"{id(self)}:{multiprocessing.current_process().ident} Worker iter closing")
            # Prefetch can observe EOF before the caller. Keep this transition separate
            # from the last yield, which has the same physical output counter.
            self._final_checkpoint = _save_worker_checkpoint(self._stride_dataset)

    def _store_checkpoint(self, force: bool = False) -> None:
        """
        Internally create a checkpoint for the current state. This is required to store states
        from the past, which have already been yielded here, but not yet been retrieved from the
        intermediate queues.

        Args:
            force: If true, ignore time or frequency condition.
        """
        if (
            force
            or (
                self._last_checkpoints[-1].checkpoint_time + self.checkpoint_every_sec
                < time.perf_counter()
                and self._last_checkpoints[-1].sample_index + self.checkpoint_every_min_n_samples
                <= self._sample_index
            )
            or self._sample_index <= 1
        ):
            # print(f"Storing checkpoint at {self._worker_id}:{self._sample_index}")
            self._last_checkpoints.append(
                _CheckpointHistoryEntry(
                    checkpoint=_save_worker_checkpoint(self._stride_dataset),
                    checkpoint_time=time.perf_counter(),
                    sample_index=self._sample_index,
                )
            )
            if len(self._last_checkpoints) > self.n_checkpoints:
                self._last_checkpoints.pop(0)

    def get_checkpoint(
        self, last_sample_indexes: List[int], source_exhausted: bool = False
    ) -> SavableDatasetCheckpoint:
        """
        Get a checkpoint given the last emitted sample indexes for all workers.

        Args:
            last_sample_indexes: The last emitted sample indexes for all workers.
            source_exhausted: The main loader has observed source EOF. Only then may
                the worker's post-EOF state replace its last-yield checkpoint.

        Returns:
            The found checkpoint including the offset to the next sample index
        """
        if source_exhausted:
            assert self._final_checkpoint is not None, "Worker has not reached EOF"
            return self._final_checkpoint
        sample_index = last_sample_indexes[self._worker_id] + 1
        for historical in reversed(self._last_checkpoints):
            if historical.sample_index <= sample_index:
                checkpoint = historical.checkpoint
                outputs = sample_index - historical.sample_index
                if outputs == 0:
                    return checkpoint
                # History is addressed by physical outputs; replay advances the full stream.
                assignment = self.worker_config.assignment_for_current_physical_worker()
                offset = outputs * assignment.stride
                if (
                    checkpoint.state is None
                    or checkpoint.state.dataset_state["_logical_sample_index"] == 0
                ):
                    offset -= assignment.stride - 1 - assignment.stride_offset
                return PhysicalWorkerCheckpoint(
                    state=checkpoint.state,
                    offset=checkpoint.offset + offset,
                    previous_worker_id=self.worker_config.physical_global_worker_id(),
                )
        raise ValueError("No checkpoint found")

    def restore_checkpoint(
        self,
        worker_states: Optional[List[SavableDatasetCheckpoint]],
        worker_offset: int,
        *,
        iteration_index: int = 0,
    ) -> None:
        """
        Restores the merged checkpoint before starting replacement worker processes.

        Args:
            worker_states: The state to restore for each worker
            worker_offset: The offset of the last worker which has emitted a sample. This will be
                set in all worker processes to ensure the right worker starts as first.
            iteration_index: Scheduling count used to assign logical checkpoints to current workers.
        """
        assert torch.utils.data.get_worker_info() is None, "Cannot restore in worker process"
        num_workers = max(self.worker_config.num_workers, 1)
        self._restore_iteration_index = iteration_index

        # A stopped worker's command thread can leave a queue reader lock acquired.
        # Recreate the channels before starting replacement workers. Update the shared
        # lists in place so the loader and dataset wrapper keep using the same channels.
        for queues in (self._cmd_queues, self._result_queues):
            for channel in queues:
                channel.close()
            queues[:] = [multiprocessing.Queue() for _ in range(num_workers)]

        if worker_states is None:
            self._workers_restore_from = [None] * num_workers
            assert worker_offset == 0
            self._worker_offset = 0
        else:
            assert isinstance(worker_states, list)
            assert len(worker_states) == num_workers
            assert all(isinstance(state, SavableDatasetCheckpoint) for state in worker_states)

            self._worker_offset = worker_offset

            self._workers_restore_from = list(worker_states)

    def reset_state_deep(self) -> None:
        """Reset the dataset tree and pending worker restore assignments."""
        self.dataset.reset_state_deep()
        self.restore_checkpoint(None, worker_offset=0)

    def get_initial_checkpoint(
        self,
    ) -> Optional[List[SavableDatasetCheckpoint]]:
        """
        Get the initial checkpoint for all worker processes if they have not started yet.

        Returns:
            The initial checkpoint for all worker processes and the worker offset.
        """
        assert torch.utils.data.get_worker_info() is None, (
            "Cannot get initial checkpoint in worker process"
        )
        if all(state is None for state in self._workers_restore_from):
            return None
        return [
            state if state is not None else SavableDatasetCheckpoint(state=None, offset=0)
            for state in self._workers_restore_from
        ]

    def can_restore_sample(self) -> bool:
        return self.dataset.can_restore_sample()

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T:
        id, logical_worker_id, sample_idx = restore_key[:3]
        assert id == type(self).__name__
        restore_key = restore_key[3:]
        self.worker_config.worker_activate(
            sample_idx, override_logical_global_rank=logical_worker_id
        )
        try:
            return add_sample_restore_key(
                self.dataset.restore_sample(restore_key),
                logical_worker_id,
                sample_idx,
                src=self,
            )
        finally:
            self.worker_config.worker_deactivate()

    def config(self) -> Dict[str, Any]:
        return self.dataset.config()

    def __str__(self):
        return f"SavableDatasetWrapper(dataset={self.dataset})"


@edataclass
class SavableDataLoaderState(State):
    """Worker checkpoints and the loader metadata needed to resume them."""

    #: Physical saves contain PhysicalWorkerCheckpoint entries in local worker order.
    #: Logical saves contain a
    #: dictionary keyed by global logical id, with one group per returned logical state.
    worker_states: List[SavableDatasetCheckpoint] | Dict[int, SavableDatasetCheckpoint]

    #: Total logical worker count across all ranks, including groups absent from this state.
    #: Must match the restored WorkerConfig's effective logical worker count.
    logical_workers: int

    #: Configured micro batch size inferred from the pipeline, or None if unavailable.
    #: Restore uses this to validate batching and scale replay offsets for smaller batches.
    micro_batch_size: Optional[int]

    #: Outputs delivered per rank in the current worker iterator pass.
    #: Restarts after its StopIteration; excludes prefetch and checkpoint replay.
    #: Modulo the current workers per rank, this gives the next physical worker.
    iteration_index: int


def _save_worker_checkpoint(stride: StrideDataset) -> SavableDatasetCheckpoint:
    config = stride.worker_config
    previous_worker = (
        config.physical_global_worker_id()
        if stride._logical_sample_index
        else config.logical_worker_last_physical_worker(config.logical_global_worker_id(), 0)
    )
    return PhysicalWorkerCheckpoint(
        state=SavableDatasetState(dataset_state=stride.save_state(), rng=SystemRng.save_state()),
        offset=0,
        previous_worker_id=previous_worker,
    )


def _restore_worker_checkpoint(
    dataset: SavableDataset,
    stride: StrideDataset,
    checkpoint: SavableDatasetCheckpoint,
    worker_config: WorkerConfig,
    iteration_index: int,
    boundary: Optional[Tuple[Union[EpochizeDataset, LimitDataset], int]],
) -> int:
    """Install a worker snapshot, taking ownership of one copy of its mutable state."""
    dataset.reset_state_deep()
    snapshot = checkpoint.state
    state = None if snapshot is None else deepcopy(snapshot.dataset_state)
    if state is not None:
        stride.restore_state(state)
    last_worker = worker_config.logical_worker_last_physical_worker(
        worker_config.logical_global_worker_id(), iteration_index
    )
    # Physical saves can precede a sibling's last consumed output. Keep the common
    # logical epoch/limit endpoint while resuming this worker's own stride position.
    boundary_gap = 0
    if (
        isinstance(checkpoint, PhysicalWorkerCheckpoint)
        and checkpoint.previous_worker_id != last_worker
    ):
        boundary_gap = (
            worker_config.logical_worker_next_offset(last_worker, checkpoint.previous_worker_id) + 1
        )
    _configure_worker_boundary(
        boundary,
        worker_config,
        iteration_index=iteration_index,
        replay_outputs=checkpoint.offset + boundary_gap,
    )
    stride.prepare_replay(
        checkpoint.offset,
        previous_worker_id=(
            checkpoint.previous_worker_id
            if isinstance(checkpoint, PhysicalWorkerCheckpoint)
            else last_worker
        ),
    )
    if snapshot is not None:
        if snapshot.rng is None:
            SystemRng.seed(torch.initial_seed() & 0xFFFFFFFF)
        else:
            SystemRng.restore_state(snapshot.rng)
    workers = worker_config.physical_workers_per_rank()
    return iteration_index // workers + int(
        worker_config.rank_worker_id() < iteration_index % workers
    )


def _merge_worker_states(states: Sequence[SavableDataLoaderState]) -> SavableDataLoaderState:
    if not states:
        raise ValueError("Missing logical worker group checkpoints")
    first = states[0]
    checkpoints = {}
    for state in states:
        if not isinstance(state, SavableDataLoaderState):
            raise ValueError("Expected a dataloader checkpoint")
        if not isinstance(state.worker_states, dict):
            raise ValueError("Expected logical worker checkpoints")
        if (
            state.logical_workers != first.logical_workers
            or state.micro_batch_size != first.micro_batch_size
        ):
            raise ValueError("Worker checkpoints require unchanged logical_workers and batch size")
        if checkpoints.keys() & state.worker_states.keys():
            raise ValueError("Duplicate logical worker group checkpoint")
        if state.iteration_index != first.iteration_index:
            raise ValueError("Worker checkpoints require synchronized dataloader iteration")
        checkpoints.update(state.worker_states)
    return SavableDataLoaderState(
        worker_states=checkpoints,
        logical_workers=first.logical_workers,
        micro_batch_size=first.micro_batch_size,
        iteration_index=first.iteration_index,
    )


def logical_worker_group_ids(
    worker_config: WorkerConfig, iteration_index: int, *, for_save: bool = False
) -> tuple[int, ...]:
    workers = worker_config.physical_workers_per_rank()
    groups = dict.fromkeys(
        worker_config.logical_assignment_for_physical(
            worker_config.rank * workers + worker
        ).logical_global_worker_id
        for worker in range(workers)
    )
    return tuple(
        group
        for group in groups
        if not for_save
        or worker_config.logical_worker_last_physical_worker(group, iteration_index) // workers
        == worker_config.rank
    )


def logical_checkpoints_for_rank(
    state: SavableDataLoaderState,
    worker_config: WorkerConfig,
) -> List[SavableDatasetCheckpoint]:
    if worker_config.logical_worker_count() != state.logical_workers:
        raise ValueError("Checkpoint requires the same logical worker count")
    if any(not 0 <= group < state.logical_workers for group in state.worker_states):
        raise ValueError("Invalid logical worker group id")
    missing = (
        set(logical_worker_group_ids(worker_config, state.iteration_index))
        - state.worker_states.keys()
    )
    if missing:
        raise ValueError(f"Missing logical worker group checkpoints: {sorted(missing)}")
    if state.iteration_index < 0:
        raise ValueError("Invalid iteration index")
    workers = worker_config.physical_workers_per_rank()
    return [
        state.worker_states[
            worker_config.logical_assignment_for_physical(
                worker_config.rank * workers + worker
            ).logical_global_worker_id
        ]
        for worker in range(workers)
    ]


class _BaseDataLoader(DataLoader[T], Generic[T]):
    """Construct worker pipelines and coordinate their shared stream lifecycle."""

    def _prepare_dataset(
        self,
        dataset: SavableDataset[T],
        *,
        epochize: int,
        limit: Optional[int],
    ) -> SavableDataset[T]:
        self._epochize = max(epochize, 0)
        self._limit = limit if limit is not None and limit > 0 else None
        self._has_boundaries = bool(self._epochize or self._limit)
        self._stream_iterator = None
        self._epoch_iterator = None

        self._boundary = None
        length = self._limit
        if self._epochize:
            length = min(length, self._epochize) if length else self._epochize
        if length:
            fanout = (
                self.worker_config.physical_worker_count()
                // self.worker_config.logical_worker_count()
            )
            if self._limit is not None:
                dataset = LimitDataset(dataset, length * fanout, worker_config=self.worker_config)
            else:
                dataset = EpochizeDataset(
                    dataset, length * fanout, worker_config=self.worker_config
                )
            self._boundary = (dataset, length)
        self._stride_dataset = StrideDataset(dataset, worker_config=self.worker_config)
        dataset = self._stride_dataset
        if self.worker_config.should_log(level=1):
            dataset = LogSampleDataset(dataset, worker_config=self.worker_config)
        return dataset

    def _source_exhausted(self) -> None:
        self._stream_iterator = None
        self._sample_idx = 0

    def _epoch_outputs(self):
        if self._stream_iterator is None:
            self._stream_iterator = DataLoader.__iter__(self)
            self._has_workers = self.num_workers > 0
        while True:
            try:
                output = next(self._stream_iterator)
            except StopIteration:
                self._source_exhausted()
                return
            yield output

    def _len_rank(self) -> int:
        if self._epochize:
            return min(self._epochize, self._limit) if self._limit else self._epochize
        if self._boundary is None:
            return self.dataset.len_rank()
        boundary, length = self._boundary
        # Estimate the bounded logical inputs, then apply each physical stride.
        # The worker's actual endpoint is configured at startup, after its id is known.
        result = 0
        for worker in range(self.worker_config.physical_workers_per_rank()):
            assignment = self.worker_config.logical_assignment_for_physical(
                self.worker_config.physical_global_worker_id(worker)
            )
            logical_length = min(
                self.worker_config.logical_worker_length(length, worker),
                boundary.dataset.len_worker(worker),
            )
            result += max(
                0,
                (logical_length - assignment.stride_offset + assignment.stride - 1)
                // assignment.stride,
            )
        return result

    def _iteration_config(self) -> Dict[str, Any]:
        return {"epochize": self._epochize, "limit": self._limit}


class SavableDataLoader(_BaseDataLoader[T], Generic[T]):
    """DataLoader that saves and restores the consumed position of each logical worker.

    A logical worker defines a dataset partition and processing stream. Physical workers
    assigned to the same logical worker advance that stream with different strides.
    Without explicit ``logical_workers``, each physical worker is its own logical worker.

    How snapshots track consumed outputs
    ------------------------------------

    With no subprocess workers, the loader snapshots the pipeline directly in the main
    process. With subprocess workers, PyTorch prefetches outputs through internal queues,
    so a worker's current state may be ahead of the last output consumed by the caller.

    1. ``SavableDatasetWrapper`` periodically retains historical snapshots of its logical
       pipeline, logical output cursor, and system RNG state. Each history entry also
       records a physical output counter and creation time for worker-local bookkeeping.
    2. The main process tracks the physical output counter of every consumed output.
       When saving, it asks each worker for a snapshot at or before that consumed position.
    3. The worker returns the selected snapshot and the distance to replay, expressed in
       full logical outputs, including outputs discarded by striding. Prefetched outputs
       do not advance the saved consumed position.

    A worker can also observe source EOF before the caller. Its post-EOF snapshot is kept
    separately from the last-yield snapshot, even when their physical counters are equal.
    Saving at the final yield preserves the pending StopIteration; saving after the caller
    observes EOF preserves the state for the next source pass.

    Selecting and storing logical worker checkpoints
    ------------------------------------------------

    :meth:`save_state_rank` saves every local physical worker without communication.
    Each rank can reload its own file through :meth:`restore_state_rank`, including
    when a logical group spans ranks. Physical saves require unchanged rank assignments,
    world size, workers per rank, and logical worker count. :meth:`save_state_global`
    gathers these states into a list ordered by data parallel rank, restored with
    :meth:`restore_state_global`.

    All ranks must advance their dataloaders synchronously before saving; zero consumed
    outputs and incomplete local worker rounds are supported. For each logical group,
    saving selects the consumed checkpoint of its last physical contributor at that
    scheduling position. This contributor can change after every dataloader call.

    Synchronized calls also require matching delivered output counts and boundary phases.
    The loader does not coordinate early EOF across ranks. Repeated sources with a common
    output budget or an externally coordinated stopping policy are needed when finite
    partitions can exhaust at different times.

    :meth:`save_state_logical_rank` returns (logical id, state) pairs for groups whose
    last contributor is on this rank, or an empty tuple if there are no such groups.
    :meth:`get_logical_worker_group_ids` identifies groups to save or load when callers
    manage checkpoint files themselves. :meth:`save_state_logical_global` gathers the selected
    groups to one destination rank. Both methods use ``SavableDataLoaderState`` and store
    one checkpoint per logical worker, without source physical worker assignments.

    Restoring pipeline state and delivery order
    -------------------------------------------

    Restore into a fresh loader before starting iteration. Keep the dataset pipeline and
    logical worker count unchanged. With unchanged batching, the physical rank and worker
    counts may change provided their global total remains a multiple of the logical count.
    Configure that logical count before building the restored dataset; an implicit count
    otherwise defaults to the new physical total. Restore methods also validate batching
    and handle supported reductions of micro batch size by scaling replay offsets.

    1. :meth:`restore_state_logical_rank` routes checkpoints by logical worker id to this rank's
       physical workers. :meth:`restore_state_logical_global` can first broadcast the global state
       from one source rank, then uses the same routing and restore operation.
    2. Each physical worker restores its logical pipeline and RNG state. The loader-owned
       stride wrapper replays the saved number of full logical outputs before applying
       the current physical assignment, so replay produces no caller-visible outputs.
    3. The saved ``iteration_index`` determines the next local worker modulo the current
       number of workers per rank. PyTorch's worker queue order is fixed, so worker
       identities are rotated using ``WorkerConfig.worker_id_offset`` to resume delivery
       at that worker. The current group assignment determines each member's next stride.

    Epoch and limit progress
    ------------------------

    The loader builds EpochizeDataset or LimitDataset inside its stride, around the
    TaskEncoder-built pipeline. Public lengths remain delivered outputs per rank;
    logical worker budgets account for physical fanout and remainder assignments.
    Replicas of a logical worker receive identical boundaries.

    Ordinary recursive dataset save/restore includes the stride cursor and boundary
    offset, retaining a pending StopIteration after the final yield. No loader-level
    boundary field or separately saved logical cursor is needed. Epochization retains
    the inner iterator between segments; a limit resets the source after completion.
    With both settings active, the smaller budget is used and the source resets.

    Lengths come from the restored loader configuration. Replay reaches the saved consumed
    position before enforcing a newly configured boundary. If its budget is already
    exhausted, the restored iterator completes without yielding new outputs. Replay
    never reopens an exhausted iterator or crosses into another worker pass.
    """

    #: The worker config
    worker_config: WorkerConfig
    #: The wrapped dataset. For multiprocessing, this is a :class:`megatron.energon.SavableDatasetWrapper`
    dataset: Union[SavableDatasetWrapper[T], SimpleSavableDatasetWrapper[T]]

    #: The global ID counter
    _next_id: ClassVar[int] = 0
    #: Class instance id
    id: int = 0

    #: The queues used to send commands to the workers
    cmd_queues: List[torch.multiprocessing.Queue]
    #: The queues used to receive results from the workers
    result_queues: List[torch.multiprocessing.Queue]

    #: Instance of the current data iterator. There shall be only one active iterator, such that the
    # dataset is not iterated multiple times in parallel. The state will continue between epochs.
    _epoch_iterator: Optional[Iterator[T]] = None
    #: Whether the dataloader has running workers.
    _has_workers: bool = False
    #: The index of the current worker. -1 if not started yet.
    _worker_sample_counters: List[int]
    #: Global index of the last yielded sample
    _global_sample_idx: int = 0
    #: Current iterator index of the last yielded sample
    _sample_idx: int = 0

    def __init__(
        self,
        dataset: SavableDataset[T],
        *,
        checkpoint_every_sec: float = 60,
        checkpoint_every_min_n_samples: Optional[int] = None,
        n_checkpoints: Optional[int] = None,
        gc_collect_every_n_steps: int = GC_DEFAULT_EVERY_N_ITER,
        gc_freeze_at_start: bool = True,
        prefetch_factor: int = 2,
        cache_pool: Optional[CachePool] = None,
        watchdog_timeout_seconds: Optional[float] = 60,
        watchdog_initial_timeout_seconds: Optional[float] = None,
        fail_on_timeout: bool = False,
        epochize: int = 0,
        limit: Optional[int] = None,
    ):
        """
        Create the dataloader supporting saving and restoring the state.

        Args:
            dataset: The dataset to load.
            epochize: Maximum delivered outputs per rank in a virtual epoch.
                The next iteration continues the worker stream. Zero disables epochization.
                Without a limit, the reported length is this budget; finite sources may end earlier.
            limit: Maximum delivered outputs per rank in one iteration. Always resets the
                source on completion. None or zero disables it.
                Used alone, its reported length is capped by the source estimate. When used with
                epochize, the limit bounds each segment of the virtual epoch.
            worker_config: The worker config to use
            checkpoint_every_sec: This is the time in seconds after which a checkpoint is saved.
                It may take the same duration to restore a checkpoint, but introduces additional
                overhead during reading data from the dataset, so this should be chosen accordingly.
                Only applies if using workers.
            checkpoint_every_min_n_samples: Overwrites the minimum number of samples between
                checkpoints. Defaults to `number of workers * 2`. Only applies if using workers.
            n_checkpoints: The number of checkpoints to keep in memory. Only applies if using
                workers. If None, computes a suitable value.
            gc_collect_every_n_steps: The number of steps after which the garbage collector is
                called. As we're usually handling large (but few) tensors here, and the python
                garbage collection is already full of objects just by importing, this can improve
                the memory footprint quite a lot, and may even be necessary to avoid memory
                overflow.
            gc_freeze_at_start: If true, the garbage collector is frozen at the start of the worker
                processes. This improves the garbage collection performance by a lot.
                In rare cases, this may cause issues and can be disabled. Keep enabled if you
                experience no issues.
            cache_pool: If set, the cache pool to use for the dataset.
            watchdog_timeout_seconds: The timeout in seconds. If None, the watchdog is disabled.
            watchdog_initial_timeout_seconds: The initial timeout in seconds. If None, the timeout is the same as watchdog_timeout_seconds.
            fail_on_timeout: If True, stops the whole process upon timeout, after printing a stack trace.
        """
        self.worker_config = dataset.worker_config
        dataset = self._prepare_dataset(
            dataset,
            epochize=epochize,
            limit=limit,
        )
        self.id = self.next_id()

        dataset = WatchdogDataset(
            dataset,
            worker_config=self.worker_config,
            timeout_seconds=watchdog_timeout_seconds,
            initial_timeout_seconds=watchdog_initial_timeout_seconds,
            fail_on_timeout=fail_on_timeout,
        )

        if gc_collect_every_n_steps > 0:
            dataset = GcDataset(
                dataset,
                worker_config=self.worker_config,
                every_n_iter=gc_collect_every_n_steps,
                freeze=gc_freeze_at_start,
            )

        self.cmd_queues = [multiprocessing.Queue() for _ in range(self.worker_config.num_workers)]
        self.result_queues = [
            multiprocessing.Queue() for _ in range(self.worker_config.num_workers)
        ]

        num_procs = max(self.worker_config.num_workers, 1)

        if n_checkpoints is None:
            n_checkpoints = prefetch_factor * num_procs + 1

        if self.worker_config.num_workers > 0:
            if checkpoint_every_min_n_samples is None:
                checkpoint_every_min_n_samples = self.worker_config.num_workers * 2

            dataset = SavableDatasetWrapper(
                dataset,
                self.worker_config,
                checkpoint_every_sec=checkpoint_every_sec,
                checkpoint_every_min_n_samples=checkpoint_every_min_n_samples,
                n_checkpoints=n_checkpoints,
                cmd_queues=self.cmd_queues,
                result_queues=self.result_queues,
                cache_pool=cache_pool,
                stride_dataset=self._stride_dataset,
                boundary=self._boundary,
            )
        else:
            dataset = SimpleSavableDatasetWrapper(
                dataset,
                self.worker_config,
                cache_pool=cache_pool,
                stride_dataset=self._stride_dataset,
                boundary=self._boundary,
            )

        self._worker_sample_counters = [-1] * num_procs

        kwargs = {}
        if self.worker_config.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = prefetch_factor
            kwargs["multiprocessing_context"] = "fork"

        # Assert that prefetch_factor works well with num_checkpoints.
        # This ensures that the oldest checkpoint is old enough to cover
        # all the buffered samples in the torch dataloader.
        assert prefetch_factor * num_procs + 1 <= n_checkpoints, (
            "When increasing prefetch_factor, also increase n_checkpoints, so that "
            "the number of checkpoints is at least as large as num_workers * prefetch_factor + 1"
        )

        # Compute seeds for each worker, based on current rank
        seed_per_worker = [
            self.worker_config.worker_seed(i) for i in range(self.worker_config.num_workers)
        ]

        super().__init__(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.worker_config.num_workers,
            pin_memory=True,
            worker_init_fn=partial(_init_worker, seed_per_worker),
            **kwargs,
        )

        if self.worker_config.should_log(level=1):
            self.worker_config.worker_log(
                {
                    "t": "SavableLoader.__init__",
                    "r": self.worker_config.rank,
                    "w": None,
                    "id": self.id,
                    "config": dataset.config(),
                }
            )

    def _source_exhausted(self) -> None:
        if self.num_workers > 0 and self.dataset._worker_offset:
            # Restoring rotates physical worker identities within PyTorch's fixed queue
            # order. Start the next pass in ordinary order, retaining each identity's
            # advanced dataset/RNG state instead of reassigning it to another worker.
            checkpoints = self._worker_command("get_checkpoint", self._worker_sample_counters, True)
            _stop_loader_stream(self)
            self.dataset.restore_checkpoint(checkpoints, worker_offset=0)
            self._worker_sample_counters = [-1 for checkpoint in checkpoints]
            self._has_workers = False
        super()._source_exhausted()

    @staticmethod
    def next_id() -> int:
        next_id = SavableDataLoader._next_id
        SavableDataLoader._next_id += 1
        return next_id

    def __len__(self):
        # We override this, because otherwise we'll see warnings
        return self._len_rank()

    def close(self) -> None:
        _close_data_loader(self)

    def __enter__(self) -> "SavableDataLoader[T]":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors may run during interpreter shutdown.
            pass

    def _epoch_iter(self):
        """Iterator for one epoch, i.e. until the inner dataset raises StopIteration."""
        iter_idx = 0
        id = self.next_id()
        if self.worker_config.should_log(level=1):
            self.worker_config.worker_log(
                {
                    "t": "SavableDataLoader.iter",
                    "r": self.worker_config.rank,
                    "w": None,
                    "id": self.id,
                    "iter_id": id,
                }
            )
        try:
            for worker_id, sample_idx, sample in self._epoch_outputs():
                self._worker_sample_counters[worker_id] = sample_idx
                # self._debugf.write(
                #     f"[w={worker_id}, s={sample_idx}] {self._sample_str(sample)}\n"
                # )
                # self._debugf.flush()
                if self.worker_config.should_log(level=1):
                    keys = default_get_batch_keys(sample)
                    self.worker_config.worker_log(
                        {
                            **{
                                "t": "SavableDataLoader.yield",
                                "r": self.worker_config.rank,
                                "w": None,
                                "id": self.id,
                                "iter_id": id,
                                "worker_id": worker_id,
                                "worker_idx": sample_idx,
                                "idx": self._sample_idx,
                                "iter_idx": iter_idx,
                                "global_idx": self._global_sample_idx,
                            },
                            **({} if keys is None else {"keys": keys}),
                        }
                    )
                self._sample_idx += 1
                self._global_sample_idx += 1
                iter_idx += 1
                yield sample
        finally:
            self._epoch_iterator = None
            if self.worker_config.should_log(level=1):
                self.worker_config.worker_log(
                    {
                        "t": "SavableDataLoader.StopIteration",
                        "r": self.worker_config.rank,
                        "w": None,
                        "id": self.id,
                        "iter_id": self.id,
                    }
                )

    def __iter__(self):
        if self.num_workers == 0 and not self._has_boundaries:
            # Preserve finite-dataset reopening without requiring an extra next() at EOF.
            _stop_loader_stream(self)
            return self._epoch_iter()
        if self._epoch_iterator is None:
            self._epoch_iterator = self._epoch_iter()
        return self._epoch_iterator

    def _worker_command(self, *cmd_args) -> List[Any]:
        """Executes a command in all workers and returns the results."""
        # print(f"cmd: {cmd_args}")
        for cmd_queue in self.cmd_queues:
            cmd_queue.put(cmd_args)
        # print(f"waiting for res")
        assert len(self.result_queues) == self.worker_config.num_workers
        res = {k: v for results_queue in self.result_queues for k, v in results_queue.get().items()}
        res = [res[i] for i in range(len(res))]
        # print(f"res: {res}")
        for r in res:
            if isinstance(r, Exception):
                raise r
        return res

    def _get_batch_size(self) -> Optional[int]:
        """Try to infer micro batch size from the dataset"""
        if isinstance(self.dataset, (SavableDatasetWrapper, SimpleSavableDatasetWrapper)):
            dataset = self.dataset.dataset
        else:
            dataset = self.dataset

        if (
            isinstance(dataset, BaseWrapperDataset)
            and (bds := dataset._find_wrapped_dataset(BatchDataset)) is not None
        ):
            assert isinstance(bds, BatchDataset)
            return bds.batch_size
        else:
            return None

    def get_logical_worker_group_ids(self, *, for_save: bool = False) -> Tuple[int, ...]:
        """Return global logical group ids to load on this rank.

        With ``for_save=True``, return only groups whose last physical member is on this
        rank at the current synchronized iteration. Group ids are stable when physical fanout
        changes with fixed logical_workers; checkpoint ownership may change on every call.
        """
        return logical_worker_group_ids(self.worker_config, self._sample_idx, for_save=for_save)

    def save_state_rank(self) -> Optional[SavableDataLoaderState]:
        """Save every local physical worker without communication between ranks.

        Restore this rank's file with unchanged rank assignments, world size, worker
        count, and logical count. Returns None before subprocess workers have started.
        Use save_state_logical_rank for checkpoints supporting another physical topology.
        """
        if self.num_workers == 0:
            checkpoints = [self.dataset.get_checkpoint()]
        elif self._has_workers:
            checkpoints = self._worker_command(
                "get_checkpoint", self._worker_sample_counters, self._stream_iterator is None
            )
        else:
            checkpoints = self.dataset.get_initial_checkpoint()
        if checkpoints is None:
            return None
        return SavableDataLoaderState(
            worker_states=checkpoints,
            logical_workers=self.worker_config.logical_worker_count(),
            micro_batch_size=self._get_batch_size(),
            iteration_index=self._sample_idx,
        )

    def save_state_logical_rank(self) -> Tuple[Tuple[int, SavableDataLoaderState], ...]:
        """Return (global logical id, state) pairs contributed by this rank.

        Each state is independently serializable and includes its loader metadata.
        Selects each group's last contributor after synchronized delivered batches;
        zero batches and partial worker rounds are supported. Uses no collectives.
        Returns an empty tuple when this rank contributes no logical checkpoints.
        """
        groups = self.get_logical_worker_group_ids(for_save=True)
        if not groups:
            return ()
        workers = self.worker_config.physical_workers_per_rank()
        state = self.save_state_rank()
        if state is None:
            state = SavableDataLoaderState(
                worker_states=[
                    SavableDatasetCheckpoint(state=None, offset=0) for _ in range(workers)
                ],
                logical_workers=self.worker_config.logical_worker_count(),
                micro_batch_size=self._get_batch_size(),
                iteration_index=self._sample_idx,
            )
        states = []
        for group in groups:
            checkpoint = state.worker_states[
                self.worker_config.logical_worker_last_physical_worker(group, self._sample_idx)
                - self.worker_config.rank * workers
            ]
            states.append(
                (
                    group,
                    replace(
                        state,
                        worker_states={
                            group: SavableDatasetCheckpoint(
                                state=checkpoint.state, offset=checkpoint.offset
                            )
                        },
                    ),
                )
            )
        return tuple(states)

    def restore_state_rank(self, state: Optional[SavableDataLoaderState]) -> None:
        """Restore this rank's physical checkpoint with an unchanged topology.

        No collectives are used. None leaves a fresh loader at its initial position;
        it does not retrieve another rank's checkpoint. Restore before iteration.
        """
        if state is None:
            self._check_restore_before_iteration()
            return
        if not isinstance(state, SavableDataLoaderState):
            raise ValueError("Expected a physical rank checkpoint")
        if state.logical_workers != self.worker_config.logical_worker_count():
            raise ValueError("Checkpoint requires the same logical worker count")
        if not isinstance(state.worker_states, list):
            raise ValueError("Expected physical worker checkpoints")
        if len(state.worker_states) != self.worker_config.physical_workers_per_rank():
            raise ValueError("Physical restore requires the same number of workers per rank")
        self._restore_rank_checkpoints(state, state.worker_states)

    def restore_state_logical_rank(
        self,
        states: Optional[
            Mapping[int, SavableDataLoaderState]
            | Sequence[Tuple[int, SavableDataLoaderState]]
            | SavableDataLoaderState
            | Sequence[SavableDataLoaderState]
        ],
    ) -> None:
        """Restore the logical groups needed by this rank, without collectives.

        Supply a mapping of logical ids to states, or the pairs returned by a logical
        save. Extra groups are allowed; missing or duplicate groups raise ValueError.
        Restore before iteration, with the same logical worker count and pipeline.
        The physical topology may change. None leaves a fresh loader unchanged.
        """
        self._check_restore_before_iteration()
        if states is None:
            return
        if isinstance(states, Mapping):
            states = list(states.items())
        if isinstance(states, SavableDataLoaderState):
            state = states
        else:
            checkpoints = []
            for entry in states:
                if isinstance(entry, tuple):
                    logical_id, entry = entry
                    if set(entry.worker_states) != {logical_id}:
                        raise ValueError("Logical id must match the checkpoint's worker id")
                checkpoints.append(entry)
            state = _merge_worker_states(checkpoints)
        if not isinstance(state.worker_states, dict):
            raise ValueError("Expected logical worker checkpoints")
        self._restore_rank_checkpoints(
            state, logical_checkpoints_for_rank(state, self.worker_config)
        )

    def _check_restore_before_iteration(self) -> None:
        if self._has_workers or self._epoch_iterator is not None:
            raise ValueError("Restore before starting the loader")

    def _restore_rank_checkpoints(
        self, state: SavableDataLoaderState, checkpoints: List[SavableDatasetCheckpoint]
    ) -> None:
        self._check_restore_before_iteration()
        if state.iteration_index < 0:
            raise ValueError("Invalid iteration index")
        batch_size = self._get_batch_size()
        if batch_size != state.micro_batch_size:
            if self.num_workers == 0 or batch_size is None or state.micro_batch_size is None:
                raise ValueError("Changing batch size requires subprocess workers and batching")
            if batch_size > state.micro_batch_size or state.micro_batch_size % batch_size:
                raise ValueError("The saved micro batch size must be a multiple of the new size")
            ratio = state.micro_batch_size // batch_size
            # Replay units are full pipeline outputs, so smaller batches require more outputs.
            checkpoints = [
                replace(checkpoint, offset=checkpoint.offset * ratio) for checkpoint in checkpoints
            ]
        workers = self.worker_config.physical_workers_per_rank()
        next_worker_id = state.iteration_index % workers
        if isinstance(state.worker_states, dict):
            # Pending logical replay has a shared starting phase. Retain that phase
            # if a physical save happens before every replacement worker has yielded.
            checkpoints = [
                PhysicalWorkerCheckpoint(
                    state=checkpoint.state,
                    offset=checkpoint.offset,
                    previous_worker_id=self.worker_config.logical_worker_last_physical_worker(
                        self.worker_config.logical_assignment_for_physical(
                            self.worker_config.physical_global_worker_id(worker)
                        ).logical_global_worker_id,
                        state.iteration_index,
                    ),
                )
                for worker, checkpoint in enumerate(checkpoints)
            ]
        if self.num_workers == 0:
            self.dataset.restore_checkpoint(checkpoints[0], state.iteration_index)
        else:
            self.dataset.restore_checkpoint(
                checkpoints,
                worker_offset=next_worker_id,
                iteration_index=state.iteration_index,
            )
            self._worker_sample_counters = [
                state.iteration_index // workers + int(worker_id < next_worker_id) - 1
                for worker_id in range(workers)
            ]
        self._sample_idx = state.iteration_index

    @deprecated(
        "`save_state` is deprecated and was renamed to `save_state_global` and will be removed "
        "in a future update. If you actually do not want to gather the states to a rank, use "
        "`save_state_rank` instead."
    )
    def save_state(self, dst_rank: int) -> Optional[Sequence[Optional[SavableDataLoaderState]]]:
        """Deprecated. Use save_state_global or save_state_rank."""
        return self.save_state_global(dst_rank)

    def save_state_global(
        self, global_dst_rank: int
    ) -> Optional[Sequence[Optional[SavableDataLoaderState]]]:
        """Gather physical rank states into a list ordered by data parallel rank.

        All ranks participate. Only global_dst_rank receives the list. Restore with
        the same topology using restore_state_global, or select a rank's list entry
        for restore_state_rank.
        """
        return self._gather_rank_states(self.save_state_rank(), global_dst_rank)

    def save_state_logical_global(
        self, global_dst_rank: int = 0
    ) -> Optional[Tuple[Tuple[int, SavableDataLoaderState], ...]]:
        """Gather (logical id, state) pairs to a global destination rank.

        All ranks participate after synchronized delivered batches. Only the
        destination receives the pairs; other ranks receive None. Each state is
        independently serializable and supports changing the physical topology.
        """
        ranks = self._gather_rank_states(self.save_state_logical_rank(), global_dst_rank)
        if ranks is None:
            return None
        pairs = tuple(pair for rank_states in ranks for pair in rank_states)
        _merge_worker_states([state for _, state in pairs])
        return pairs

    def _gather_rank_states(self, local: Any, global_dst_rank: int) -> Optional[List[Any]]:
        if self.worker_config.world_size == 1:
            return [local]
        if self.worker_config.data_parallel_group is not None:
            torch.distributed.get_group_rank(
                self.worker_config.data_parallel_group, global_dst_rank
            )
        output = (
            [None] * self.worker_config.world_size
            if self.worker_config.global_rank() == global_dst_rank
            else None
        )
        torch.distributed.gather_object(
            local, output, global_dst_rank, group=self.worker_config.data_parallel_group
        )
        return output

    @deprecated(
        "`restore_state` was renamed to `restore_state_global` and will be removed in a future update."
    )
    def restore_state(self, state: Optional[Sequence[Optional[SavableDataLoaderState]]]) -> None:
        """Deprecated. Use restore_state_global or restore_state_rank."""
        return self.restore_state_global(state)

    def restore_state_global(
        self,
        state: Optional[Sequence[Optional[SavableDataLoaderState]]],
        *,
        src_rank: Optional[int] = None,
    ) -> None:
        """Restore this rank's physical state from a global list with unchanged topology.

        All ranks must call when src_rank is supplied. It identifies the source rank
        within the data parallel group. Only that rank needs to load the list;
        its entries are scattered to their ranks. Otherwise load the list on every rank.
        """
        self._check_restore_before_iteration()
        if src_rank is None or self.worker_config.world_size == 1:
            if not isinstance(state, list) or len(state) != self.worker_config.world_size:
                raise ValueError("Expected a list of states of size world_size")
            rank_state = state[self.worker_config.rank]
        else:
            global_src = (
                src_rank
                if self.worker_config.data_parallel_group is None
                else torch.distributed.get_global_rank(
                    self.worker_config.data_parallel_group, src_rank
                )
            )
            if self.worker_config.rank == src_rank:
                if not isinstance(state, list) or len(state) != self.worker_config.world_size:
                    raise ValueError("Expected a list of states of size world_size")
            elif state is not None:
                raise ValueError("Only src_rank should provide the global state")
            local = [None]
            torch.distributed.scatter_object_list(
                local, state, src=global_src, group=self.worker_config.data_parallel_group
            )
            rank_state = local[0]
        self.restore_state_rank(rank_state)

    def restore_state_logical_global(
        self,
        states: Optional[
            Mapping[int, SavableDataLoaderState] | Sequence[Tuple[int, SavableDataLoaderState]]
        ],
        *,
        src_rank: Optional[int] = None,
    ) -> None:
        """Restore logical states on every rank, optionally broadcasting from src_rank.

        src_rank is within the data parallel group; all ranks participate in the
        broadcast. Without src_rank, load the logical states independently on every
        rank. Physical topology may change with a fixed logical worker count.
        """
        self.restore_state_logical_rank(self._broadcast_state(states, src_rank))

    def _broadcast_state(self, state: Any, src_rank: Optional[int]) -> Any:
        if src_rank is not None and self.worker_config.world_size > 1:
            global_src = (
                src_rank
                if self.worker_config.data_parallel_group is None
                else torch.distributed.get_global_rank(
                    self.worker_config.data_parallel_group, src_rank
                )
            )
            objects = [state]
            torch.distributed.broadcast_object_list(
                objects, src=global_src, group=self.worker_config.data_parallel_group
            )
            state = objects[0]
        return state

    def can_restore_sample(self) -> bool:
        return self.dataset.can_restore_sample()

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T:
        """Restores a sample from a key. This is useful to debug the dataset."""
        return self.dataset.restore_sample(restore_key)

    def config(self):
        """Get the configuration, which defines the dataset. Useful in conjunction with `save_state`
        and `restore_state` to match the configuration as well."""
        return {
            "type": type(self).__qualname__,
            "num_workers": self.num_workers,
            "iteration": self._iteration_config(),
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": None if self.num_workers == 0 else self.prefetch_factor,
            "dataset": self.dataset.config(),
        }


class BasicDataLoader(_BaseDataLoader[T], Generic[T]):
    """DataLoader that supports debugging the dataset without saving capability (e.g. for val/eval)."""

    #: The worker config
    worker_config: WorkerConfig
    #: The wrapped dataset. For multiprocessing, this is a :class:`megatron.energon.SavableDatasetWrapper`
    dataset: Union[SavableDatasetWrapper[T], SavableDataset[T]]

    id: int
    _sample_idx: int = 0

    def __init__(
        self,
        dataset: SavableDataset[T],
        gc_collect_every_n_steps: int = GC_DEFAULT_EVERY_N_ITER,
        gc_freeze_at_start: bool = True,
        prefetch_factor: int = 2,
        cache_pool: Optional[CachePool] = None,
        watchdog_timeout_seconds: Optional[float] = 60,
        watchdog_initial_timeout_seconds: Optional[float] = None,
        fail_on_timeout: bool = False,
        epochize: int = 0,
        limit: Optional[int] = None,
    ):
        """
        Create the dataloader supporting saving and restoring the state.

        Args:
            dataset: The dataset to load.
            epochize: Maximum delivered outputs per rank in a virtual epoch.
                The next iteration continues the worker stream. Zero disables epochization.
                Without a limit, the reported length is this budget; finite sources may end earlier.
            limit: Maximum delivered outputs per rank in one iteration. Always resets the
                source on completion. None or zero disables it.
                Used alone, its reported length is capped by the source estimate. When used with
                epochize, the limit bounds each segment of the virtual epoch.
            gc_collect_every_n_steps: The number of steps after which the garbage collector is
                called. As we're usually handling large (but few) tensors here, and the python
                garbage collection is already full of objects just by importing, this can improve
                the memory footprint quite a lot, and may even be necessary to avoid memory
                overflow.
            gc_freeze_at_start: If true, the garbage collector is frozen at the start of the worker
                processes. This improves the garbage collection performance by a lot.
                In rare cases, this may cause issues and can be disabled. Keep enabled if you
                experience no issues.
            cache_pool: If set, the cache pool to use for the dataset.
            watchdog_timeout_seconds: The timeout in seconds. If None, the watchdog is disabled.
            watchdog_initial_timeout_seconds: The initial timeout in seconds. If None, the timeout is the same as watchdog_timeout_seconds.
            fail_on_timeout: If True, stops the whole process upon timeout, after printing a stack trace.
        """
        self.worker_config = dataset.worker_config
        dataset = self._prepare_dataset(
            dataset,
            epochize=epochize,
            limit=limit,
        )

        self.id = SavableDataLoader.next_id()

        dataset = WatchdogDataset(
            dataset,
            worker_config=self.worker_config,
            timeout_seconds=watchdog_timeout_seconds,
            initial_timeout_seconds=watchdog_initial_timeout_seconds,
            fail_on_timeout=fail_on_timeout,
        )

        if gc_collect_every_n_steps > 0:
            dataset = GcDataset(
                dataset,
                worker_config=self.worker_config,
                every_n_iter=gc_collect_every_n_steps,
                freeze=gc_freeze_at_start,
            )

        dataset = SimpleSavableDatasetWrapper(
            dataset,
            worker_config=self.worker_config,
            cache_pool=cache_pool,
            stride_dataset=self._stride_dataset,
            boundary=self._boundary,
        )

        self._worker_sample_counters = [0] * max(self.worker_config.num_workers, 1)

        kwargs = {}
        if self.worker_config.num_workers > 0:
            # These must not be specified for num_workers =0
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = prefetch_factor
            kwargs["multiprocessing_context"] = "fork"

        seed_per_worker = [
            self.worker_config.worker_seed(i) for i in range(self.worker_config.num_workers)
        ]

        gc.collect()  # This ensures that we don't include any old worker refs in the newly forked worker processes

        super().__init__(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.worker_config.num_workers,
            pin_memory=True,
            worker_init_fn=partial(_init_worker, seed_per_worker),
            **kwargs,
        )
        if self.worker_config.should_log(level=1):
            self.worker_config.worker_log(
                {
                    "t": "BasicDataLoader.__init__",
                    "r": self.worker_config.rank,
                    "w": None,
                    "id": self.id,
                    "config": self.config(),
                }
            )

    def __len__(self):
        # We override this, because otherwise we'll see warnings
        return self._len_rank()

    def close(self) -> None:
        _close_data_loader(self)

    def __enter__(self) -> "BasicDataLoader[T]":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors may run during interpreter shutdown.
            pass

    def __iter__(self):
        def _inner_generator(iterator):
            iter_idx = 0
            id = SavableDataLoader.next_id()

            if self.worker_config.should_log(level=1):
                self.worker_config.worker_log(
                    {
                        "t": "BasicDataLoader.iter",
                        "r": self.worker_config.rank,
                        "w": None,
                        "id": self.id,
                        "iter_id": id,
                    }
                )

            try:
                for worker_id, sample_idx, sample in iterator:
                    # If the next sample will be from the first worker, we can safely resume
                    if self.worker_config.should_log(level=1):
                        keys = default_get_batch_keys(sample)
                        self.worker_config.worker_log(
                            {
                                **{
                                    "t": "BasicDataLoader.yield",
                                    "r": self.worker_config.rank,
                                    "w": None,
                                    "id": self.id,
                                    "iter_id": self.id,
                                    "worker_id": worker_id,
                                    "worker_idx": sample_idx,
                                    "idx": iter_idx,
                                    "iter_idx": iter_idx,
                                    "global_idx": self._sample_idx,
                                },
                                **({} if keys is None else {"keys": keys}),
                            }
                        )
                    self._sample_idx += 1
                    iter_idx += 1
                    yield sample
            finally:
                self._epoch_iterator = None
                if self.worker_config.should_log(level=1):
                    self.worker_config.worker_log(
                        {
                            "t": "BasicDataLoader.StopIteration",
                            "r": self.worker_config.rank,
                            "w": None,
                            "id": self.id,
                            "iter_id": id,
                        }
                    )

        if self.num_workers == 0 and not self._has_boundaries:
            _stop_loader_stream(self)
            return _inner_generator(self._epoch_outputs())
        if self._epoch_iterator is None:
            self._epoch_iterator = _inner_generator(self._epoch_outputs())
        return self._epoch_iterator

    def config(self):
        """Get the configuration, which defines the dataset. Useful in conjunction with `save_state`
        and `restore_state` to match the configuration as well."""
        return {
            "type": type(self).__qualname__,
            "num_workers": self.worker_config.num_workers,
            "iteration": self._iteration_config(),
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": None if self.num_workers == 0 else self.prefetch_factor,
            "dataset": self.dataset.config(),
        }

    def can_restore_sample(self) -> bool:
        return self.dataset.can_restore_sample()

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T:
        """Restores a sample from a key. This is useful to debug the dataset."""
        return self.dataset.restore_sample(restore_key)


def _sample_str(self, sample):
    """Returns a human readable debug string for a single sample, also uniquely identifying it."""
    import dataclasses
    import hashlib

    if isinstance(sample, torch.Tensor):
        return f"Tensor(shape={sample.shape}, dtype={sample.dtype}, sha256={hashlib.sha256(sample.detach().cpu().numpy().tobytes()).hexdigest()!r})"
    elif isinstance(sample, np.ndarray):
        return f"ndarray(shape={sample.shape}, dtype={sample.dtype}, sha256={hashlib.sha256(sample.tobytes()).hexdigest()!r})"
    elif isinstance(sample, (str, bytes)):
        if len(sample) > 100:
            return f"{sample[:100]!r}..."
        return repr(sample)
    elif isinstance(sample, (int, float)):
        return repr(sample)
    elif isinstance(sample, dict):
        innerstr = ", ".join(f"{k}={self._sample_str(v)}, " for k, v in sample.items())
        return f"dict({innerstr})"
    elif isinstance(sample, (list, tuple)):
        innerstr = ", ".join(self._sample_str(v) for v in sample)
        return f"{type(sample)}({innerstr})"
    elif dataclasses.is_dataclass(sample):
        innerstr = ", ".join(
            f"{field.name}={self._sample_str(getattr(sample, field.name))}, "
            for field in dataclasses.fields(sample)
        )
        return f"{type(sample).__name__}({innerstr})"
    else:
        return repr(sample)
