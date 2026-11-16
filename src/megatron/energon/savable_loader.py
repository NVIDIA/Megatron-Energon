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
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
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
from megatron.energon.wrappers.gc_dataset import GC_DEFAULT_EVERY_N_ITER, GcDataset, gc_init_worker
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


class SimpleSavableDatasetWrapper(BaseWrapperDataset[T, Tuple[int, int, T]], Generic[T]):
    """Wrapper for non-multiprocessing savable datasets. Restarts the inner dataset. This class is
    not intended to be used directly."""

    #: The cache pool to use for the dataset.
    cache_pool: CachePool

    _state_restored: bool
    _sample_index: int

    _savable_fields = ("_sample_index",)

    def __init__(
        self,
        dataset: SavableDataset[T],
        worker_config: WorkerConfig,
        cache_pool: CachePool,
        stride_dataset: StrideDataset[T],
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

        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._sample_index = 0
        self._state_restored = False
        self._pending_checkpoint = None

    def get_checkpoint(self) -> SavableDatasetCheckpoint:
        if self._pending_checkpoint is not None:
            return self._pending_checkpoint
        return _save_worker_checkpoint(self.dataset)

    def restore_checkpoint(
        self,
        checkpoint: SavableDatasetCheckpoint,
        iteration_index: int,
    ) -> None:
        self.reset_state_own()
        self._sample_index = _restore_worker_checkpoint(
            self.dataset, self._stride_dataset, checkpoint, self.worker_config, iteration_index
        )
        self._pending_checkpoint = checkpoint

    def reset_for_iteration(self) -> None:
        self.reset_state_deep()

    def len_worker(self, worker_idx: int | None = None) -> int:
        return self.dataset.len_worker(worker_idx)

    @property
    def __len__(self):
        # Note: This disables hasattr(self, "__len__"), because that attr will
        raise AttributeError("Disabled direct length access to avoid DataLoader warnings.")

    def __iter__(self):
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
    sample_index: int
    rng: Optional[SystemRngState]


@edataclass
class SavableDatasetCheckpoint(State):
    """Pipeline snapshot plus the number of full logical outputs to replay."""

    state: Optional[SavableDatasetState]
    offset: int


@edataclass
class SavableCheckpoint:
    """Checkpoint data for :class:`megatron.energon.SavableDatasetWrapper`. An instance is created
    regularly to be able to save the state of the dataset wrapper before the currently emitted
    sample.
    """

    #: The state of the wrapper
    checkpoint: SavableDatasetCheckpoint
    #: The time at which the checkpoint was created
    checkpoint_time: float
    #: Index of the next sample to be returned from the dataset after restoring the checkpoint
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
    _last_checkpoints: List[SavableCheckpoint]
    _final_checkpoint: Optional[SavableDatasetCheckpoint]

    _workers_restore_from: List[Optional[SavableDatasetCheckpoint]]

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
            SavableCheckpoint(
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
                    )
                    self._last_checkpoints = [
                        SavableCheckpoint(
                            checkpoint=pending,
                            checkpoint_time=time.perf_counter(),
                            sample_index=self._sample_index,
                        )
                    ]
                    self._workers_restore_from[self._worker_id] = None
                else:
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
            self._final_checkpoint = _save_worker_checkpoint(self.dataset)

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
                SavableCheckpoint(
                    checkpoint=_save_worker_checkpoint(self.dataset),
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
                if checkpoint.state is None or checkpoint.state.sample_index == 0:
                    offset -= assignment.stride - 1 - assignment.stride_offset
                return SavableDatasetCheckpoint(
                    state=checkpoint.state, offset=checkpoint.offset + offset
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
        Restores the merged checkpoint from all worker processes.

        Args:
            worker_states: The state to restore for each worker
            worker_offset: The offset of the last worker which has emitted a sample. This will be
                set in all worker processes to ensure the right worker starts as first.
            iteration_index: Scheduling count used to assign logical checkpoints to current workers.
        """
        assert torch.utils.data.get_worker_info() is None, "Cannot restore in worker process"
        num_workers = max(self.worker_config.num_workers, 1)
        self._restore_iteration_index = iteration_index

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

    def reset_for_iteration(self) -> None:
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
    """Worker checkpoints keyed by global logical id, independent of physical topology."""

    worker_states: Dict[int, SavableDatasetCheckpoint]
    logical_workers: int
    micro_batch_size: Optional[int]
    #: Completed dataloader calls per rank; determines the next physical worker.
    iteration_index: int
    #: Progress within the public epoch/limit, outside worker replay.
    boundary_offsets: List[int]


def _save_worker_checkpoint(dataset: SavableDataset) -> SavableDatasetCheckpoint:
    pipeline_state, sample_index = dataset.export_logical_state(dataset.save_state())
    return SavableDatasetCheckpoint(
        state=SavableDatasetState(
            dataset_state=pipeline_state,
            sample_index=sample_index,
            rng=SystemRng.save_state(),
        ),
        offset=0,
    )


def _restore_worker_checkpoint(
    dataset: SavableDataset,
    stride: StrideDataset,
    checkpoint: SavableDatasetCheckpoint,
    worker_config: WorkerConfig,
    iteration_index: int,
) -> int:
    """Install a worker snapshot, taking ownership of one copy of its mutable state."""
    dataset.reset_state_deep()
    snapshot = checkpoint.state
    state = None if snapshot is None else deepcopy(snapshot.dataset_state)
    stride.restore_logical_checkpoint(
        state,
        0 if snapshot is None else snapshot.sample_index,
        replay_outputs=checkpoint.offset,
        previous_worker_id=worker_config.logical_worker_last_physical_worker(
            worker_config.logical_global_worker_id(), iteration_index
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
        if (
            state.logical_workers != first.logical_workers
            or state.micro_batch_size != first.micro_batch_size
        ):
            raise ValueError("Worker checkpoints require unchanged logical_workers and batch size")
        if checkpoints.keys() & state.worker_states.keys():
            raise ValueError("Duplicate logical worker group checkpoint")
        if state.iteration_index != first.iteration_index:
            raise ValueError("Worker checkpoints require synchronized dataloader iteration")
        if state.boundary_offsets != first.boundary_offsets:
            raise ValueError("Worker checkpoints require synchronized epoch/limit progress")
        checkpoints.update(state.worker_states)
    return SavableDataLoaderState(
        worker_states=checkpoints,
        logical_workers=first.logical_workers,
        micro_batch_size=first.micro_batch_size,
        iteration_index=first.iteration_index,
        boundary_offsets=list(first.boundary_offsets),
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


@dataclass
class _Boundary:
    kind: Literal["epoch", "limit"]
    length: int
    offset: int = 0


class IterationController:
    """Count delivered outputs and report when the loader should reset its source."""

    def __init__(
        self,
        *,
        epochize: int = 0,
        limit: int | None = None,
    ):
        self._boundaries = []
        if limit is not None and limit > 0:
            self._boundaries.append(_Boundary("limit", limit))
        if epochize > 0:
            self._boundaries.append(_Boundary("epoch", epochize))

    def __bool__(self) -> bool:
        return bool(self._boundaries)

    def has_room(self) -> bool:
        return all(boundary.offset < boundary.length for boundary in self._boundaries)

    def advance(self) -> None:
        for boundary in self._boundaries:
            boundary.offset += 1

    def finish(self, *, source_exhausted: bool = False) -> bool:
        """Finish all enclosing boundaries; return whether the source must reset."""
        last = (
            len(self._boundaries) - 1
            if source_exhausted
            else max(
                idx
                for idx, boundary in enumerate(self._boundaries)
                if boundary.offset == boundary.length
            )
        )
        ended = self._boundaries[: last + 1]
        reset_source = any(boundary.kind == "limit" for boundary in ended)
        for boundary in self._boundaries if reset_source else ended:
            boundary.offset = 0
        return reset_source

    def save_state(self) -> list[int]:
        return [boundary.offset for boundary in self._boundaries]

    def restore_state(self, offsets: list[int]) -> None:
        if len(offsets) != len(self._boundaries) or any(
            not 0 <= offset <= boundary.length
            for offset, boundary in zip(offsets, self._boundaries)
        ):
            raise ValueError("Checkpoint requires the same epoch/limit configuration")
        for offset, boundary in zip(offsets, self._boundaries):
            boundary.offset = offset

    def len_rank(self, source_length: int) -> int:
        for boundary in reversed(self._boundaries):
            source_length = (
                boundary.length if boundary.kind == "epoch" else min(source_length, boundary.length)
            )
        return source_length

    def config(self) -> list[dict]:
        return [
            {
                "type": boundary.kind,
                "length": boundary.length,
            }
            for boundary in self._boundaries
        ]


class _BaseDataLoader(DataLoader[T], Generic[T]):
    """Shared worker-stream lifecycle; boundary counting is delegated to its controller."""

    def _prepare_dataset(
        self,
        dataset: SavableDataset[T],
        *,
        epochize: int,
        limit: Optional[int],
    ) -> SavableDataset[T]:
        self._iteration = IterationController(
            epochize=epochize,
            limit=limit,
        )
        self._stream_iterator = None
        self._epoch_iterator = None

        self._stride_dataset = StrideDataset(dataset, worker_config=self.worker_config)
        dataset = self._stride_dataset
        if self.worker_config.should_log(level=1):
            dataset = LogSampleDataset(dataset, worker_config=self.worker_config)
        return dataset

    def _reset_stream(self) -> None:
        """Discard prefetched outputs and restart the worker pipelines."""
        _stop_loader_stream(self)
        self.dataset.reset_for_iteration()
        self._has_workers = False
        self._worker_sample_counters = [-1] * max(1, self.num_workers)
        self._sample_idx = 0

    def _source_exhausted(self) -> None:
        self._stream_iterator = None
        self._sample_idx = 0

    def _epoch_outputs(self):
        while self._iteration.has_room():
            if self._stream_iterator is None:
                self._stream_iterator = DataLoader.__iter__(self)
                self._has_workers = self.num_workers > 0
            try:
                output = next(self._stream_iterator)
            except StopIteration:
                if self._iteration.finish(source_exhausted=True):
                    self._reset_stream()
                else:
                    self._source_exhausted()
                return
            self._iteration.advance()
            yield output
        if self._iteration.finish():
            self._reset_stream()


class SavableDataLoader(_BaseDataLoader[T], Generic[T]):
    """DataLoader with one checkpoint per logical worker.

    Without explicit logical_workers, each physical worker is its own logical worker.
    Workers retain historical snapshots to describe the consumed position despite
    prefetching. Saving selects each logical group's last contributor; restoring replays
    full logical outputs before applying the current physical assignment.

    Rank methods support external checkpoint storage and routing by logical group id.
    Global methods gather and broadcast the same checkpoint format. Physical topology
    may change with fixed logical worker count and processing pipeline.
    Epoch/limit progress is stored separately from worker replay.
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
            )
        else:
            dataset = SimpleSavableDatasetWrapper(
                dataset,
                self.worker_config,
                cache_pool=cache_pool,
                stride_dataset=self._stride_dataset,
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
        return self._iteration.len_rank(self.dataset.len_rank())

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
        if self.num_workers == 0 and not self._iteration:
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
        """Save this rank's last-contributor checkpoints, keyed by global logical id.

        All ranks must have completed the same number of dataloader calls, including zero.
        No collectives are used. Returns None if this rank owns no last contributor.
        Use get_logical_worker_group_ids(for_save=True) to discover the saved ids.
        Without explicit logical_workers, every local physical worker contributes.
        """
        groups = self.get_logical_worker_group_ids(for_save=True)
        if not groups:
            return None
        workers = self.worker_config.physical_workers_per_rank()
        if self.num_workers == 0:
            checkpoints = [self.dataset.get_checkpoint()]
        elif self._has_workers:
            checkpoints = self._worker_command(
                "get_checkpoint", self._worker_sample_counters, self._stream_iterator is None
            )
        else:
            checkpoints = self.dataset.get_initial_checkpoint()
        if checkpoints is None:
            checkpoints = [SavableDatasetCheckpoint(state=None, offset=0) for _ in range(workers)]
        return SavableDataLoaderState(
            worker_states={
                group: checkpoints[
                    self.worker_config.logical_worker_last_physical_worker(group, self._sample_idx)
                    - self.worker_config.rank * workers
                ]
                for group in groups
            },
            logical_workers=self.worker_config.logical_worker_count(),
            micro_batch_size=self._get_batch_size(),
            iteration_index=self._sample_idx,
            boundary_offsets=self._iteration.save_state(),
        )

    def restore_state_rank(
        self,
        state: Optional[SavableDataLoaderState | Sequence[SavableDataLoaderState]],
    ) -> None:
        """Restore the logical groups needed by this rank, without collectives.

        Accept one state or a sequence of states covering get_logical_worker_group_ids().
        Extra groups are allowed; duplicate or missing groups raise ValueError.
        None leaves a fresh loader at its initial position. Restore before iteration.
        Physical topology may change if the logical worker count remains fixed.
        """
        if self._has_workers or self._epoch_iterator is not None:
            raise ValueError("Restore before starting the loader")
        if state is None:
            return
        if not isinstance(state, SavableDataLoaderState):
            state = _merge_worker_states(state)
        checkpoints = logical_checkpoints_for_rank(state, self.worker_config)
        batch_size = self._get_batch_size()
        if batch_size != state.micro_batch_size:
            if self.num_workers == 0 or batch_size is None or state.micro_batch_size is None:
                raise ValueError("Changing batch size requires subprocess workers and batching")
            if batch_size > state.micro_batch_size or state.micro_batch_size % batch_size:
                raise ValueError("The saved micro batch size must be a multiple of the new size")
            ratio = state.micro_batch_size // batch_size
            # Replay units are full pipeline outputs, so smaller batches require more outputs.
            checkpoints = [
                SavableDatasetCheckpoint(state=checkpoint.state, offset=checkpoint.offset * ratio)
                for checkpoint in checkpoints
            ]
        workers = self.worker_config.physical_workers_per_rank()
        next_worker_id = state.iteration_index % workers
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
        self._iteration.restore_state(state.boundary_offsets)
        self._sample_idx = state.iteration_index

    @deprecated(
        "`save_state` is deprecated and was renamed to `save_state_global` and will be removed "
        "in a future update. If you actually do not want to gather the states to a rank, use "
        "`save_state_rank` instead."
    )
    def save_state(self, dst_rank: int) -> Optional[SavableDataLoaderState]:
        """Deprecated. Use save_state_global or save_state_rank."""
        return self.save_state_global(dst_rank)

    def save_state_global(self, global_dst_rank: int = 0) -> Optional[SavableDataLoaderState]:
        """Gather one checkpoint per logical group to global_dst_rank.

        All ranks participate after synchronized dataloader calls. Only the destination
        receives a state. The destination is a global rank, including when using a data
        parallel group. The returned format is the same as save_state_rank.
        """
        local = self.save_state_rank()
        if self.worker_config.world_size == 1:
            return local
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
        if output is None:
            return None
        return _merge_worker_states([state for state in output if state is not None])

    @deprecated(
        "`restore_state` was renamed to `restore_state_global` and will be removed in a future update."
    )
    def restore_state(self, state: Optional[SavableDataLoaderState]) -> None:
        """Deprecated. Use restore_state_global or restore_state_rank."""
        return self.restore_state_global(state)

    def restore_state_global(
        self, state: Optional[SavableDataLoaderState], *, src_rank: Optional[int] = None
    ) -> None:
        """Restore from a global checkpoint, routing logical groups to this rank.

        Load the state on every rank, or supply src_rank to broadcast it from one rank
        within the data parallel group. Call before starting the loader and keep the
        logical worker count and processing pipeline unchanged.
        """
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
        self.restore_state_rank(state)

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
            "iteration": self._iteration.config(),
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
        return self._iteration.len_rank(self.dataset.len_rank())

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

        if self.num_workers == 0 and not self._iteration:
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
            "iteration": self._iteration.config(),
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
