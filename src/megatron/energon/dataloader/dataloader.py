# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import inspect
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from threading import Condition, get_ident
from time import monotonic
from typing import (
    Any,
    ClassVar,
    Generator,
    Generic,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
)
from weakref import proxy

import torch

from megatron.energon.cache.base import CachePool
from megatron.energon.cache.no_cache import NoCachePool
from megatron.energon.dataloader.asynchronous.base import (
    Asynchronous,
    WorkerExitedError,
)
from megatron.energon.dataloader.future import Future
from megatron.energon.dataloader.logical import LogicalCheckpoint
from megatron.energon.dataloader.pin_memory import NoPinMemory, PinMemory, PinMemoryThread
from megatron.energon.dataloader.workers.base_worker import (
    DataLoaderNoWorker,
    DataLoaderWorker,
    WorkerSampleRestoreKey,
    WorkerState,
)
from megatron.energon.dataloader.workers.fork_worker import ForkDataLoaderWorker
from megatron.energon.dataloader.workers.thread_worker import ThreadDataLoaderWorker
from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import RestoreKey, SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import (
    BaseWrapperDataset,
    get_sample_restore_key,
)
from megatron.energon.wrappers.batch_dataset import BatchDataset
from megatron.energon.wrappers.bounded_dataset import BoundedDataset
from megatron.energon.wrappers.gc_dataset import GC_DEFAULT_EVERY_N_ITER, GcDataset
from megatron.energon.wrappers.group_batch_dataset import GroupBatchDataset
from megatron.energon.wrappers.log_sample_dataset import default_get_batch_keys
from megatron.energon.wrappers.stride_dataset import StrideDataset
from megatron.energon.wrappers.watchdog_dataset import WatchdogDataset

TSample = TypeVar("TSample", covariant=True)
TBatchDataset = TypeVar("TBatchDataset", BatchDataset, GroupBatchDataset)

DEBUG_LEVEL = 0


@edataclass
class WorkerCheckpoint:
    """One worker's execution snapshot and the loader's progress through its outputs."""

    state: WorkerState | None
    prefetched_restore_keys: list[RestoreKey]
    observed_eof: bool
    history: dict[int, WorkerState] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class GroupedBatchCheckpointConfig:
    """The grouped batch settings that must agree at save and restore."""

    fixed_batch_size: int | None
    drop_last: bool


@edataclass
class RankState:
    """State of a rank at a delivery boundary."""

    workers: list[WorkerCheckpoint]
    next_worker_id: int
    micro_batch_size: int | None
    grouped_batch_config: GroupedBatchCheckpointConfig | None = None
    iteration_index: int = 0
    delivered_counts: list[int] | None = None
    logical_workers: int | None = None
    physical_layout: tuple[int, int, int] | None = None

    @property
    def worker_states(self) -> list[WorkerState | None]:
        """Execution snapshots in rank-local worker order."""
        return [worker.state for worker in self.workers]

    @property
    def prefetched_restore_keys(self) -> list[list[RestoreKey]]:
        """Undelivered outputs in rank-local worker order."""
        return [worker.prefetched_restore_keys for worker in self.workers]


class WorkerType(Protocol[TSample]):
    """Protocol for a worker type, i.e. for the constructor of a worker class."""

    def __call__(
        self,
        dataset: SavableDataset,
        worker_config: WorkerConfig,
        rank_worker_id: int,
        cache_pool: CachePool | None,
    ) -> DataLoaderWorker[TSample]: ...


class DataLoader(Generic[TSample]):
    """
    Implementation for a data loader. Orchestrates the workers for prefetching samples.
    Opposing the `torch.utils.data.DataLoader`, this loader needs explicit shutdown when done,
    to avoid leaking workers (fixes a bug).
    """

    _next_id: ClassVar[int] = 0
    _id: int
    _next_epoch_id: int = 0
    _iteration_index: int = 0
    _delivered_counts: list[int]
    _logical_restore: dict[int, LogicalCheckpoint] | None
    _epochize: int | None
    _limit: int | None
    _stride: StrideDataset | None
    #: Execution wrappers outside the logical stream, from innermost to outermost.
    _execution_wrappers: tuple[type[BaseWrapperDataset], ...]
    _bounded: BoundedDataset | None

    _workers: list[DataLoaderWorker[TSample]] | None = None
    _exhausted_workers: list[bool]
    _next_worker_id: int = 0

    _restore_state: RankState | None = None

    _dataset: SavableDataset
    _worker_config: WorkerConfig
    _cache_pool: CachePool
    _prefetch_factor: int
    _worker_request_timeout: float | None
    #: Checkpoint callers take priority over the next delivery once they are waiting.
    _delivery_condition: Condition
    _delivery_thread_id: int | None
    _checkpoint_waiters: int
    _worker_shutdown_timeout: float | None
    _worker_type: WorkerType
    _prefetching_samples: list[list[Future[TSample]]]
    _pin_memory: PinMemory[TSample]

    _current_epoch_iter: Generator[TSample, None, None] | None = None

    _config_pin_memory: bool
    _spawning_process: int

    _global_sample_idx: int = 0

    def __init__(
        self,
        dataset: SavableDataset,
        *,
        prefetch_factor: int = 1,
        epochize: int | None = None,
        limit: int | None = None,
        worker_type: WorkerType = ForkDataLoaderWorker,
        cache_pool: CachePool | None = None,
        worker_request_timeout: float | None = 60,
        worker_shutdown_timeout: float | None = None,
        # Garbage collection configuration
        gc_collect_every_n_steps: int = GC_DEFAULT_EVERY_N_ITER,
        gc_freeze_at_start: bool = True,
        # Watchdog configuration
        watchdog_timeout_seconds: float | None = 60,
        watchdog_initial_timeout_seconds: float | None = None,
        fail_on_timeout: bool = False,
        # Pin memory configuration
        pin_memory: PinMemory[TSample] | None | Literal["automatic"] = "automatic",
    ):
        """
        Create the dataloader supporting saving and restoring the state.

        Args:
            dataset: The dataset to load. The loader takes ownership of the dataset, i.e. it cannot be shared and will be closed on shutdown.
            epochize: Positive output budget per rank; continue the source next pass.
            limit: Positive output budget per rank; reset the source at pass completion.
            prefetch_factor: The number of samples to prefetch from each worker.
            worker_type: The type of worker to use.
            cache_pool: If set, the cache pool to use for the dataset.
            gc_collect_every_n_steps: The number of steps after which the garbage collector is
                called. As we're usually handling large (but few) tensors here, and the python
                garbage collection is already full of objects just by importing, this can improve
                the memory footprint quite a lot, and may even be necessary to avoid memory
                overflow.
            gc_freeze_at_start: If true, the garbage collector is frozen at the start of the worker
                processes. This improves the garbage collection performance by a lot.
                In rare cases, this may cause issues and can be disabled. Keep enabled if you
                experience no issues.
            worker_request_timeout: Seconds to wait for an asynchronous startup or checkpoint
                request. Set to None to wait without a deadline.
            worker_shutdown_timeout: Seconds to allow in-flight work and cleanup during shutdown.
                The built-in async workers use a 15-second grace period by default.
            watchdog_timeout_seconds: The timeout in seconds. If `None`, the watchdog is disabled.
            watchdog_initial_timeout_seconds: The initial timeout in seconds. If `None`, the timeout is the same as `watchdog_timeout_seconds`.
            fail_on_timeout: If True, stops the whole process upon timeout, after printing a stack trace.
            pin_memory: The memory pinner to use. If `None`, no memory is not pinned.
                If "automatic", the memory is pinned automatically if cuda is available.
                If a `PinMemory` instance, the instance may only be used for one `DataLoader`.
        """
        if worker_request_timeout is not None and not 0 < worker_request_timeout < float("inf"):
            raise ValueError("worker_request_timeout must be a finite positive number or None")
        if worker_shutdown_timeout is not None and not 0 < worker_shutdown_timeout < float("inf"):
            raise ValueError("worker_shutdown_timeout must be a finite positive number or None")
        if epochize is not None and (
            not isinstance(epochize, int) or isinstance(epochize, bool) or epochize <= 0
        ):
            raise ValueError("epochize must be a positive integer or None")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ValueError("limit must be a positive integer or None")
        self._epochize, self._limit = epochize, limit
        self._logical_restore = None
        self._delivered_counts = [0] * dataset.worker_config.safe_num_workers
        self._bounded = None
        self._stride = None
        lengths = [value for value in (epochize, limit) if value is not None]
        if lengths:
            self._bounded = BoundedDataset(
                dataset,
                length=min(lengths),
                reset_source=limit is not None,
                worker_config=dataset.worker_config,
            )
            dataset = self._bounded
        if dataset.worker_config.logical_workers is not None:
            self._stride = StrideDataset(dataset, worker_config=dataset.worker_config)
            dataset = self._stride
        self._delivery_condition = Condition()
        self._delivery_thread_id = None
        self._checkpoint_waiters = 0
        self._worker_request_timeout = worker_request_timeout
        self._worker_shutdown_timeout = worker_shutdown_timeout
        self._id = DataLoader._next_id
        DataLoader._next_id += 1

        if dataset.worker_config.num_workers == 0:
            worker_type = DataLoaderNoWorker

        execution_wrappers = []
        if watchdog_timeout_seconds is not None:
            execution_wrappers.append(WatchdogDataset)
            dataset = WatchdogDataset(
                dataset,
                worker_config=dataset.worker_config,
                timeout_seconds=watchdog_timeout_seconds,
                initial_timeout_seconds=watchdog_initial_timeout_seconds,
                fail_on_timeout=fail_on_timeout,
            )

        if gc_collect_every_n_steps > 0 and worker_type is not ThreadDataLoaderWorker:
            execution_wrappers.append(GcDataset)
            dataset = GcDataset(
                dataset,
                worker_config=dataset.worker_config,
                every_n_iter=gc_collect_every_n_steps,
                freeze=gc_freeze_at_start,
            )

        self._execution_wrappers = tuple(execution_wrappers)
        self._dataset = dataset
        self._worker_config = dataset.worker_config
        self._prefetch_factor = prefetch_factor
        self._worker_type = worker_type
        self._cache_pool = NoCachePool() if cache_pool is None else cache_pool
        self._prefetching_samples = [[] for _ in range(self._worker_config.safe_num_workers)]
        self._exhausted_workers = [False] * self._worker_config.safe_num_workers
        if pin_memory == "automatic":
            # Automatic pinning
            if torch.cuda.is_available():
                # Use cuda
                self._pin_memory = PinMemoryThread(torch.device("cuda"))
            else:
                self._pin_memory = NoPinMemory()
        else:
            if pin_memory is None:
                self._pin_memory = NoPinMemory()
            else:
                self._pin_memory = pin_memory

        if self._worker_config.num_workers == 0:
            assert prefetch_factor == 1, "prefetch_factor must be 1 for num_workers == 0"
        else:
            assert prefetch_factor >= 0, "prefetch_factor must be non-negative"
        self._config_pin_memory = pin_memory is not None

        self._spawning_process = os.getpid()

        if self._worker_config.should_log(level=1):
            self._worker_config.worker_log(
                {
                    "t": "DataLoader.__init__",
                    "r": self._worker_config.rank,
                    "w": None,
                    "id": self._id,
                    "config": dataset.config(),
                }
            )

    def start(self) -> None:
        """Start the workers and restore the state if available."""
        assert self._workers is None, "DataLoader workers are already running"
        if self._logical_restore is not None:
            self._install_logical_restore()
        worker_count = self._worker_config.safe_num_workers
        worker_states = (
            [None] * worker_count
            if self._restore_state is None
            else self._restore_state.worker_states
        )
        if len(worker_states) != worker_count:
            raise ValueError("Number of initial states must match number of workers")

        self._workers = []
        try:
            self._start_workers(worker_states)
        except BaseException as error:
            try:
                self._stop_workers(in_del=False)
            except Exception as cleanup_error:
                error.add_note(f"Worker startup cleanup failed: {cleanup_error}")
            finally:
                self._prefetching_samples = [[] for _ in self._prefetching_samples]
                self._next_epoch_id = 0
            raise

    def _start_workers(self, worker_states: Sequence[WorkerState | None]) -> None:
        assert self._workers is not None
        for worker_id in range(len(worker_states)):
            worker = self._worker_type(
                self._dataset, self._worker_config, worker_id, self._cache_pool
            )
            worker.history_limit = max(2, self._prefetch_factor + 2)
            worker.request_timeout = self._worker_request_timeout
            self._workers.append(worker)
            worker.start()

        for worker_idx, (worker, worker_state) in enumerate(zip(self._workers, worker_states)):
            worker.dataset_init(worker_state)
            if self._restore_state is not None:
                history = self._restore_state.workers[worker_idx].history
                if history is not None:
                    worker.checkpoint_history(history)

        if self._restore_state is not None:
            self._prefetching_samples = [
                [
                    self._pin_memory(self._restore_sample(restore_key))
                    for restore_key in worker.prefetched_restore_keys
                ]
                for worker in self._restore_state.workers
            ]
            # Surface failed replays before iteration starts.
            for pending in self._prefetching_samples:
                for future in pending:
                    future.get_with_timeout(self._worker_request_timeout)
            self._iteration_index = self._restore_state.iteration_index
            saved_counts = self._restore_state.delivered_counts
            self._delivered_counts = (
                list(saved_counts)
                if saved_counts is not None and len(saved_counts) == len(self._workers)
                else [
                    0
                    if worker.state is None
                    else worker.state.sample_index - len(worker.prefetched_restore_keys)
                    for worker in self._restore_state.workers
                ]
            )
            self._next_worker_id = self._restore_state.next_worker_id
            self._exhausted_workers = [
                worker.observed_eof for worker in self._restore_state.workers
            ]
            # State was restored, clear
            self._restore_state = None

    def shutdown(self, in_del: bool = False) -> None:
        """
        Shutdown the workers and the pin memory thread.

        Args:
            in_del: Whether the shutdown is called from the garbage collector (in __del__).
                Users should not need to set this.
        """
        try:
            self._stop_workers(in_del=in_del)
        except Exception as error:
            try:
                self._close_dataset()
            except Exception as cleanup_error:
                error.add_note(f"Dataset cleanup failed: {cleanup_error}")
            raise
        else:
            self._close_dataset()

    def _close_dataset(self) -> None:
        iterator, self._current_epoch_iter = self._current_epoch_iter, None
        self._prefetching_samples = [[] for _ in self._prefetching_samples]
        try:
            if iterator is not None:
                iterator.close()
        finally:
            if self._workers is None:
                self._dataset.close()

    def _stop_workers(self, in_del: bool) -> None:
        errors = []
        running = []
        workers = self._workers or []
        resources = [*workers, self._pin_memory]
        async_resources = [
            resource
            for resource in resources
            if isinstance(resource, Asynchronous) and resource.running()
        ]
        grace_period = self._worker_shutdown_timeout
        if grace_period is None:
            grace_period = max(
                (resource.shutdown_grace_period for resource in async_resources), default=0
            )
        deadline = monotonic() + grace_period
        if not in_del:
            for resource in async_resources:
                try:
                    resource.begin_shutdown()
                except WorkerExitedError:
                    pass
                except Exception as error:
                    errors.append(error)
        # Interrupt blocked processes concurrently so each can be reaped within the same deadline.
        concurrent_resources = [
            worker for worker in async_resources if not in_del and worker.stop_concurrently
        ]
        with ThreadPoolExecutor(max_workers=max(1, len(concurrent_resources))) as shutdown_pool:
            pending_shutdowns = {
                id(worker): shutdown_pool.submit(worker.shutdown, in_del=in_del, deadline=deadline)
                for worker in concurrent_resources
            }
            # The pinning thread can be waiting for a worker reply. Reap the workers first.
            for resource in resources:
                try:
                    if pending := pending_shutdowns.get(id(resource)):
                        pending.result()
                    elif isinstance(resource, Asynchronous):
                        resource.shutdown(in_del=in_del, deadline=deadline)
                    else:
                        resource.shutdown(in_del=in_del)
                except Exception as error:
                    errors.append(error)
                    if (
                        isinstance(resource, DataLoaderWorker)
                        and isinstance(resource, Asynchronous)
                        and resource.running()
                    ):
                        running.append(resource)
        self._workers = running or None
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Errors while shutting down the dataloader", errors)

    def close(self) -> None:
        """Release the loader workers and dataset resources."""
        self.shutdown()

    def __del__(self) -> None:
        if (
            getattr(self, "_workers", None) is not None
            and self._spawning_process == os.getpid()
            and WorkerConfig.active_worker_config is None
        ):
            self.shutdown(in_del=True)

    def __enter__(self) -> "DataLoader[TSample]":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.shutdown()
        except Exception as error:
            if exc_value is None:
                raise
            exc_value.add_note(f"Dataloader shutdown failed: {error}")

    def _epoch_iter(self) -> Generator[TSample, None, None]:
        """Iterate over the dataset for one epoch (i.e. all workers StopIteration).
        One epoch may also be infinite (if looping the dataset)."""
        epoch_id = self._next_epoch_id
        self._next_epoch_id += 1

        if self._worker_config.should_log(level=1):
            self._worker_config.worker_log(
                {
                    "t": "DataLoader.epoch_iter",
                    "r": self._worker_config.rank,
                    "w": None,
                    "id": self._id,
                    "epoch_id": epoch_id,
                }
            )

        if self._workers is None:
            self.start()
            assert self._workers is not None, "DataLoader not started"

        if all(self._exhausted_workers):
            # All workers are exhausted, restart for the next epoch.
            for worker in self._workers:
                worker.new_iter()
            self._exhausted_workers = [False] * self._worker_config.safe_num_workers
            # Ensure deterministic interleaving across epochs by starting from worker 0
            self._next_worker_id = 0
            self._iteration_index = 0

        # For all workers, enqueue prefetching samples.
        for worker_idx, (worker, exhausted) in enumerate(
            zip(self._workers, self._exhausted_workers)
        ):
            while (
                self.num_workers
                and len(self._prefetching_samples[worker_idx]) < self._prefetch_factor
                and not exhausted
            ):
                self._prefetching_samples[worker_idx].append(
                    self._pin_memory(worker.prefetch_next())
                )

        # Main loop:
        # - Get the next worker to prefetch samples from.
        # - Prefetch samples from the worker.
        # - Pop the first sample future from the prefetching samples.
        # - Get the sample from the sample future (may wait for the sample to be prefetched).
        # - Yield the sample.
        if DEBUG_LEVEL >= 1:
            print(f"{self._exhausted_workers=}\n", end="")
        epoch_sample_idx = 0
        while not all(self._exhausted_workers):
            # Get the next worker to prefetch samples from.
            worker_idx = self._next_worker_id
            worker = self._workers[worker_idx]
            if DEBUG_LEVEL >= 2:
                print(f"{worker_idx=} {worker=}\n", end="")
            self._next_worker_id = (worker_idx + 1) % self._worker_config.safe_num_workers
            if self._exhausted_workers[worker_idx]:
                if DEBUG_LEVEL >= 1:
                    print(f"{worker_idx=} exhausted, continue with next worker\n", end="")
                continue
            # Pop the first sample future from the prefetching samples.
            pending = self._prefetching_samples[worker_idx]
            sample_future = pending.pop(0) if pending else self._pin_memory(worker.prefetch_next())
            if DEBUG_LEVEL >= 2:
                print(f"{sample_future=}\n", end="")
            # Prefetch samples from the worker.
            while (
                self.num_workers
                and len(self._prefetching_samples[worker_idx]) < self._prefetch_factor
            ):
                # Add a new sample future to the prefetching samples if the worker has not prefetched enough samples.
                self._prefetching_samples[worker_idx].append(
                    self._pin_memory(worker.prefetch_next())
                )
            try:
                # Get the sample from the sample future (may wait for the sample to be ready).
                sample = sample_future.get()
            except StopIteration:
                if DEBUG_LEVEL >= 1:
                    print(f"{worker_idx=} exhausted, remove from prefetching samples\n", end="")
                # If the sample future raises StopIteration, remove the worker from the list.
                self._prefetching_samples[worker_idx] = []
                self._exhausted_workers[worker_idx] = True
                if self._worker_config.should_log(level=1):
                    self._worker_config.worker_log(
                        {
                            "t": "DataLoader.epoch_iter.StopIteration",
                            "r": self._worker_config.rank,
                            "w": None,
                            "id": self._id,
                            "epoch_id": epoch_id,
                        }
                    )
                continue
            else:
                if DEBUG_LEVEL >= 2:
                    print(f"{worker_idx=} got sample, yield\n", end="")
                if self._worker_config.should_log(level=1):
                    keys = default_get_batch_keys(sample)
                    restore_key = get_sample_restore_key(sample)
                    restore_key_payload = (
                        {} if restore_key is None else {"restore_key": restore_key.to_json_object()}
                    )
                    self._worker_config.worker_log(
                        {
                            "t": "DataLoader.epoch_iter.yield",
                            "r": self._worker_config.rank,
                            "w": None,
                            "id": self._id,
                            "epoch_id": epoch_id,
                            "worker_id": worker_idx,
                            "worker_sample_idx": restore_key.sample_idx
                            if isinstance(restore_key, WorkerSampleRestoreKey)
                            else None,
                            "epoch_sample_idx": epoch_sample_idx,
                            "global_sample_idx": self._global_sample_idx,
                            **({} if keys is None else {"keys": keys}),
                            **restore_key_payload,
                        }
                    )
                    # Compatibility with the previous SavableDataLoader debug schema.
                    self._worker_config.worker_log(
                        {
                            "t": "SavableDataLoader.yield",
                            "r": self._worker_config.rank,
                            "w": None,
                            "id": self._id,
                            "idx": epoch_sample_idx,
                            "worker_id": worker_idx,
                            "global_idx": self._global_sample_idx,
                            **({} if keys is None else {"keys": keys}),
                            **restore_key_payload,
                        }
                    )
                epoch_sample_idx += 1
                self._global_sample_idx += 1
                self._iteration_index += 1
                self._delivered_counts[worker_idx] += 1
                # Yield the sample.
                yield sample

    def __iter__(self) -> Generator[TSample, None, None]:
        while True:
            with self._delivery_condition:
                self._delivery_condition.wait_for(
                    lambda: self._delivery_thread_id is None and not self._checkpoint_waiters
                )
                self._delivery_thread_id = get_ident()
            try:
                if self._current_epoch_iter is None:
                    # The public iterator keeps the loader alive while in use.
                    self._current_epoch_iter = type(self)._epoch_iter(proxy(self))
                sample = next(self._current_epoch_iter)
            except StopIteration:
                self._current_epoch_iter = None
                self._iteration_index = 0
                return
            except BaseException:
                self._current_epoch_iter = None
                raise
            finally:
                with self._delivery_condition:
                    self._delivery_thread_id = None
                    self._delivery_condition.notify_all()
            yield sample

    def __len__(self):
        if self._epochize is not None:
            return self._epochize if self._limit is None else min(self._epochize, self._limit)
        if self._limit is None:
            return len(self._dataset)
        result = 0
        for local in range(self._worker_config.safe_num_workers):
            assignment = self._worker_config.logical_assignment_for_physical(
                self._worker_config.physical_global_worker_id(local)
            )
            logical_length = min(
                self._worker_config.logical_worker_length(self._limit, local),
                self._bounded.dataset.len_worker(local),
            )
            result += max(
                0,
                (logical_length - assignment.stride_offset + assignment.stride - 1)
                // assignment.stride,
            )
        return result

    @property
    def dataset(self) -> SavableDataset[TSample]:
        """The runtime dataset owned by the loader."""
        return self._dataset

    @property
    def num_workers(self) -> int:
        return self._worker_config.num_workers

    def _find_batch_dataset(self, dataset_type: type[TBatchDataset]) -> TBatchDataset | None:
        dataset = self._dataset
        if isinstance(dataset, BaseWrapperDataset) and not isinstance(dataset, dataset_type):
            dataset = dataset._find_wrapped_dataset(dataset_type)
        return dataset if isinstance(dataset, dataset_type) else None

    @property
    def _grouped_batch_config(self) -> GroupedBatchCheckpointConfig | None:
        dataset = self._find_batch_dataset(GroupBatchDataset)
        if dataset is not None:
            return GroupedBatchCheckpointConfig(
                fixed_batch_size=dataset.fixed_batch_size, drop_last=dataset.drop_last
            )
        return None

    @property
    def micro_batch_size(self) -> int | None:
        """Try to infer micro-batch size from the dataset."""
        dataset = self._find_batch_dataset(BatchDataset)
        return dataset.batch_size if dataset is not None else None

    def save_state_rank(self) -> RankState:
        """Save at the next delivery boundary before allowing iteration to advance."""
        with self._delivery_condition:
            if self._delivery_thread_id == get_ident():
                raise RuntimeError("Cannot save a checkpoint from inside sample delivery")
            self._checkpoint_waiters += 1
            try:
                self._delivery_condition.wait_for(lambda: self._delivery_thread_id is None)
                return self._save_state_at_delivery_boundary()
            finally:
                self._checkpoint_waiters -= 1
                self._delivery_condition.notify_all()

    def _save_state_at_delivery_boundary(self) -> RankState:
        if self._logical_restore is not None:
            self._install_logical_restore()
        if self._restore_state is not None:
            return self._restore_state
        # Detect failures from any worker's pending outputs before asking workers to snapshot.
        prefetched_restore_keys = self._collect_pending_restore_keys()
        worker_states: list[WorkerState | None]
        if self._workers is None:
            worker_states = [None] * self._worker_config.safe_num_workers
        else:
            worker_states = [worker.save_state() for worker in self._workers]

        histories = (
            [None] * len(worker_states)
            if self._workers is None or self._worker_config.logical_workers is None
            else [worker.checkpoint_history() for worker in self._workers]
        )
        return RankState(
            iteration_index=self._iteration_index,
            delivered_counts=list(self._delivered_counts),
            physical_layout=(
                self._worker_config.rank,
                self._worker_config.world_size,
                self.num_workers,
            )
            if self._worker_config.logical_workers is not None
            else None,
            logical_workers=self._worker_config.logical_worker_count(),
            workers=[
                WorkerCheckpoint(
                    state=state, prefetched_restore_keys=keys, observed_eof=eof, history=history
                )
                for state, keys, eof, history in zip(
                    worker_states,
                    prefetched_restore_keys,
                    self._exhausted_workers,
                    histories,
                    strict=True,
                )
            ],
            next_worker_id=self._next_worker_id,
            micro_batch_size=self.micro_batch_size,
            grouped_batch_config=self._grouped_batch_config,
        )

    def _collect_pending_restore_keys(self) -> list[list[RestoreKey]]:
        prefetched_restore_keys = []
        for pending in self._prefetching_samples:
            keys = []
            for future in pending:
                try:
                    sample = future.get_with_timeout(self._worker_request_timeout)
                except StopIteration:
                    break
                key = get_sample_restore_key(sample)
                if key is None or not self.can_restore_sample():
                    raise ValueError(
                        "Cannot checkpoint prefetched samples without deterministic restore keys"
                    )
                keys.append(key)
            prefetched_restore_keys.append(keys)
        return prefetched_restore_keys

    def get_logical_worker_group_ids(self, *, for_save: bool = False) -> tuple[int, ...]:
        """Groups to load, or groups whose last consumed contributor is on this rank.

        Logical saves require synchronized delivered counts and boundary phases across ranks.
        This method and the rank checkpoint methods perform no distributed communication.
        """
        config = self._worker_config
        groups = sorted(
            {config.logical_global_worker_id(i) for i in range(config.safe_num_workers)}
        )
        if for_save:
            groups = [
                group
                for group in groups
                if config.logical_worker_last_physical_worker(group, self._iteration_index)
                // config.safe_num_workers
                == config.rank
            ]
        return tuple(groups)

    def save_state_logical_rank(self) -> tuple[tuple[int, LogicalCheckpoint], ...]:
        """Save consumed stream boundaries, keyed by logical ID, for topology changes.

        Configure an explicit ``WorkerConfig.logical_workers`` before constructing the dataset.
        Prefetch is supported; undelivered outputs do not advance the saved logical boundary.
        """
        if self._worker_config.logical_workers is None:
            raise ValueError("Logical checkpoints require an explicit logical_workers count")
        with self._delivery_condition:
            if self._delivery_thread_id == get_ident():
                raise RuntimeError("Cannot save from inside sample delivery")
            self._checkpoint_waiters += 1
            try:
                self._delivery_condition.wait_for(lambda: self._delivery_thread_id is None)
                if self._logical_restore is not None:
                    return tuple(
                        (group, deepcopy(self._logical_restore[group]))
                        for group in self.get_logical_worker_group_ids(for_save=True)
                    )
                if not self.get_logical_worker_group_ids(for_save=True):
                    return ()
                rank_state = self._save_state_at_delivery_boundary()
                result = []
                config = self._worker_config
                for group in self.get_logical_worker_group_ids(for_save=True):
                    physical = config.logical_worker_last_physical_worker(
                        group, self._iteration_index
                    )
                    local = physical % config.safe_num_workers
                    checkpoint = rank_state.workers[local]
                    consumed = (rank_state.delivered_counts or self._delivered_counts)[local]
                    if checkpoint.state is None:
                        state = None
                    elif checkpoint.observed_eof:
                        state = checkpoint.state
                    else:
                        state = (
                            checkpoint.history.get(consumed)
                            if checkpoint.history is not None
                            else None
                        )
                        if state is None:
                            raise RuntimeError(
                                "Consumed checkpoint fell outside the prefetch history"
                            )
                    if state is not None:
                        node = state.dataset
                        for wrapper in reversed(self._execution_wrappers):
                            node = wrapper._unwrap_child_state(node)
                        state = replace(state, dataset=node)
                    # Physical invocation counts and replay assignments are not logical progress.
                    state = (
                        None if state is None else replace(state, sample_index=0, logical_skip=None)
                    )
                    result.append(
                        (
                            group,
                            LogicalCheckpoint(
                                logical_worker_id=group,
                                logical_workers=config.logical_worker_count(),
                                iteration_index=rank_state.iteration_index,
                                state=deepcopy(state),
                                observed_eof=checkpoint.observed_eof,
                                micro_batch_size=rank_state.micro_batch_size,
                                grouped_batch_config=rank_state.grouped_batch_config,
                            ),
                        )
                    )
                return tuple(result)
            finally:
                self._checkpoint_waiters -= 1
                self._delivery_condition.notify_all()

    def restore_state_logical_rank(self, states) -> None:
        """Install logical group checkpoints on a fresh loader before workers start.

        Accept a mapping or a sequence of ``(logical_id, checkpoint)`` pairs. Extra groups
        are allowed so callers may load files shared by several destination ranks.
        """
        if (
            self._workers is not None
            or self._current_epoch_iter is not None
            or self._restore_state is not None
            or self._logical_restore is not None
        ):
            raise RuntimeError("Logical restore requires a fresh loader")
        if states is None:
            return
        if self._worker_config.logical_workers is None:
            raise ValueError("Set logical_workers to the saved count before building the dataset")
        items = list(states.items()) if isinstance(states, dict) else list(states)
        normalized = {}
        for group, checkpoint in items:
            if group in normalized:
                raise ValueError(f"Duplicate logical checkpoint: {group}")
            if (
                not isinstance(checkpoint, LogicalCheckpoint)
                or checkpoint.logical_worker_id != group
            ):
                raise ValueError("Logical checkpoint identity mismatch")
            if checkpoint.logical_workers != self._worker_config.logical_worker_count():
                raise ValueError("The logical worker count must remain unchanged")
            if checkpoint.grouped_batch_config != self._grouped_batch_config:
                raise ValueError("Cannot change grouped batch settings on logical restore")
            old_size, new_size = checkpoint.micro_batch_size, self.micro_batch_size
            if old_size != new_size:
                if (
                    old_size is None
                    or new_size is None
                    or not 0 < new_size < old_size
                    or old_size % new_size
                ):
                    raise ValueError(
                        "The target micro-batch size must be a smaller divisor of the saved size"
                    )
                from megatron.energon.checkpoint.rank_state import _has_pending_generator_batch

                batch_dataset = self._find_batch_dataset(BatchDataset)
                if (
                    batch_dataset is not None and inspect.isgeneratorfunction(batch_dataset.batcher)
                ) or (
                    checkpoint.state is not None
                    and _has_pending_generator_batch(checkpoint.state.dataset)
                ):
                    raise ValueError("Cannot resize a logical checkpoint with a generator batch")
                factor = old_size // new_size
                state = checkpoint.state
                if state is not None:
                    state = replace(
                        state, dataset=StrideDataset.resize_saved_cursor(state.dataset, factor)
                    )
                checkpoint = replace(
                    checkpoint,
                    state=state,
                    micro_batch_size=new_size,
                    iteration_index=checkpoint.iteration_index * factor,
                )
            normalized[group] = checkpoint
        missing = set(self.get_logical_worker_group_ids()) - normalized.keys()
        if missing:
            raise ValueError(f"Missing logical checkpoints: {sorted(missing)}")
        if len({(state.iteration_index, state.observed_eof) for state in normalized.values()}) != 1:
            raise ValueError("Logical checkpoints have inconsistent delivery or boundary phases")
        self._logical_restore = deepcopy(normalized)
        self._iteration_index = next(iter(normalized.values())).iteration_index

    def _install_logical_restore(self) -> None:
        config = self._worker_config
        entries = self._logical_restore
        assert entries is not None and self._stride is not None
        workers = []
        for local in range(config.safe_num_workers):
            checkpoint = entries[config.logical_global_worker_id(local)]
            state = deepcopy(checkpoint.state)
            if state is not None:
                dataset_state, skip = self._stride.state_for_assignment(
                    state.dataset, checkpoint.iteration_index, local, boundary=self._bounded
                )
                for wrapper in self._execution_wrappers:
                    dataset_state = wrapper._wrap_child_state(dataset_state)
                state = replace(state, dataset=dataset_state, sample_index=0, logical_skip=skip)
            workers.append(
                WorkerCheckpoint(
                    state=state, prefetched_restore_keys=[], observed_eof=checkpoint.observed_eof
                )
            )
        self._restore_state = RankState(
            workers=workers,
            next_worker_id=self._iteration_index % config.safe_num_workers,
            micro_batch_size=self.micro_batch_size,
            grouped_batch_config=self._grouped_batch_config,
            iteration_index=self._iteration_index,
            delivered_counts=[0] * config.safe_num_workers,
            logical_workers=config.logical_worker_count(),
            physical_layout=(config.rank, config.world_size, config.num_workers),
        )
        self._logical_restore = None

    def restore_state_rank(self, state: RankState | None) -> None:
        """
        Restore the state of the DataLoader on the current rank.
        The state is actually restored when the processes are started, in the iterator.
        """
        if (
            self._workers is not None
            and self._next_epoch_id == 0
            and self._current_epoch_iter is None
        ):
            # A context manager may have started idle workers before restore_state_rank().
            self._stop_workers(in_del=False)
        assert self._workers is None and self._current_epoch_iter is None, (
            "Cannot restore state after iteration has started"
        )
        assert self._restore_state is None, "Restore state already set"

        if state is None:
            # Assume initial state.
            return

        assert isinstance(state, RankState)
        if (
            state.logical_workers is not None
            and state.logical_workers != self._worker_config.logical_worker_count()
        ):
            raise ValueError("The logical worker count must remain unchanged")
        if state.physical_layout is not None and state.physical_layout != (
            self._worker_config.rank,
            self._worker_config.world_size,
            self.num_workers,
        ):
            raise ValueError("Use logical checkpoints to change the physical topology")
        grouped_config = self._grouped_batch_config
        if state.grouped_batch_config is not None and state.grouped_batch_config != grouped_config:
            raise ValueError("Cannot change grouped batch size or drop_last on checkpoint restore")
        batch_dataset = self._find_batch_dataset(BatchDataset)
        batch_size = batch_dataset.batch_size if batch_dataset is not None else None
        if (
            batch_size != state.micro_batch_size
            and batch_dataset is not None
            and inspect.isgeneratorfunction(batch_dataset.batcher)
        ):
            raise ValueError("Cannot resize a checkpoint for a generator batcher")
        from megatron.energon.checkpoint.rank_state import resize_rank_state

        if self._worker_config.logical_workers is not None and batch_size != state.micro_batch_size:
            raise ValueError("Use a logical checkpoint to resize batches with logical workers")
        self._restore_state = resize_rank_state(
            state, batch_size, drop_last=batch_dataset is not None and batch_dataset.drop_last
        )
        self._iteration_index = state.iteration_index

    def _restore_sample(self, restore_key: RestoreKey) -> Future[TSample]:
        """
        Restore a sample from a restore key.

        Args:
            restore_key: The restore key to restore the sample from.

        Returns:
            A future that will be resolved to the restored sample.
        """
        if not isinstance(restore_key, WorkerSampleRestoreKey):
            raise TypeError(f"Expected WorkerSampleRestoreKey, got {type(restore_key).__name__}")
        if self._workers is None:
            self.start()
        assert self._workers is not None
        if restore_key.logical_worker_id is None:
            rank_worker_id = self._worker_config.rank_worker_id(
                override_global_worker_id=restore_key.worker_id
            )
        else:
            candidates = [
                i
                for i in range(self._worker_config.safe_num_workers)
                if self._worker_config.logical_global_worker_id(i) == restore_key.logical_worker_id
            ]
            if not candidates:
                raise ValueError("This rank does not own the restore key's logical stream")
            # Preserve original execution context when it still belongs to this rank.
            original = (
                restore_key.worker_id
                - self._worker_config.rank * self._worker_config.safe_num_workers
            )
            rank_worker_id = original if original in candidates else candidates[0]
        return self._workers[rank_worker_id].restore_sample(restore_key)

    def restore_sample(self, restore_key: RestoreKey) -> TSample:
        """
        Restore a sample from a restore key.

        Args:
            restore_key: The restore key to restore the sample from.

        Returns:
            The restored sample.
        """
        return self._restore_sample(restore_key).get()

    def with_restored_state_rank(self, state: RankState | None) -> "DataLoader[TSample]":
        """
        Use this data loader and restore the state. Useful for chaining commands. See `save_state_rank` for more details.
        """
        self.restore_state_rank(state)
        return self

    def can_restore_sample(self) -> bool:
        """Check if the dataset can save and restore samples."""
        return self._dataset.can_restore_sample()

    def config(self) -> dict[str, Any]:
        """Get the configuration of the dataset."""
        return {
            "type": "SavableDataLoader",
            "num_workers": self._worker_config.num_workers,
            "persistent_workers": False,
            "epochize": self._epochize,
            "limit": self._limit,
            "pin_memory": self._config_pin_memory,
            "prefetch_factor": None
            if self._worker_config.num_workers == 0
            else self._prefetch_factor,
            "dataset": self._dataset.config(),
        }

    def __str__(self) -> str:
        return f"DataLoader(_id={self._id}, prefetch_factor={self._prefetch_factor}, worker_type={self._worker_type.__name__})"
