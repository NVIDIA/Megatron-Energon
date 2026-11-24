# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import inspect
import os
import pickle
from dataclasses import dataclass
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

import torch.distributed

from megatron.energon.cache.base import CachePool
from megatron.energon.cache.no_cache import NoCachePool
from megatron.energon.dataloader.future import Future
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
from megatron.energon.wrappers.gc_dataset import GC_DEFAULT_EVERY_N_ITER, GcDataset
from megatron.energon.wrappers.group_batch_dataset import GroupBatchDataset
from megatron.energon.wrappers.log_sample_dataset import default_get_batch_keys
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

    _workers: list[DataLoaderWorker[TSample]] | None = None
    _exhausted_workers: list[bool]
    _next_worker_id: int = 0

    _restore_state: RankState | None = None

    _dataset: SavableDataset
    _worker_config: WorkerConfig
    _cache_pool: CachePool
    _prefetch_factor: int
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
        worker_type: WorkerType = ForkDataLoaderWorker,
        cache_pool: CachePool | None = None,
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
            watchdog_timeout_seconds: The timeout in seconds. If `None`, the watchdog is disabled.
            watchdog_initial_timeout_seconds: The initial timeout in seconds. If `None`, the timeout is the same as `watchdog_timeout_seconds`.
            fail_on_timeout: If True, stops the whole process upon timeout, after printing a stack trace.
            pin_memory: The memory pinner to use. If `None`, no memory is not pinned.
                If "automatic", the memory is pinned automatically if cuda is available.
                If a `PinMemory` instance, the instance may only be used for one `DataLoader`.
        """
        self._id = DataLoader._next_id
        DataLoader._next_id += 1

        if dataset.worker_config.num_workers == 0:
            worker_type = DataLoaderNoWorker

        if watchdog_timeout_seconds is not None:
            dataset = WatchdogDataset(
                dataset,
                worker_config=dataset.worker_config,
                timeout_seconds=watchdog_timeout_seconds,
                initial_timeout_seconds=watchdog_initial_timeout_seconds,
                fail_on_timeout=fail_on_timeout,
            )

        if gc_collect_every_n_steps > 0 and worker_type is not ThreadDataLoaderWorker:
            dataset = GcDataset(
                dataset,
                worker_config=dataset.worker_config,
                every_n_iter=gc_collect_every_n_steps,
                freeze=gc_freeze_at_start,
            )

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
            self._workers.append(worker)
            worker.start()

        for worker, worker_state in zip(self._workers, worker_states):
            worker.dataset_init(worker_state)

        if self._restore_state is not None:
            self._prefetching_samples = [
                [
                    self._pin_memory(self._restore_sample(restore_key))
                    for restore_key in worker.prefetched_restore_keys
                ]
                for worker in self._restore_state.workers
            ]
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
        finally:
            if self._current_epoch_iter is not None:
                self._current_epoch_iter.close()
                self._current_epoch_iter = None
            self._prefetching_samples = [[] for _ in self._prefetching_samples]
            if self._workers is None:
                self._dataset.close()

    def _stop_workers(self, in_del: bool) -> None:
        errors = []
        running = []
        for worker in self._workers or ():
            try:
                worker.shutdown(in_del=in_del)
            except Exception as error:
                errors.append(error)
                if worker.running():
                    running.append(worker)
        self._workers = running or None

        # The pinning thread can be waiting for a worker reply.
        try:
            self._pin_memory.shutdown(in_del=in_del)
        except Exception as error:
            errors.append(error)
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
                # Yield the sample.
                yield sample

    def __iter__(self) -> Generator[TSample, None, None]:
        # Restart the epoch iterator if was not created yet. Otherwise, the existing epoch iterator will be continued.
        # That happens e.g. when iteration was interrupted.
        if self._current_epoch_iter is None:
            if DEBUG_LEVEL >= 1:
                print("DL: Starting epoch iterator")
            self._current_epoch_iter = self._epoch_iter()
        else:
            if DEBUG_LEVEL >= 1:
                print("DL: Continuing epoch iterator")
        assert self._current_epoch_iter is not None
        # Important: Do not use yield from here, as it will delegate .close to the inner generator.
        try:
            for sample in self._current_epoch_iter:
                yield sample
        except Exception:
            self._current_epoch_iter = None
            raise
        # Reset the epoch iterator, it was exhausted.
        if DEBUG_LEVEL >= 1:
            print("DL: Closing epoch iterator")
        self._current_epoch_iter.close()
        self._current_epoch_iter = None

    def __len__(self):
        return len(self._dataset)

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
        if self._restore_state is not None:
            return self._restore_state
        prefetched_restore_keys = []
        for pending in self._prefetching_samples:
            keys = []
            for future in pending:
                try:
                    sample = future.get()
                except StopIteration:
                    break
                key = get_sample_restore_key(sample)
                if key is None or not self.can_restore_sample():
                    raise ValueError(
                        "Cannot checkpoint prefetched samples without deterministic restore keys"
                    )
                keys.append(key)
            prefetched_restore_keys.append(keys)
        worker_states: list[WorkerState | None]
        if self._workers is None:
            worker_states = [None] * self._worker_config.safe_num_workers
        else:
            worker_states = [worker.save_state() for worker in self._workers]

        return RankState(
            workers=[
                WorkerCheckpoint(state=state, prefetched_restore_keys=keys, observed_eof=eof)
                for state, keys, eof in zip(
                    worker_states, prefetched_restore_keys, self._exhausted_workers, strict=True
                )
            ],
            next_worker_id=self._next_worker_id,
            micro_batch_size=self.micro_batch_size,
            grouped_batch_config=self._grouped_batch_config,
        )

    def save_state_global(self, global_dst_rank: int) -> Sequence[RankState | None] | None:
        """
        Saves the state of the dataset globally, collecting the state from all ranks using torch
        distributed. Allows for restoring the state later using `restore_state_global`, given the
        result of this method.
        Typical scenario: Save the state to disk only on the `dst_rank`, the other ranks do not
        save the state. Later, restore the state either only loaded on the `dst_rank` or
        loading on all ranks separately using `restore_state_global`.

        Note: If you want to save/restore the state per rank separately, use `save_state_rank` and
        the corresponding `restore_state_rank`. Also, these do not rely on torch distributed.

        Args:
            global_dst_rank: The state will be gathered to this rank. The rank refers to the
                global rank, not the rank within the data parallel group.

        Returns:
            The state of the dataset (or `None`, if not on `dst_rank`).
        """
        if self._worker_config.world_size == 1:
            return [self.save_state_rank()]

        payload = None
        error = None
        group = self._worker_config.data_parallel_group
        is_destination = self._worker_config.global_rank() == global_dst_rank
        try:
            if group is not None:
                torch.distributed.get_group_rank(group, global_dst_rank)
            payload = pickle.dumps(self.save_state_rank())
        except Exception as exc:
            error = exc
        self._check_checkpoint_collective("save", error)

        output = [None] * self._worker_config.world_size if is_destination else None
        torch.distributed.gather_object(payload, output, global_dst_rank, group=group)

        states = None
        error = None
        try:
            if output is not None:
                states = [pickle.loads(item) for item in output]
        except Exception as exc:
            error = exc
        self._check_checkpoint_collective("decode saved state", error)
        return states

    def _check_checkpoint_collective(self, operation: str, error: Exception | None) -> None:
        """Share local preparation failures before peers enter the next collective."""
        errors = [None] * self._worker_config.world_size
        message = None if error is None else f"{type(error).__name__}: {error}"
        torch.distributed.all_gather_object(
            errors, message, group=self._worker_config.data_parallel_group
        )
        if error is not None:
            raise error
        failures = [f"rank {rank}: {failure}" for rank, failure in enumerate(errors) if failure]
        if failures:
            raise RuntimeError(f"Cannot {operation} dataloader checkpoint: {', '.join(failures)}")

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
            self.shutdown()
        assert self._workers is None and self._current_epoch_iter is None, (
            "Cannot restore state after iteration has started"
        )
        assert self._restore_state is None, "Restore state already set"

        if state is None:
            # Assume initial state.
            return

        assert isinstance(state, RankState)
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

        self._restore_state = resize_rank_state(
            state, batch_size, drop_last=batch_dataset is not None and batch_dataset.drop_last
        )

    def restore_state_global(
        self,
        state: Sequence[RankState | None] | None,
        *,
        src_rank: int | None = None,
    ) -> None:
        """
        Restores the saved state from `save_state_global` (in torch distributed setup).
        The global state needs be loaded on every rank that has a data loader instance.

        Optionally, one can specify a src_rank and only provide the state once.
        In case of multiple data parallel groups, you must provide the state once
        in each data parallel group. In this case the `src_rank` is the rank within the
        data parallel group.

        Args:
            state: The state to restore, as saved by `save_state_global`.
            src_rank: The rank from which the state is broadcasted (within the data parallel group, if using DP groups).
        """

        if src_rank is None or self._worker_config.world_size == 1:
            if not isinstance(state, list) or len(state) != self._worker_config.world_size:
                raise ValueError("State must be a rank-ordered list of size world_size")
            self.restore_state_rank(state[self._worker_config.rank])
            return

        group = self._worker_config.data_parallel_group
        inputs = None
        error = None
        try:
            if not 0 <= src_rank < self._worker_config.world_size:
                raise ValueError("Checkpoint source rank is outside the data parallel group")
            global_src_rank = (
                src_rank if group is None else torch.distributed.get_global_rank(group, src_rank)
            )
            if self._current_epoch_iter is not None or self._restore_state is not None:
                raise ValueError(
                    "Cannot restore while iterating or when restore state is already set"
                )
            if self._worker_config.rank == src_rank:
                if not isinstance(state, list) or len(state) != self._worker_config.world_size:
                    raise ValueError("State must be a rank-ordered list of size world_size")
                inputs = [pickle.dumps(rank_state) for rank_state in state]
            elif state is not None:
                raise ValueError("Only the source rank may supply checkpoint state")
        except Exception as exc:
            error = exc
        self._check_checkpoint_collective("prepare to restore", error)

        output = [None]
        torch.distributed.scatter_object_list(output, inputs, src=global_src_rank, group=group)
        error = None
        try:
            self.restore_state_rank(pickle.loads(output[0]))
        except Exception as exc:
            error = exc
        try:
            self._check_checkpoint_collective("restore", error)
        except Exception:
            self._restore_state = None
            raise

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
        rank_worker_id = self._worker_config.rank_worker_id(
            override_global_worker_id=restore_key.worker_id
        )
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

    def with_restored_state_global(
        self, state: Sequence[RankState | None] | None, src_rank: int | None = None
    ) -> "DataLoader[TSample]":
        """
        Use this data loader and restore the state. Useful for chaining commands. See `save_state_global` for more details.
        """
        self.restore_state_global(state, src_rank=src_rank)
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
            "pin_memory": self._config_pin_memory,
            "prefetch_factor": None
            if self._worker_config.num_workers == 0
            else self._prefetch_factor,
            "dataset": self._dataset.config(),
        }

    def __str__(self) -> str:
        return f"DataLoader(_id={self._id}, prefetch_factor={self._prefetch_factor}, worker_type={self._worker_type.__name__})"
