# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import os
import sys
import warnings
from typing import (
    Any,
    Generator,
    Generic,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
)

import torch.distributed

from megatron.energon.cache.base import CachePool
from megatron.energon.dataloader.future import CallableFuture, Future
from megatron.energon.dataloader.pin_memory import NoPinMemory, PinMemory, PinMemoryThread
from megatron.energon.dataloader.workers.base_worker import DataLoaderWorker
from megatron.energon.dataloader.workers.fork_worker import ForkDataLoaderWorker
from megatron.energon.flavors.base_dataset import SavableDataset, add_sample_restore_key
from megatron.energon.state import FlexState
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset, get_sample_restore_key
from megatron.energon.wrappers.batch_dataset import BatchDataset
from megatron.energon.wrappers.gc_dataset import GC_DEFAULT_EVERY_N_ITER, GcDataset
from megatron.energon.wrappers.watchdog_dataset import WatchdogDataset

TSample = TypeVar("TSample", covariant=True)


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

    _workers: list[DataLoaderWorker[TSample]] | None = None
    _exhausted_workers: list[bool]
    _next_worker_id: int = 0

    _restore_state: FlexState | None = None

    _dataset: SavableDataset
    _worker_config: WorkerConfig
    _prefetch_factor: int
    _worker_type: WorkerType
    _prefetching_samples: list[list[Future[TSample]]]
    _pin_memory: PinMemory[TSample]

    _current_epoch_iter: Generator[TSample, None, None] | None = None

    _spawning_process: int

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
            dataset: The dataset to load.
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
        if dataset.worker_config.num_workers == 0 and worker_type == ForkDataLoaderWorker:
            worker_type = DataLoaderWorker

        if watchdog_timeout_seconds is not None:
            dataset = WatchdogDataset(
                dataset,
                worker_config=dataset.worker_config,
                timeout_seconds=watchdog_timeout_seconds,
                initial_timeout_seconds=watchdog_initial_timeout_seconds,
                fail_on_timeout=fail_on_timeout,
            )

        if gc_collect_every_n_steps > 0:
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
        self._cache_pool = cache_pool
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
            assert prefetch_factor > 0, "prefetch_factor must be > 0 for num_workers > 0"

        self._spawning_process = os.getpid()

    def _start(self) -> None:
        """Start the workers and restore the state if available."""
        self._workers = [
            self._worker_type(self._dataset, self._worker_config, local_worker_id, self._cache_pool)
            for local_worker_id in range(self._worker_config.safe_num_workers)
        ]
        for worker in self._workers:
            worker.start()

        if self._restore_state is None:
            worker_states = [None] * self._worker_config.safe_num_workers
        else:
            worker_states = self._restore_state["worker_states"]

        assert len(worker_states) == self._worker_config.safe_num_workers, (
            "Number of initial states must match number of workers"
        )

        for worker, worker_state in zip(self._workers, worker_states):
            worker.dataset_init(worker_state)

        if self._restore_state is not None:
            self._prefetching_samples = [
                [
                    self._pin_memory(
                        CallableFuture(functools.partial(self.restore_sample, sample_key))
                    )
                    for sample_key in prefetched_samples_keys
                ]
                for prefetched_samples_keys in self._restore_state["prefetched_samples_keys"]
            ]
            self._next_worker_id = self._restore_state["next_worker_id"]
            self._exhausted_workers = [
                False if worker_state is None else worker_state["exhausted"]
                for worker_state in worker_states
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
        if self._workers is not None:
            if in_del:
                warnings.warn(
                    "Explicitly call DataLoader.shutdown() to avoid leaking workers or run as context manager.",
                    ResourceWarning,
                )
                print(
                    "WARNING: Explicitly call DataLoader.shutdown() to avoid leaking workers or run as context manager.\n",
                    end="",
                    file=sys.stderr,
                )
            for worker in self._workers:
                worker.shutdown(in_del=in_del)
            self._workers = None
        self._pin_memory.shutdown(in_del=in_del)

    def __del__(self) -> None:
        self.shutdown(in_del=True)

    def __enter__(self) -> "DataLoader[TSample]":
        # Already start if using the context manager. This ensures the lifecycle is fixed.
        # Otherwise, will start when iterating.
        self._start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown()

    def _epoch_iter(self) -> Generator[TSample, None, None]:
        """Iterate over the dataset for one epoch (i.e. all workers StopIteration).
        One epoch may also be infinite (if looping the dataset)."""
        if self._workers is None:
            self._start()
            assert self._workers is not None, "DataLoader not started"

        if all(self._exhausted_workers):
            # All workers are exhausted, restart for the next epoch.
            for worker in self._workers:
                worker.new_iter()
            self._exhausted_workers = [False] * self._worker_config.safe_num_workers

        # For all workers, enqueue prefetching samples.
        for worker_idx, (worker, exhausted) in enumerate(
            zip(self._workers, self._exhausted_workers)
        ):
            while (
                len(self._prefetching_samples[worker_idx]) < self._prefetch_factor and not exhausted
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
        print(f"{self._exhausted_workers=}\n", end="")
        while not all(self._exhausted_workers):
            # Get the next worker to prefetch samples from.
            worker_idx = self._next_worker_id
            worker = self._workers[worker_idx]
            print(f"{worker_idx=} {worker=}\n", end="")
            self._next_worker_id = (worker_idx + 1) % self._worker_config.safe_num_workers
            if self._exhausted_workers[worker_idx]:
                print(f"{worker_idx=} exhausted, continue with next worker\n", end="")
                continue
            # Pop the first sample future from the prefetching samples.
            sample_future = self._prefetching_samples[worker_idx].pop(0)
            print(f"{sample_future=}\n", end="")
            # Prefetch samples from the worker.
            while len(self._prefetching_samples[worker_idx]) < self._prefetch_factor:
                # Add a new sample future to the prefetching samples if the worker has not prefetched enough samples.
                self._prefetching_samples[worker_idx].append(
                    self._pin_memory(worker.prefetch_next())
                )
            try:
                # Get the sample from the sample future (may wait for the sample to be ready).
                sample = sample_future.get()
            except StopIteration:
                print(f"{worker_idx=} exhausted, remove from prefetching samples\n", end="")
                # If the sample future raises StopIteration, remove the worker from the list.
                self._prefetching_samples[worker_idx] = []
                self._exhausted_workers[worker_idx] = True
                continue
            else:
                print(f"{worker_idx=} got sample, yield\n", end="")
                # Yield the sample.
                yield sample

    def __iter__(self) -> Generator[TSample, None, None]:
        # Restart the epoch iterator if was not created yet. Otherwise, the existing epoch iterator will be continued.
        # That happens e.g. when iteration was interrupted.
        if self._current_epoch_iter is None:
            self._current_epoch_iter = self._epoch_iter()
        assert self._current_epoch_iter is not None
        yield from self._current_epoch_iter
        # Reset the epoch iterator, it was exhausted.
        self._current_epoch_iter.close()
        self._current_epoch_iter = None

    def __len__(self):
        return len(self._dataset)

    def _get_batch_size(self) -> int | None:
        """Try to infer micro batch size from the dataset"""
        if (
            isinstance(self._dataset, BaseWrapperDataset)
            and (bds := self._dataset._find_wrapped_dataset(BatchDataset)) is not None
        ):
            assert isinstance(bds, BatchDataset)
            return bds.batch_size
        else:
            return None

    def save_state_rank(self) -> FlexState:
        # TODO: The redist tool must be able to change the batch size.
        # That means that the redist tool shall split a saved restore key for the "BatchDataset".
        # It should also change the saved micro batch size to match that.
        # TODO @pfischer: Add changing the batch size to the docs.
        prefetched_samples_keys = [
            [get_sample_restore_key(sample_fut.get()) for sample_fut in prefetching_sample]
            for prefetching_sample in self._prefetching_samples
        ]
        if self._workers is None:
            worker_states = [None] * self._worker_config.safe_num_workers
        else:
            worker_states = [worker.save_state() for worker in self._workers]

        return FlexState(
            __class__=type(self).__name__,
            prefetched_samples_keys=prefetched_samples_keys,
            worker_states=worker_states,
            next_worker_id=self._next_worker_id,
            micro_batch_size=self._get_batch_size(),
        )

    def save_state_global(self, global_dst_rank: int) -> Sequence[FlexState | None] | None:
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
        # Fetch current rank's worker's state
        merged_state = self.save_state_rank()

        # Gather the merged states
        if self._worker_config.world_size > 1:
            output: Sequence[FlexState | None] | None
            if self._worker_config.global_rank() == global_dst_rank:
                output = [None] * self._worker_config.world_size
            else:
                # Check if the global_dst_rank is in the same group at all
                if self._worker_config.data_parallel_group is not None:
                    try:
                        _ = torch.distributed.get_group_rank(
                            self._worker_config.data_parallel_group, global_dst_rank
                        )
                    except RuntimeError:
                        raise ValueError(
                            f"global_dst_rank {global_dst_rank} is not in the group of the current rank's worker config"
                        )

                output = None

            torch.distributed.gather_object(
                merged_state,
                output,
                global_dst_rank,
                group=self._worker_config.data_parallel_group,
            )

            return output
        else:
            # Not distributed -> return the merged state
            return [merged_state]

    def restore_state_rank(self, state: FlexState | None) -> None:
        """
        Restore the state of the DataLoader on the current rank.
        The state is actually restored when the processes are started, in the iterator.
        """
        assert self._workers is None and self._current_epoch_iter is None, (
            "Cannot restore state while workers are running"
        )
        assert self._restore_state is None, "Restore state already set"

        if state is None:
            # Assume initial state.
            return

        assert isinstance(state, FlexState)
        assert state["__class__"] == type(self).__name__, "DataLoader type mismatch"
        assert state["micro_batch_size"] == self._get_batch_size(), "Micro batch size mismatch"

        self._restore_state = state

    def restore_state_global(
        self,
        state: Sequence[FlexState | None] | None,
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

        assert self._workers is None and self._current_epoch_iter is None, (
            "Cannot restore state while workers are running"
        )
        assert self._restore_state is None, "Restore state already set"

        # Only restore multi-rank if state is actually a list and we are in a distributed setup.
        # Otherwise treat as single rank state.
        if src_rank is None or self._worker_config.world_size == 1:
            assert isinstance(state, list), "State must be a list in distributed setup"
            assert len(state) == self._worker_config.world_size, (
                "State must be a list of size world_size"
            )

            # All ranks have the state
            # Select the state of the current rank
            rank_state = state[self._worker_config.rank]
        else:
            if self._worker_config.data_parallel_group is not None:
                # Only the src_rank has the state within this dp group
                try:
                    global_src_rank = torch.distributed.get_global_rank(
                        self._worker_config.data_parallel_group, src_rank
                    )
                except RuntimeError:
                    raise ValueError(
                        f"src_rank {src_rank} is not in the group of the current rank's worker config"
                    )
            else:
                # If no DP group is given, we assume the global rank is
                # the same as the data parallel rank
                global_src_rank = src_rank

            if self._worker_config.rank != src_rank:
                # Send the state to all other ranks
                assert state is None
                # Must still be a list of Nones
                state = [None] * self._worker_config.world_size
            else:
                assert isinstance(state, list), "State must be a list in distributed setup"
                assert len(state) == self._worker_config.world_size, (
                    "State must be a list of size world_size"
                )

            local_object = [None]
            torch.distributed.scatter_object_list(
                local_object,
                state,
                src=global_src_rank,
                group=self._worker_config.data_parallel_group,
            )
            rank_state = local_object[0]

        self.restore_state_rank(rank_state)

    def restore_sample(self, restore_key: tuple) -> TSample:
        """
        Restore a sample from a restore key.

        Args:
            restore_key: The restore key to restore the sample from.

        Returns:
            The restored sample.
        """
        id, global_worker_id, sample_idx = restore_key[:3]
        assert id == type(self).__name__
        restore_key = restore_key[3:]
        self._worker_config.worker_activate(
            sample_idx, override_global_rank=global_worker_id, cache_pool=self._cache_pool
        )
        try:
            return add_sample_restore_key(
                self._dataset.restore_sample(restore_key), global_worker_id, sample_idx, src=self
            )
        finally:
            self._worker_config.worker_deactivate()

    def with_restored_state_rank(self, state: FlexState | None) -> "DataLoader[TSample]":
        """
        Use this data loader and restore the state. Useful for chaining commands. See `save_state_rank` for more details.
        """
        self.restore_state_rank(state)
        return self

    def with_restored_state_global(
        self, state: Sequence[FlexState | None] | None, src_rank: int | None = None
    ) -> "DataLoader[TSample]":
        """
        Use this data loader and restore the state. Useful for chaining commands. See `save_state_global` for more details.
        """
        self.restore_state_global(state, src_rank=src_rank)
        return self

    def config(self) -> dict[str, Any]:
        """Get the configuration of the dataset."""
        return self._dataset.config()

    def __str__(self) -> str:
        return f"DataLoader(prefetch_factor={self._prefetch_factor}, worker_type={self._worker_type.__name__})"
