# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import multiprocessing
import queue
import threading
import time
import traceback
import weakref
from functools import partial
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    List,
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
from megatron.energon.edataclass import edataclass
from megatron.energon.errors import deprecated
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
from megatron.energon.wrappers.log_sample_dataset import default_get_keys
from megatron.energon.wrappers.watchdog_dataset import WatchdogDataset

T = TypeVar("T")


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
        self, dataset: SavableDataset[T], worker_config: WorkerConfig, cache_pool: CachePool
    ):
        """
        Args:
            dataset: The dataset to wrap.
            worker_config: The worker config to use for the dataset.
            cache_pool: The cache pool to use for the dataset.
        """
        super().__init__(dataset, worker_config=worker_config)
        self.cache_pool = cache_pool

        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._sample_index = 0
        self._state_restored = False

    def len_worker(self, worker_idx: int | None = None) -> int:
        return self.dataset.len_worker(worker_idx)

    def __iter__(self):
        self._state_restored = True
        worker_id = self.worker_config.rank_worker_id()
        global_worker_id = self.worker_config.global_worker_id()
        while self._state_restored:
            self._state_restored = False
            self.worker_config.worker_activate(self._sample_index, cache_pool=self.cache_pool)
            worker_active = True
            try:
                for src_data in self.dataset:
                    self.worker_config.worker_deactivate()
                    worker_active = False
                    sample_index = self._sample_index
                    src_data = add_sample_restore_key(
                        src_data, global_worker_id, sample_index, src=self
                    )
                    self._sample_index += 1
                    yield worker_id, sample_index, src_data
                    if self._state_restored:
                        # Restart iterator after restore
                        break
                    self.worker_config.worker_activate(
                        self._sample_index, cache_pool=self.cache_pool
                    )
                    worker_active = True
            finally:
                if worker_active:
                    self.worker_config.worker_deactivate()

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T:
        id, global_worker_id, sample_idx = restore_key[:3]
        assert id == type(self).__name__
        restore_key = restore_key[3:]
        self.worker_config.worker_activate(
            sample_idx, override_global_rank=global_worker_id, cache_pool=self.cache_pool
        )
        try:
            return add_sample_restore_key(
                self.dataset.restore_sample(restore_key),
                global_worker_id,
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
    """State of the dataset wrapper. It stores the global random states and the index of the next
    sample to be returned from the dataset. This class is not intended to be used directly, but by
    :class:`megatron.energon.SavableDatasetWrapper`."""

    #: The state of all the system random number generators
    rng: SystemRngState
    #: The state of the savable dataset
    dataset_state: FlexState
    #: Index of the next sample to be returned from the dataset
    sample_index: int

    def __repr__(self):
        return f"SavableDatasetState(rng={self.rng!r}, sample_index={self.sample_index})"


@edataclass
class SavableCheckpoint:
    """Checkpoint data for :class:`megatron.energon.SavableDatasetWrapper`. An instance is created
    regularly to be able to save the state of the dataset wrapper before the currently emitted
    sample.
    """

    #: The state of the wrapper
    state: Optional[SavableDatasetState]
    #: The time at which the checkpoint was created
    checkpoint_time: float
    #: Index of the next sample to be returned from the dataset after restoring the checkpoint
    sample_index: int


@edataclass
class SavableDatasetCheckpoint(State):
    """Checkpoint data for :class:`megatron.energon.SavableDatasetWrapper`. The checkpoint state
    represents a state before that checkpoint, with an offset (i.e. samples to be skipped)."""

    #: The state of the wrapper at the sample index when the checkpoint was created.
    state: Optional[SavableDatasetState]
    #: Offset of the checkpoint to the actual sample index to be restored.
    offset: int


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

    _workers_restore_from: List[Optional[SavableDatasetState]] = list()
    _workers_skip_samples: List[int]

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
            SavableCheckpoint(state=None, checkpoint_time=time.perf_counter(), sample_index=-1)
        ]
        self._workers_restore_from = [None] * num_workers
        self._workers_skip_samples = [0] * num_workers
        self._cmd_queues = cmd_queues
        self._result_queues = result_queues
        self.cache_pool = cache_pool

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

    def __len__(self):
        return len(self.dataset)

    def __del__(self):
        if self._cmd_thread is not None:
            # print(f"{id(self)}:{multiprocessing.current_process().ident} Closing cmd thread")
            self._running = False
            self._cmd_thread.join()
            self._command_lock = None
            self._cmd_thread = None
            # print(f"{id(self)}:{multiprocessing.current_process().ident} Cmd thread closed")

    def __iter__(self):
        # First: Set the worker offset globally for the current worker
        WorkerConfig.worker_id_offset = self._worker_offset
        self._worker_id = self.worker_config.rank_worker_id()
        global_worker_id = self.worker_config.global_worker_id()
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
                if self._workers_restore_from[self._worker_id] is not None:
                    my_state = self._workers_restore_from[self._worker_id]
                    my_ds_state = my_state.dataset_state
                    assert my_state is not None
                    if my_ds_state is None:
                        self.dataset.reset_state_deep()
                    else:
                        self.dataset.restore_state(my_ds_state)
                    self._restore_state(my_state)
                    self._workers_restore_from[self._worker_id] = None
                else:
                    # Store the initial state of the worker if we stop before the first sample
                    self._store_checkpoint()
                # If skipping, also restart the iterator to reach the start of the restored
                # checkpoint
                last_was_skip = True
                while last_was_skip:
                    dataset_has_samples = False
                    self.worker_config.worker_activate(
                        self._sample_index, cache_pool=self.cache_pool
                    )
                    worker_active = True
                    try:
                        for src_data in self.dataset:
                            self.worker_config.worker_deactivate()
                            worker_active = False
                            dataset_has_samples = True
                            if self._workers_skip_samples[self._worker_id] > 0:
                                # Skip ahead to reach the start of the restored checkpoint
                                # print(f"Skip [{self._sample_index}:{self._worker_id}] {src_data}")
                                self._workers_skip_samples[self._worker_id] -= 1
                                self._sample_index += 1
                                last_was_skip = True
                                self.worker_config.worker_activate(
                                    self._sample_index, cache_pool=self.cache_pool
                                )
                                worker_active = True
                                continue
                            last_was_skip = False
                            sample_index = self._sample_index
                            add_sample_restore_key(
                                src_data, global_worker_id, sample_index, src=self
                            )
                            self._sample_index += 1
                            self._store_checkpoint()
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
                                self._sample_index, cache_pool=self.cache_pool
                            )
                            worker_active = True
                    finally:
                        if worker_active:
                            self.worker_config.worker_deactivate()

                    # If the dataset is empty, don't try again and again
                    if not dataset_has_samples:
                        break
        finally:
            # print(f"{id(self)}:{multiprocessing.current_process().ident} Worker iter closing")
            # Always store a final checkpoint (it's likely to be saved)
            self._store_checkpoint(force=True)

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
                    state=self._save_state(),
                    checkpoint_time=time.perf_counter(),
                    sample_index=self._sample_index,
                )
            )
            if len(self._last_checkpoints) > self.n_checkpoints:
                self._last_checkpoints.pop(0)

    def _save_state(self) -> SavableDatasetState:
        """Saves the internal state"""
        (
            np_tp,
            np_state,
            pos,
            has_gauss,
            cached_gaussian,
        ) = np.random.get_state()
        return SavableDatasetState(
            rng=SystemRng.save_state(),
            dataset_state=self.dataset.save_state(),
            sample_index=self._sample_index,
        )

    def _restore_state(self, state: SavableDatasetState) -> None:
        """Restores the internal worker state"""
        assert torch.utils.data.get_worker_info() is not None, "Can only restore in worker process"
        if state.rng is None:
            SystemRng.seed(torch.initial_seed() & 0xFFFFFFFF)
        else:
            SystemRng.restore_state(state.rng)

        self._sample_index = state.sample_index
        self._last_checkpoints = [
            SavableCheckpoint(
                state=self._save_state(),
                checkpoint_time=time.perf_counter(),
                sample_index=self._sample_index,
            )
        ]

    def get_checkpoint(self, last_sample_indexes: List[int]) -> SavableDatasetCheckpoint:
        """
        Get a checkpoint given the last emitted sample indexes for all workers.

        Args:
            last_sample_indexes: The last emitted sample indexes for all workers.

        Returns:
            The found checkpoint including the offset to the next sample index
        """
        sample_index = last_sample_indexes[self._worker_id] + 1
        for checkpoint in reversed(self._last_checkpoints):
            if checkpoint.sample_index <= sample_index:
                # print(f"Found cp for {sample_index} at {checkpoint.sample_index}")
                return SavableDatasetCheckpoint(
                    state=checkpoint.state,
                    offset=sample_index - checkpoint.sample_index,
                )

        # Immediate save after restore
        if len(self._last_checkpoints) == 0:
            return SavableDatasetCheckpoint(
                state=self._workers_restore_from[self._worker_id],
                offset=self._workers_skip_samples[self._worker_id],
            )
        raise ValueError("No checkpoint found")

    def restore_checkpoint(
        self,
        worker_states: Optional[List[SavableDatasetCheckpoint]],
        worker_offset: int,
    ) -> None:
        """
        Restores the merged checkpoint from all worker processes.

        Args:
            worker_states: The state to restore for each worker
            worker_offset: The offset of the last worker which has emitted a sample. This will be
                set in all worker processes to ensure the right worker starts as first.
        """
        assert torch.utils.data.get_worker_info() is None, "Cannot restore in worker process"
        num_workers = max(self.worker_config.num_workers, 1)

        if worker_states is None:
            self._workers_restore_from = [None] * num_workers
            assert worker_offset == 0
            self._worker_offset = 0
            self._workers_skip_samples = [0] * num_workers
        else:
            assert isinstance(worker_states, list)
            assert len(worker_states) == num_workers
            assert isinstance(worker_states[0], SavableDatasetCheckpoint)

            self._worker_offset = worker_offset

            # Tear the state_list apart (which has len=num_workers)
            # and store the states in the internal arrays
            self._workers_restore_from = [state.state for state in worker_states]
            self._workers_skip_samples = [state.offset for state in worker_states]

    def get_initial_checkpoint(self) -> Optional[List[SavableDatasetCheckpoint]]:
        """
        Get the initial checkpoint for all worker processes if they have not started yet.

        Returns:
            The initial checkpoint for all worker processes and the worker offset.
        """
        assert torch.utils.data.get_worker_info() is None, (
            "Cannot get initial checkpoint in worker process"
        )
        if all(s is None for s in self._workers_restore_from):
            assert all(s == 0 for s in self._workers_skip_samples)
            # Initial state, no checkpoint
            return None

        return [
            SavableDatasetCheckpoint(
                state=state,
                offset=offset,
            )
            for state, offset in zip(self._workers_restore_from, self._workers_skip_samples)
        ]

    def can_restore_sample(self) -> bool:
        return self.dataset.can_restore_sample()

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T:
        id, global_worker_id, sample_idx = restore_key[:3]
        assert id == type(self).__name__
        restore_key = restore_key[3:]
        self.worker_config.worker_activate(sample_idx, override_global_rank=global_worker_id)
        try:
            return add_sample_restore_key(
                self.dataset.restore_sample(restore_key),
                global_worker_id,
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
    """Saved state of the :class:`megatron.energon.SavableDataLoader`. Contains the state for all worker
    processed of a single rank."""

    #: The internal state of the dataset (for each worker process)
    worker_states: List[Union[SavableDatasetCheckpoint, FlexState]]
    #: Which worker will be the next to emit a sample. Used to restore the proper order
    next_worker_id: int

    #: The micro batch size that was used, if available.
    #: On restore, this is used to compare the new and old micro batch size.
    micro_batch_size: Optional[int]


class SavableDataLoader(DataLoader[T], Generic[T]):
    """DataLoader that supports saving and restoring the state of the dataset.
    When restoring, the dataloader and dataset must be instantiated with the exactly same
    parameters.

    How this works (for no worker processes)
    ----------------------------------------

    1. The state of the dataset is saved using :meth:`megatron.energon.SavableDataset.save_state`
    2. (for compatibility) The state of the dataset is converted to using inner arrays using
       :meth:`megatron.energon.SavableDataset.merge_states`.
    3. The state can be restored using :meth:`megatron.energon.SavableDataset.restore_state` given the
       previously saved (and merged) state.

    How this works (for worker processes)
    -------------------------------------

    - First issue is, that worker processes work with internal queues between processes to pass
      loaded samples to the main process (also to perform collating). This means that the whole
      state of the dataset is not directly accessible from the main process.
    - To solve this issue, the dataset regularly saves a checkpoint of its state to be able to
      resume from that state (and skip the samples that have already been yielded).
    - To have a consistent state, the sample index from the latest yielded samples is saved for all
      worker instances. Thus, the main process knows exactly which sample indexes should come next
      from which worker.
    - Internally, pytorch iterates through the workers in order to retrieve the next worker's
      samples. Unfortunately, that next worker index cannot be restored in pytorch's dataloader,
      thus the workers are shifted internally by that offset
      (see :attr:`megatron.energon.WorkerConfig.worker_id_offset`).

    1. The dataset is wrapped in a :class:`megatron.energon.SavableDatasetWrapper`. This allows the main
       process to communicate with the worker and send commands to the workers and retrieve the
       results.
    2. The state of the dataset is saved using
       :meth:`megatron.energon.SavableDatasetWrapper.get_checkpoint`. This gives the last checkpoint
       from the requested sample index and stores the offset (i.e. number of samples to skip) from
       that checkpoint.
    3. The state is merged using :meth:`megatron.energon.SavableDatasetWrapper.merge_checkpoints`. This
       merges the states of all workers and returns a single state that can be used to restore the
       state of the dataset.
    4. The state can be restored using :meth:`megatron.energon.SavableDatasetWrapper.restore_state`
       before a worker is started, such that all workers initially receive the same state array.
       The worker firstly sets the worker index offset, then uses its (shifted) own index to get its
       required state from the merged state array.

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
    # dataset is not iterated multiple times in parallel. The state will proceed.
    _persistent_iterator: Optional[Iterator[T]] = None
    #: The index of the current worker. -1 if not started yet.
    _worker_sample_counters: List[int]
    #: Id of the next worker to retrieve data from
    _next_worker_id: int = 0
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
    ):
        """
        Create the dataloader supporting saving and restoring the state.

        Args:
            dataset: The dataset to load.
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
            )
        else:
            dataset = SimpleSavableDatasetWrapper(
                dataset, self.worker_config, cache_pool=cache_pool
            )

        self._worker_sample_counters = [-1] * num_procs

        kwargs = {}
        if self.worker_config.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = prefetch_factor

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

    @staticmethod
    def next_id() -> int:
        next_id = SavableDataLoader._next_id
        SavableDataLoader._next_id += 1
        return next_id

    def __iter__(self):
        outerself = self

        class InnerIterator:
            """Internal class which keeps the iterator alive across multiple `iter()` calls.
            If the inner iterator is exhausted, will also exhaust and a new instance is needed.
            Also saves the last sample index and the next worker id.
            """

            finished: bool = False
            iter_idx: int = 0
            id: int

            def __init__(self, iterator):
                self._iterator = iterator
                self.id = outerself.next_id()
                if outerself.worker_config.should_log(level=1):
                    outerself.worker_config.worker_log(
                        {
                            "t": "SavableDataLoader.iter",
                            "r": outerself.worker_config.rank,
                            "w": None,
                            "id": outerself.id,
                            "iter_id": self.id,
                        }
                    )

                # self._debugf = open(
                #     f"worker_samples_rank{outerself.worker_config.rank:02}_t{int(time.time())}.log", "w"
                # )

            def __iter__(self):
                return self

            def __next__(self):
                try:
                    worker_id, sample_idx, sample = next(self._iterator)
                    outerself._worker_sample_counters[worker_id] = sample_idx
                    # If the next sample will be from the first worker, we can safely resume
                    outerself._next_worker_id = (worker_id + 1) % max(outerself.num_workers, 1)
                    # self._debugf.write(
                    #     f"[w={worker_id}, s={sample_idx}] {self._sample_str(sample)}\n"
                    # )
                    # self._debugf.flush()
                    if outerself.worker_config.should_log(level=1):
                        keys = default_get_keys(sample)
                        outerself.worker_config.worker_log(
                            {
                                **{
                                    "t": "SavableDataLoader.yield",
                                    "r": outerself.worker_config.rank,
                                    "w": None,
                                    "id": outerself.id,
                                    "iter_id": self.id,
                                    "worker_id": worker_id,
                                    "worker_idx": sample_idx,
                                    "idx": outerself._sample_idx,
                                    "iter_idx": self.iter_idx,
                                    "global_idx": outerself._global_sample_idx,
                                },
                                **({} if keys is None else {"keys": keys}),
                            }
                        )
                    outerself._sample_idx += 1
                    outerself._global_sample_idx += 1
                    self.iter_idx += 1
                    return sample
                except StopIteration:
                    self.finished = True
                    outerself._next_worker_id = 0
                    if outerself.worker_config.should_log(level=1):
                        outerself.worker_config.worker_log(
                            {
                                "t": "SavableDataLoader.StopIteration",
                                "r": outerself.worker_config.rank,
                                "w": None,
                                "id": outerself.id,
                                "iter_id": self.id,
                            }
                        )
                    raise

        if self.num_workers > 0:
            # Always keep same iterator alive, as long as it yields data
            if self._persistent_iterator is None or self._persistent_iterator.finished:
                self._persistent_iterator = InnerIterator(super().__iter__())
                self._sample_idx = 0
                # print("New Iterator", self._persistent_iterator)
            return self._persistent_iterator
        else:
            return InnerIterator(super().__iter__())

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
        if isinstance(self.dataset, SavableDatasetWrapper):
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

    def save_state_rank(self) -> Optional[SavableDataLoaderState]:
        """
        Saves the state of the dataset for the current rank. Allows for restoring the state later
        using `restore_state_rank`, given the result of this method.

        Returns:
            The state of the dataset.
        """
        # Fetch current rank's worker's state
        if self.num_workers == 0:
            # No workers configured
            assert isinstance(self.dataset, SimpleSavableDatasetWrapper)
            worker_states = [self.dataset.save_state()]
            assert self._next_worker_id == 0
        elif self._persistent_iterator is None:
            # Workers configured, but not started yet.
            # If a state has already been restored, it will be returned.
            assert isinstance(self.dataset, SavableDatasetWrapper)
            worker_states = self.dataset.get_initial_checkpoint()
        else:
            # Fetch from worker processes
            worker_states = self._worker_command("get_checkpoint", self._worker_sample_counters)

        if worker_states is None:
            return None

        # Merge the states
        merged_state = SavableDataLoaderState(
            worker_states=worker_states,
            next_worker_id=self._next_worker_id,
            micro_batch_size=self._get_batch_size(),
        )

        # Not distributed -> return the merged state
        return merged_state

    def restore_state_rank(self, state: Optional[SavableDataLoaderState]) -> None:
        """
        Restores the saved state for the current rank.

        Args:
            state: The state to restore, as saved by `save_state_rank`.
        """
        assert self._persistent_iterator is None, "Cannot restore state while workers are running"
        if state is None:
            # Assume initial state
            return
        assert isinstance(state, SavableDataLoaderState)

        old_micro_batch_size = state.micro_batch_size
        micro_batch_size = self._get_batch_size()

        if isinstance(self.dataset, SavableDataset):
            assert micro_batch_size == old_micro_batch_size, (
                "Changing micro batch size is not allowed without workers"
            )

            assert len(state.worker_states) == 1
            assert isinstance(state.worker_states[0], FlexState)
            self.dataset.restore_state(state.worker_states[0])
        else:
            assert isinstance(self.dataset, SavableDatasetWrapper)
            assert all(isinstance(s, SavableDatasetCheckpoint) for s in state.worker_states)

            # Check batch sizes (before and after)
            if micro_batch_size != old_micro_batch_size:
                assert micro_batch_size is not None and old_micro_batch_size is not None, (
                    "Cannot resume with different batching mode "
                    "(batching to non-batching or vice versa)"
                )

                if micro_batch_size > old_micro_batch_size:
                    raise ValueError(
                        "Resuming with larger micro batch size is not allowed: "
                        f"{micro_batch_size} > {state.micro_batch_size}"
                    )
                elif (
                    micro_batch_size < old_micro_batch_size
                    and old_micro_batch_size % micro_batch_size != 0
                ):
                    raise ValueError(
                        "Resuming with smaller micro batch size only allowed if the old "
                        f"micro batch size is a multiple of the new one: {micro_batch_size} < {state.micro_batch_size}"
                    )
                batch_size_ratio = old_micro_batch_size // micro_batch_size
                for worker_state in state.worker_states:
                    assert isinstance(worker_state, SavableDatasetCheckpoint)
                    # When resuming with a smaller micro batch size, the offset must be scaled
                    # up to the new micro batch size to skip the same number of samples as before.
                    worker_state.offset *= batch_size_ratio

            self.dataset.restore_checkpoint(state.worker_states, worker_offset=state.next_worker_id)

            # Initialize the worker-sample counters so that every worker owns a valid
            # "last emitted sample" index.  Workers that have not emitted anything yet keep
            # the default value ``-1``.

            assert isinstance(state.worker_states, list)

            self._worker_sample_counters = [
                (
                    ws.state.sample_index - 1
                    if (isinstance(ws, SavableDatasetCheckpoint) and ws.state is not None)
                    else -1
                )
                for ws in state.worker_states
            ]

            self._next_worker_id = state.next_worker_id

    @deprecated(
        "`save_state` is deprecated and was renamed to `save_state_global` and will be removed "
        "in a future update. If you actually do not want to gather the states to a rank, use "
        "`save_state_rank` instead."
    )
    def save_state(self, dst_rank: int) -> Optional[Sequence[Optional[SavableDataLoaderState]]]:
        """Deprecated. Use `save_state_global` (or `save_state_rank`) instead."""

        return self.save_state_global(dst_rank)

    def save_state_global(
        self, global_dst_rank: int
    ) -> Optional[Sequence[Optional[SavableDataLoaderState]]]:
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
        if self.worker_config.world_size > 1:
            output: Optional[Sequence[Optional[SavableDataLoaderState]]]
            if self.worker_config.global_rank() == global_dst_rank:
                output = [None] * self.worker_config.world_size
            else:
                # Check if the global_dst_rank is in the same group at all
                if self.worker_config.data_parallel_group is not None:
                    try:
                        _ = torch.distributed.get_group_rank(
                            self.worker_config.data_parallel_group, global_dst_rank
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
                group=self.worker_config.data_parallel_group,
            )

            return output
        else:
            # Not distributed -> return the merged state
            return [merged_state]

    @deprecated(
        "`restore_state` was renamed to `restore_state_global` and will be removed in a future update."
    )
    def restore_state(
        self,
        state: Optional[Sequence[Optional[SavableDataLoaderState]]],
    ) -> None:
        """Deprecated. Use `restore_state_global` (or `restore_state_rank`) instead."""

        return self.restore_state_global(state)

    def restore_state_global(
        self,
        state: Optional[Sequence[Optional[SavableDataLoaderState]]],
        *,
        src_rank: Optional[int] = None,
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

        assert self._persistent_iterator is None, "Cannot restore state while workers are running"

        # Only restore multi-rank if state is actually a list and we are in a distributed setup.
        # Otherwise treat as single rank state.
        if src_rank is None or self.worker_config.world_size == 1:
            assert isinstance(state, list), "State must be a list in distributed setup"
            assert len(state) == self.worker_config.world_size, (
                "State must be a list of size world_size"
            )

            # All ranks have the state
            # Select the state of the current rank
            rank_state = state[self.worker_config.rank]
        else:
            if self.worker_config.data_parallel_group is not None:
                # Only the src_rank has the state within this dp group
                try:
                    global_src_rank = torch.distributed.get_global_rank(
                        self.worker_config.data_parallel_group, src_rank
                    )
                except RuntimeError:
                    raise ValueError(
                        f"src_rank {src_rank} is not in the group of the current rank's worker config"
                    )
            else:
                # If no DP group is given, we assume the global rank is
                # the same as the data parallel rank
                global_src_rank = src_rank

            if self.worker_config.rank != src_rank:
                # Send the state to all other ranks
                assert state is None
                # Must still be a list of Nones
                state = [None] * self.worker_config.world_size
            else:
                assert isinstance(state, list), "State must be a list in distributed setup"
                assert len(state) == self.worker_config.world_size, (
                    "State must be a list of size world_size"
                )

            local_object = [None]
            torch.distributed.scatter_object_list(
                local_object,
                state,
                src=global_src_rank,
                group=self.worker_config.data_parallel_group,
            )
            rank_state = local_object[0]

        self.restore_state_rank(rank_state)

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
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": None if self.num_workers == 0 else self.prefetch_factor,
            "dataset": self.dataset.config(),
        }


class BasicDataLoader(DataLoader[T], Generic[T]):
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
    ):
        """
        Create the dataloader supporting saving and restoring the state.

        Args:
            dataset: The dataset to load.
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
            dataset, worker_config=self.worker_config, cache_pool=cache_pool
        )

        self._worker_sample_counters = [0] * max(self.worker_config.num_workers, 1)

        kwargs = {}
        if self.worker_config.num_workers > 0:
            # These must not be specified for num_workers =0
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = prefetch_factor

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

    def __iter__(self):
        outerself = self

        class InnerIterator:
            """Internal class which keeps the iterator alive across multiple `iter()` calls.
            If the inner iterator is exhausted, will also exhaust and a new instance is needed.
            Also saves the last sample index and the next worker id.
            """

            iter_idx: int = 0
            id: int

            def __init__(self, iterator):
                self._iterator = iterator
                self.id = SavableDataLoader.next_id()

                if outerself.worker_config.should_log(level=1):
                    outerself.worker_config.worker_log(
                        {
                            "t": "BasicDataLoader.iter",
                            "r": outerself.worker_config.rank,
                            "w": None,
                            "id": outerself.id,
                            "iter_id": self.id,
                        }
                    )

            def __iter__(self):
                return self

            def __next__(self):
                try:
                    worker_id, sample_idx, sample = next(self._iterator)
                    # If the next sample will be from the first worker, we can safely resume
                    self.next_worker_id = (worker_id + 1) % max(outerself.num_workers, 1)
                    if outerself.worker_config.should_log(level=1):
                        keys = default_get_keys(sample)
                        outerself.worker_config.worker_log(
                            {
                                **{
                                    "t": "BasicDataLoader.yield",
                                    "r": outerself.worker_config.rank,
                                    "w": None,
                                    "id": outerself.id,
                                    "iter_id": self.id,
                                    "worker_id": worker_id,
                                    "worker_idx": sample_idx,
                                    "idx": self.iter_idx,
                                    "iter_idx": self.iter_idx,
                                    "global_idx": outerself._sample_idx,
                                },
                                **({} if keys is None else {"keys": keys}),
                            }
                        )
                    outerself._sample_idx += 1
                    self.iter_idx += 1
                    return sample
                except StopIteration:
                    self.next_worker_id = 0
                    if outerself.worker_config.should_log(level=1):
                        outerself.worker_config.worker_log(
                            {
                                "t": "BasicDataLoader.StopIteration",
                                "r": outerself.worker_config.rank,
                                "w": None,
                                "id": outerself.id,
                                "iter_id": self.id,
                            }
                        )
                    raise

        return InnerIterator(super().__iter__())

    def config(self):
        """Get the configuration, which defines the dataset. Useful in conjunction with `save_state`
        and `restore_state` to match the configuration as well."""
        return {
            "type": type(self).__qualname__,
            "num_workers": self.num_workers,
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
