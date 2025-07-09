# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import multiprocessing
import os
import queue
import sys
import threading
import warnings
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    ParamSpec,
    Protocol,
    TypeVar,
    override,
)

from megatron.energon.cache.base import CachePool
from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import SavableDataset, add_sample_restore_key
from megatron.energon.rng import SystemRng
from megatron.energon.state import FlexState
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset, SampleIndex, get_sample_restore_key
from megatron.energon.wrappers.batch_dataset import BatchDataset
from megatron.energon.wrappers.gc_dataset import GC_DEFAULT_EVERY_N_ITER, GcDataset, gc_init_worker
from megatron.energon.wrappers.watchdog_dataset import WatchdogDataset

P = ParamSpec("P")
R = TypeVar("R", covariant=True)
T = TypeVar("T")
TSelf = TypeVar("TSelf", bound="DataLoaderWorker")
TSample = TypeVar("TSample")


class QueueProtocol(Protocol[TSample]):
    def get(self, /) -> TSample: ...

    def put(self, item: TSample, /) -> None: ...


class Future(Protocol[R]):
    def get(self) -> R: ...


class DoneFuture(Future[TSample]):
    """Future that is already done."""

    def __init__(self, result: TSample):
        self._result = result

    def get(self) -> TSample:
        return self._result


class CallableFuture(Future[R]):
    """Future that calls a callable to get the result."""

    _callable: Callable[[], R]
    _value: R
    _exception: Exception

    def __init__(self, callable: Callable[[], R]):
        self._callable = callable

    def get(self) -> R:
        if not hasattr(self, "_value") and not hasattr(self, "_exception"):
            try:
                self._value = self._callable()
            except Exception as e:
                self._exception = e
        if hasattr(self, "_exception"):
            raise self._exception
        return self._value

    @staticmethod
    def chain(future: Future[T], fn: Callable[[Future[T]], R]) -> Future[R]:
        """
        Chain a function to a future.

        Args:
            future: The future which provides the input for the function.
            fn: The function to call on the result of the future, to transform the result.

        Returns:
            A future that will be resolved to the result of the function given the result of the future.
        """
        return CallableFuture(lambda: fn(future))


class ExceptionFuture(Future[Any]):
    """Future that raises an exception."""

    def __init__(self, exception: Exception):
        self._exception = exception

    def get(self) -> Any:
        raise self._exception


class DataLoaderWorker(Generic[TSample]):
    """
    A worker for a :class:`DataLoader`.

    The basic implementation iterates the dataset.
    The async extension implements the main commands via a command and results queue.
    """

    dataset: SavableDataset[TSample]
    worker_config: WorkerConfig

    _rank_worker_id: int
    _global_worker_id: int
    _seed: int
    _cache_pool: CachePool | None
    _sample_index: SampleIndex
    _exhausted: bool = True

    def __init__(
        self,
        dataset: SavableDataset[TSample],
        worker_config: WorkerConfig,
        rank_worker_id: int,
        cache_pool: CachePool | None,
    ):
        """
        Initialize the worker.

        Args:
            dataset: The dataset to iterate over.
            worker_config: The worker configuration.
            rank_worker_id: The rank of the worker.
            cache_pool: The cache pool to use.
        """
        self.dataset = dataset
        self.worker_config = worker_config
        self._rank_worker_id = rank_worker_id
        self._global_worker_id = worker_config.global_worker_id(rank_worker_id)
        self._seed = self.worker_config.worker_seed(rank_worker_id)
        self._cache_pool = cache_pool

    # ------------------------------------------------------------------------------------------------
    # Section: Main control methods

    def start(self) -> None:
        """
        Start the worker.
        """
        pass

    def shutdown(self) -> None:
        """
        Shutdown the worker.
        """
        pass

    def running(self) -> bool:
        """
        Check if the worker is running.
        """
        return True

    def _assert_running(self) -> None:
        """
        Assert that the worker is running and alive.
        """
        assert self.running(), "Worker must be running"

    # ------------------------------------------------------------------------------------------------
    # Section: Worker methods

    def dataset_init(self, state: FlexState | None) -> None:
        """
        Initialize the worker (may restore the state).
        Calls `new_iter` if the worker is not exhausted and also initially (`state=None`).

        Args:
            state: The state to restore the worker from or None for using the initial state.
        """
        # This is called in the worker context (process/thread).
        self._sample_index = SampleIndex(worker_config=self.worker_config, src=self)
        assert self._global_worker_id == self.worker_config.global_worker_id(), (
            "Global worker ID mismatch"
        )
        assert self._seed == self.worker_config.worker_seed(self._rank_worker_id), "Seed mismatch"
        if state is None:
            self.dataset.reset_state_deep()
            self.new_iter()
        else:
            assert state["__class__"] == "DataLoaderWorker", "Worker type mismatch"
            self._sample_index.restore_state(state["_sample_index"])
            self.dataset.restore_state(state["datasets"][0])
            if not state["exhausted"]:
                self.new_iter()

    def new_iter(self) -> None:
        """
        Start a new iterator of the dataset.
        Called after the dataset is initialized and to start a new epoch (if the dataset is not infinite).
        The iterator is stored in the worker and is used by the `prefetch_next` method, which calls `next` on it.
        Updates the exhausted flag to False.
        """
        # This is called in the worker context (process/thread).
        self._dataset_iter = iter(self.dataset)
        self._exhausted = False

    def prefetch_next(self) -> Future[TSample]:
        """
        Fetch the next sample (i.e. call `next` on the iterator) and return a future for getting the result.
        Updates the exhausted flag if the iterator is exhausted.

        Returns:
            A future that will either be resolved to the next sample or raise StopIteration if the iterator is exhausted.
        """
        # This is called in the worker context (process/thread).
        assert self._dataset_iter is not None, "start_iter must be called before prefetch_next"
        with self._sample_index.ctx() as sample_idx:
            self.worker_config.worker_activate(sample_idx, cache_pool=self._cache_pool)
            try:
                next_sample = next(self._dataset_iter)
                add_sample_restore_key(next_sample, self._global_worker_id, sample_idx, src=self)
            except StopIteration as e:
                self._exhausted = True
                return ExceptionFuture(e)
            finally:
                self.worker_config.worker_deactivate()
        return DoneFuture(next_sample)

    def save_state(self) -> FlexState:
        """
        Save the state of the worker.
        """
        # This is called in the worker context (process/thread).
        return FlexState(
            __class__="DataLoaderWorker",
            rng=SystemRng.save_state(),
            dataset=self.dataset.save_state(),
            exhausted=self._exhausted,
            _sample_index=self._sample_index.save_state(),
        )


class _DataLoaderAsynchronousWorker(DataLoaderWorker[TSample]):
    """
    Extension of the `DataLoaderWorker`, which implements commands via a command and results queue.

    There are different implementations of the async worker:
    - :class:`ForkDataLoaderWorker` - A worker that forks a new process for each worker.
    - :class:`ThreadDataLoaderWorker` - A worker that uses threads to execute the commands.
    """

    _cmd_queue: QueueProtocol["WorkerCommand"]
    _result_queue: QueueProtocol["WorkerResult"]
    _next_future_id: int
    _futures: dict[int, "FutureImpl"]

    def __init__(
        self,
        dataset: SavableDataset,
        worker_config: WorkerConfig,
        rank_worker_id: int,
        cmd_queue: QueueProtocol["WorkerCommand"],
        result_queue: QueueProtocol["WorkerResult"],
        cache_pool: CachePool | None,
    ):
        super().__init__(dataset, worker_config, rank_worker_id, cache_pool)
        assert worker_config.num_workers > 0, "Async workers require num_workers > 0"
        self._cmd_queue = cmd_queue
        self._result_queue = result_queue
        self._next_future_id = 0
        self._futures = {}

    # ------------------------------------------------------------------------------------------------
    # Section: Remote call implementation

    @edataclass
    class WorkerResult:
        """Internal class for communicating a result from the worker via the result queue."""

        future_id: int
        result: Any = None
        exception: Exception | None = None

    @edataclass
    class WorkerCommand:
        """Internal class for communicating a command to the worker via the command queue."""

        cmd: str
        args: tuple[Any, ...]
        kwargs: dict[str, Any]
        future_id: int

    class FutureImpl(Future[Any]):
        """Class for returning a future result from the worker.."""

        _worker: "_DataLoaderAsynchronousWorker"
        _future_id: int
        _result: Any
        _exception: Exception

        def __init__(self, worker: "_DataLoaderAsynchronousWorker", future_id: int):
            self._worker = worker
            self._future_id = future_id

        def get(self) -> Any:
            if not hasattr(self, "_result") and not hasattr(self, "_exception"):
                self._worker._wait_for_worker_result(self._future_id)
            if hasattr(self, "_exception"):
                raise self._exception
            return self._result

        def _set_result(self, result: Any) -> None:
            self._result = result

        def _set_exception(self, exception: Exception) -> None:
            self._exception = exception

    def _wait_for_worker_result(self, future_id: int) -> None:
        """
        Wait for the result of a future.
        If another result comes first, update the corresponding future.

        Args:
            future_id: The ID of the future to wait for.
        """
        while True:
            res = self._result_queue.get()
            fut = self._futures.pop(res.future_id)
            if res.exception is not None:
                fut._set_exception(res.exception)
            else:
                fut._set_result(res.result)
            if res.future_id == future_id:
                return
            else:
                continue

    def _worker_call(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        """
        Call a function in the worker and return a future for getting the result.
        The function must be an instance method of `self`. Uses the name to identify the function in the worker
        instance.

        Args:
            fn: The function to call.
            *args: The arguments to pass to the function.
            **kwargs: The keyword arguments to pass to the function.
        """
        self._assert_running()
        future_id = self._next_future_id
        self._next_future_id += 1

        self._futures[future_id] = self.FutureImpl(self, future_id)
        self._cmd_queue.put(
            self.WorkerCommand(cmd=fn.__name__, args=args, kwargs=kwargs, future_id=future_id)
        )
        return self._futures[future_id]

    def _worker_run(
        self, cmd_queue: QueueProtocol[WorkerCommand], result_queue: QueueProtocol[WorkerResult]
    ) -> None:
        """
        The worker main loop.
        It waits for commands via the command queue and executes them.
        The functions to call are identified by their name.
        The result of the call is put into the result queue.
        The worker exits when the command `_shutdown_worker` is received.

        Args:
            cmd_queue: The command queue to wait for commands.
            result_queue: The result queue to put the results into.
        """
        SystemRng.seed(self._seed)
        import torch.utils.data._utils

        torch.utils.data._utils.worker._worker_info = torch.utils.data._utils.worker.WorkerInfo(
            id=self._rank_worker_id,
            num_workers=self.worker_config.num_workers,
            seed=self._seed,
            dataset=self.dataset,
        )
        self._global_worker_id = self.worker_config.global_worker_id()
        self.worker_config.assert_worker()
        while True:
            cmd = cmd_queue.get()
            if cmd.cmd == "_shutdown_worker":
                break
            try:
                fn = getattr(self, cmd.cmd)
                result = getattr(fn, "_orig")(self, *cmd.args, **cmd.kwargs)
            except Exception as e:
                result_queue.put(self.WorkerResult(future_id=cmd.future_id, exception=e))
            else:
                result_queue.put(self.WorkerResult(future_id=cmd.future_id, result=result))
                del result

    # ------------------------------------------------------------------------------------------------
    # Section: Worker methods - now calling to workers via queues.

    def _wrk_shutdown_worker(self) -> None:
        """Shutdown the worker. The actual shutdown is handled in the _worker_run method."""
        # This is not actually a recursive call, because the worker loop will exit before calling this method.
        self._worker_call(self._wrk_shutdown_worker)

    def _wrk_dataset_init(self, initial_state: FlexState | None) -> None:
        """Wraps the super class method to call it in the worker process."""
        super().dataset_init(initial_state)

    def _wrk_new_iter(self) -> None:
        """Wraps the super class method to call it in the worker process."""
        super().new_iter()

    def _wrk_prefetch_next(self) -> TSample:
        """Wraps the super class method to call it in the worker process."""
        # The super class implementation already returns a resolved future (to be interface compatible),
        # so immediately resolve the future to the result (get returns immediately).
        return super().prefetch_next().get()

    def _wrk_save_state(self) -> FlexState:
        """Wraps the super class method to call it in the worker process."""
        return super().save_state()

    @override
    def dataset_init(self, initial_state: FlexState | None) -> None:
        self._worker_call(self._wrk_dataset_init, initial_state).get()

    @override
    def new_iter(self) -> None:
        self._worker_call(self._wrk_new_iter).get()

    @override
    def prefetch_next(self) -> Future[TSample]:
        # Do not resolve the future here, but return it.
        return self._worker_call(self._wrk_prefetch_next)

    @override
    def save_state(self) -> FlexState:
        return self._worker_call(self._wrk_save_state).get()


class ForkDataLoaderWorker(_DataLoaderAsynchronousWorker[TSample], Generic[TSample]):
    """
    Implements the `DataLoaderWorker` interface using processes.
    """

    _process: multiprocessing.Process | None = None
    _cmd_queue: multiprocessing.Queue
    _result_queue: multiprocessing.Queue

    _spawning_process: int

    def __init__(
        self,
        dataset: SavableDataset,
        worker_config: WorkerConfig,
        rank_worker_id: int,
        cache_pool: CachePool | None,
    ):
        super().__init__(
            dataset,
            worker_config=worker_config,
            rank_worker_id=rank_worker_id,
            cmd_queue=multiprocessing.Queue(),
            result_queue=multiprocessing.Queue(),
            cache_pool=cache_pool,
        )
        self._spawning_process = os.getpid()

    def _check_parent_process(self, evt_exit: threading.Event) -> None:
        """Check if the parent process is alive. If it is dead, exit the worker process."""
        parent_proc = multiprocessing.parent_process()
        parent_pid = os.getppid()
        if parent_proc is None:
            print("No parent process, exiting", file=sys.stderr)
            os._exit(-1)
        while not evt_exit.wait(1):
            if parent_proc.exitcode is not None or os.getppid() != parent_pid:
                print("Parent process died, exiting", file=sys.stderr)
                os._exit(-1)

    def _worker_run(
        self,
        cmd_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
    ) -> None:
        gc_init_worker(self._rank_worker_id)
        worker_exit_evt = threading.Event()
        parent_check_thread = threading.Thread(
            target=self._check_parent_process, args=(worker_exit_evt,), daemon=True
        )
        parent_check_thread.start()
        try:
            super()._worker_run(cmd_queue, result_queue)
        finally:
            worker_exit_evt.set()
            parent_check_thread.join()
            cmd_queue.cancel_join_thread()
            cmd_queue.close()
            result_queue.cancel_join_thread()
            result_queue.close()

    @override
    def start(self) -> None:
        self._process = multiprocessing.Process(
            target=self._worker_run,
            args=(self._cmd_queue, self._result_queue),
        )
        self._process.start()

    @override
    def shutdown(self) -> None:
        if self._spawning_process != os.getpid():
            # Should avoid forked process containing a forked worker on exit.
            warnings.warn(
                "Shutting down worker from a different process than the one that spawned it, skipping"
            )
            return
        if self._process is not None:
            self._wrk_shutdown_worker()
            self._process.join()
            self._cmd_queue.cancel_join_thread()
            self._cmd_queue.close()
            self._result_queue.cancel_join_thread()
            self._result_queue.close()

    @override
    def running(self) -> bool:
        return self._process is not None

    def _assert_running(self) -> None:
        assert self._process is not None, "Worker must be started first"
        assert self._process.is_alive(), "Worker died"


class ThreadDataLoaderWorker(_DataLoaderAsynchronousWorker[TSample], Generic[TSample]):
    """
    Implements the `DataLoaderWorker` interface using threads.
    """

    _thread: threading.Thread | None = None
    _cmd_queue: queue.Queue
    _result_queue: queue.Queue

    def __init__(
        self,
        dataset: SavableDataset,
        worker_config: WorkerConfig,
        rank_worker_id: int,
        cache_pool: CachePool | None,
    ):
        super().__init__(
            dataset,
            worker_config=worker_config,
            rank_worker_id=rank_worker_id,
            cmd_queue=queue.Queue(),
            result_queue=queue.Queue(),
            cache_pool=cache_pool,
        )

    def _worker_run(self, cmd_queue: queue.Queue, result_queue: queue.Queue) -> None:
        super()._worker_run(cmd_queue, result_queue)

    @override
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._worker_run,
            args=(self._cmd_queue, self._result_queue),
        )
        self._thread.start()

    @override
    def shutdown(self) -> None:
        if self._thread is not None:
            self._wrk_shutdown_worker()
            self._thread.join()
            self._thread = None

    @override
    def running(self) -> bool:
        return self._thread is not None


class WorkerType(Protocol[TSample]):
    def __call__(
        self,
        dataset: SavableDataset,
        worker_config: WorkerConfig,
        rank_worker_id: int,
        cache_pool: CachePool | None,
    ) -> DataLoaderWorker[TSample]: ...


class DataLoader(Generic[TSample]):
    _workers: list[DataLoaderWorker[TSample]] | None = None
    _exhausted_workers: list[bool]
    _next_worker_id: int = 0

    _restore_state: FlexState | None = None

    _dataset: SavableDataset
    _worker_config: WorkerConfig
    _prefetch_factor: int
    _worker_type: WorkerType
    _prefetching_samples: list[list[Future[TSample]]]

    _current_epoch_iter: Generator[TSample, None, None] | None = None

    _spawning_process: int

    def __init__(
        self,
        dataset: SavableDataset,
        *,
        prefetch_factor: int = 2,
        worker_type: WorkerType = ForkDataLoaderWorker,
        cache_pool: CachePool | None = None,
        # Garbage collection configuration
        gc_collect_every_n_steps: int = GC_DEFAULT_EVERY_N_ITER,
        gc_freeze_at_start: bool = True,
        # Watchdog configuration
        watchdog_timeout_seconds: float | None = 60,
        watchdog_initial_timeout_seconds: float | None = None,
        fail_on_timeout: bool = False,
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
            watchdog_timeout_seconds: The timeout in seconds. If None, the watchdog is disabled.
            watchdog_initial_timeout_seconds: The initial timeout in seconds. If None, the timeout is the same as watchdog_timeout_seconds.
            fail_on_timeout: If True, stops the whole process upon timeout, after printing a stack trace.
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
        self._prefetching_samples = [[] for _ in range(self._worker_config.num_workers)]
        self._exhausted_workers = [False] * self._worker_config.num_workers

        if self._worker_config.num_workers == 0:
            assert prefetch_factor == 1, "prefetch_factor must be 1 for num_workers == 0"
        else:
            assert prefetch_factor > 0, "prefetch_factor must be > 0 for num_workers > 0"

        self._spawning_process = os.getpid()

    def shutdown(self) -> None:
        if self._workers is not None:
            for worker in self._workers:
                worker.shutdown()
            self._workers = None

    def start_iter(self) -> None:
        if self._workers is not None:
            for worker in self._workers:
                worker.new_iter()

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
            self._exhausted_workers = [False] * self._worker_config.num_workers

        # For all workers, enqueue prefetching samples.
        for worker_idx, (worker, exhausted) in enumerate(
            zip(self._workers, self._exhausted_workers)
        ):
            while (
                len(self._prefetching_samples[worker_idx]) < self._prefetch_factor and not exhausted
            ):
                self._prefetching_samples[worker_idx].append(worker.prefetch_next())

        # Main loop:
        # - Get the next worker to prefetch samples from.
        # - Prefetch samples from the worker.
        # - Pop the first sample future from the prefetching samples.
        # - Get the sample from the sample future (may wait for the sample to be prefetched).
        # - Yield the sample.
        while not all(self._exhausted_workers):
            # Get the next worker to prefetch samples from.
            worker_idx = self._next_worker_id
            worker = self._workers[worker_idx]
            self._next_worker_id = (worker_idx + 1) % self._worker_config.num_workers
            if self._exhausted_workers[worker_idx]:
                continue
            # Pop the first sample future from the prefetching samples.
            sample_future = self._prefetching_samples[worker_idx].pop(0)
            # Prefetch samples from the worker.
            while len(self._prefetching_samples[worker_idx]) < self._prefetch_factor:
                # Add a new sample future to the prefetching samples if the worker has not prefetched enough samples.
                self._prefetching_samples[worker_idx].append(worker.prefetch_next())
            try:
                # Get the sample from the sample future (may wait for the sample to be ready).
                sample = sample_future.get()
            except StopIteration:
                # If the sample future raises StopIteration, remove the worker from the list.
                self._prefetching_samples[worker_idx] = []
                self._exhausted_workers[worker_idx] = True
                continue
            else:
                # Yield the sample.
                yield sample

    def __iter__(self) -> Generator[TSample, None, None]:
        # Restart the epoch iterator if was not created yet. Otherwise, the existing epoch iterator will be continued.
        # That happens e.g. when iteration was interrupted.
        if self._current_epoch_iter is None:
            self._current_epoch_iter = self._epoch_iter()
        yield from self._current_epoch_iter
        # Reset the epoch iterator, it was exhausted.
        self._current_epoch_iter = None

    def __del__(self) -> None:
        if self._spawning_process == os.getpid():
            # Otherwise we may be in a forked process which is not the one that spawned the DataLoader.
            self.shutdown()

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

    def save_state(self) -> FlexState:
        # TODO: The redist tool must be able to change the batch size.
        # That means that the redist tool shall split a saved restore key for the "BatchDataset".
        # It should also change the saved micro batch size to match that.
        # TODO @pfischer: Add changing the batch size to the docs.
        prefetched_samples_keys = [
            [get_sample_restore_key(sample_fut.get()) for sample_fut in prefetching_sample]
            for prefetching_sample in self._prefetching_samples
        ]
        if self._workers is None:
            worker_states = [None] * self._worker_config.num_workers
        else:
            worker_states = [worker.save_state() for worker in self._workers]

        return FlexState(
            __class__=type(self).__name__,
            prefetched_samples_keys=prefetched_samples_keys,
            worker_states=worker_states,
            next_worker_id=self._next_worker_id,
            micro_batch_size=self._get_batch_size(),
        )

    def _start(self, initial_state: FlexState | None = None) -> None:
        assert self._workers is None and self._current_epoch_iter is None, (
            "DataLoader already started"
        )
        self._workers = [
            self._worker_type(self._dataset, self._worker_config, local_worker_id, self._cache_pool)
            for local_worker_id in range(max(self._worker_config.num_workers, 1))
        ]
        for worker in self._workers:
            worker.start()

        if initial_state is None:
            if self._restore_state is not None:
                initial_state = self._restore_state
                self._restore_state = None

        if initial_state is None:
            worker_states = [None] * self._worker_config.num_workers
        else:
            worker_states = initial_state["worker_states"]

        assert len(worker_states) == self._worker_config.num_workers, (
            "Number of initial states must match number of workers"
        )

        for worker, worker_state in zip(self._workers, worker_states):
            worker.dataset_init(worker_state)

        if initial_state is not None:
            self._prefetching_samples = [
                [
                    CallableFuture(functools.partial(self.restore_sample, sample_key))
                    for sample_key in prefetched_samples_keys
                ]
                for prefetched_samples_keys in initial_state["prefetched_samples_keys"]
            ]
            self._next_worker_id = initial_state["next_worker_id"]
            self._exhausted_workers = [
                False if worker_state is None else worker_state["exhausted"]
                for worker_state in worker_states
            ]

    def restore_state_rank(self, state: FlexState | None) -> None:
        """
        Restore the state of the DataLoader on the current rank.
        The state is actually restored when the processes are started, in the iterator.
        """
        assert self._workers is None and self._current_epoch_iter is None, (
            "DataLoader already started"
        )
        assert self._restore_state is None, "Restore state already set"

        if state is None:
            # Assume initial state.
            return

        assert isinstance(state, FlexState)
        assert state["__class__"] == type(self).__name__, "DataLoader type mismatch"
        assert state["micro_batch_size"] == self._get_batch_size(), "Micro batch size mismatch"

        self._restore_state = state

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

    def config(self) -> dict[str, Any]:
        return self._dataset.config()

    def __str__(self) -> str:
        return f"DataLoader(prefetch_factor={self._prefetch_factor}, worker_type={self._worker_type.__name__})"
