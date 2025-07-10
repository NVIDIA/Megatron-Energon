# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import multiprocessing
import multiprocessing.managers
import os
import queue
import sys
import threading
import traceback
import warnings
from abc import abstractmethod
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
from megatron.energon.wrappers.base import BaseWrapperDataset, get_sample_restore_key
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
    _sample_index: int = 0
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

    def shutdown(self, in_del: bool = False) -> None:
        """
        Shutdown the worker.

        Args:
            in_del: If True, the worker is being deleted.
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

    def __del__(self) -> None:
        self.shutdown(in_del=True)

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
        assert self._global_worker_id == self.worker_config.global_worker_id(), (
            "Global worker ID mismatch"
        )
        assert self._seed == self.worker_config.worker_seed(self._rank_worker_id), "Seed mismatch"
        print(f"dataset_init {state=}\n", end="")
        if state is None:
            self._sample_index = 0
            self.dataset.reset_state_deep()
            print("dataset_init reset_state_deep\n", end="")
            self.new_iter()
            print("dataset_init new_iter\n", end="")
        else:
            assert state["__class__"] == "DataLoaderWorker", "state type mismatch"
            self._sample_index = state["sample_index"]
            SystemRng.restore_state(state["rng"])
            self.dataset.restore_state(state["dataset"])
            if not state["exhausted"]:
                self.new_iter()
            assert self._exhausted == state["exhausted"], "Exhausted state mismatch"

    def new_iter(self) -> None:
        """
        Start a new iterator of the dataset.
        Called after the dataset is initialized and to start a new epoch (if the dataset is not infinite).
        The iterator is stored in the worker and is used by the `prefetch_next` method, which calls `next` on it.
        Updates the exhausted flag to False.
        """
        # This is called in the worker context (process/thread).
        print("new_iter\n", end="")
        self._dataset_iter = iter(self.dataset)
        self._exhausted = False
        print("new_iter done\n", end="")

    def prefetch_next(self) -> Future[TSample]:
        """
        Fetch the next sample (i.e. call `next` on the iterator) and return a future for getting the result.
        Updates the exhausted flag if the iterator is exhausted.

        Returns:
            A future that will either be resolved to the next sample or raise StopIteration if the iterator is exhausted.
        """
        # This is called in the worker context (process/thread).
        assert self._dataset_iter is not None, "start_iter must be called before prefetch_next"
        if self._exhausted:
            try:
                raise StopIteration()
            except StopIteration as e:
                return ExceptionFuture(e)
        sample_idx = self._sample_index
        self.worker_config.worker_activate(sample_idx, cache_pool=self._cache_pool)
        try:
            next_sample = next(self._dataset_iter)
            self._sample_index += 1
            next_sample = add_sample_restore_key(
                next_sample, self._global_worker_id, sample_idx, src=self
            )
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
            sample_index=self._sample_index,
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
            print(f"[fut={future_id}] waiting for result\n", end="")
            res = self._result_queue.get()
            fut = self._futures.pop(res.future_id)
            if res.exception is not None:
                fut._set_exception(res.exception)
            else:
                fut._set_result(res.result)
            # self._result_queue.task_done()
            if res.future_id == future_id:
                print(f"[fut={future_id}] got result, return\n", end="")
                return
            else:
                print(f"[fut={future_id}] got result for {res.future_id=}, continue\n", end="")
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
        assert not self._in_worker(), "worker_call must not be called in the worker"
        future_id = self._next_future_id
        self._next_future_id += 1

        self._futures[future_id] = future = self.FutureImpl(self, future_id)
        print(
            f"[wrk={self._rank_worker_id}] worker_call {fn.__name__=} {args=} {kwargs=} {future_id=}\n",
            end="",
        )
        self._cmd_queue.put(
            self.WorkerCommand(cmd=fn.__name__, args=args, kwargs=kwargs, future_id=future_id)
        )
        print(f"[wrk={self._rank_worker_id}] queue: {self._cmd_queue.qsize()}\n", end="")
        return future

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
        assert self._in_worker(), "_worker_run must be called in the worker"
        try:
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
                print(
                    f"[wrk={self._rank_worker_id}] waiting for command, len: {cmd_queue.qsize()}\n",
                    end="",
                )
                cmd = cmd_queue.get()
                print(
                    f"[fut={cmd.future_id}] got command {cmd.cmd=} {cmd.args=} {cmd.kwargs=}\n",
                    end="",
                )
                try:
                    fn = getattr(self, cmd.cmd)
                    result = fn(*cmd.args, **cmd.kwargs)
                except Exception as e:
                    print(f"[fut={cmd.future_id}] send exception {e!r}\n", end="")
                    result_queue.put(self.WorkerResult(future_id=cmd.future_id, exception=e))
                else:
                    print(f"[fut={cmd.future_id}] send result {result!r}\n", end="")
                    result_queue.put(self.WorkerResult(future_id=cmd.future_id, result=result))
                    del result
                # cmd_queue.task_done()
                if cmd.cmd == self._wrk_shutdown_worker.__name__:
                    print(f"[fut={cmd.future_id}] got shutdown command, exit\n", end="")
                    break
                print(f"[fut={cmd.future_id}] processed, waiting for next command\n", end="")
        except:
            traceback.print_exc()
            raise

    @abstractmethod
    def _in_worker(self) -> bool:
        """Check if the execution is within the worker."""
        ...

    # ------------------------------------------------------------------------------------------------
    # Section: Worker methods - now calling to workers via queues.

    def _wrk_shutdown_worker(self) -> None:
        """Does nothing. The actual shutdown is handled in the _worker_run method."""
        assert self._in_worker(), "_wrk_shutdown_worker must be called in the worker"

    def _shutdown_worker(self) -> None:
        """Shutdown the worker. The actual shutdown is handled in the _worker_run method."""
        assert not self._in_worker(), "shutdown_worker must not be called in the worker"
        # This is not actually a recursive call, because the worker loop will exit before calling this method.
        self._worker_call(self._wrk_shutdown_worker).get()

    def _wrk_prefetch_next(self) -> TSample:
        """Wraps the super class method to call it in the worker process."""
        # The super class implementation already returns a resolved future (to be interface compatible),
        # so immediately resolve the future to the result (get returns immediately).
        return super().prefetch_next().get()

    @override
    def dataset_init(self, initial_state: FlexState | None) -> None:
        if self._in_worker():
            return super().dataset_init(initial_state)
        else:
            return self._worker_call(self.dataset_init, initial_state).get()

    @override
    def new_iter(self) -> None:
        if self._in_worker():
            return super().new_iter()
        else:
            return self._worker_call(self.new_iter).get()

    @override
    def prefetch_next(self) -> Future[TSample]:
        # Do not resolve the future here, but return it.
        if self._in_worker():
            return super().prefetch_next()
        return self._worker_call(self._wrk_prefetch_next)

    @override
    def save_state(self) -> FlexState:
        if self._in_worker():
            return super().save_state()
        else:
            return self._worker_call(self.save_state).get()


class ForkDataLoaderWorker(_DataLoaderAsynchronousWorker[TSample], Generic[TSample]):
    """
    Implements the `DataLoaderWorker` interface using processes.
    """

    _process: multiprocessing.Process | None = None
    _cmd_queue: multiprocessing.Queue
    _result_queue: multiprocessing.Queue

    _threaded_shutdown: threading.Thread | None = None

    _spawning_process: int

    def __init__(
        self,
        dataset: SavableDataset,
        worker_config: WorkerConfig,
        rank_worker_id: int,
        cache_pool: CachePool | None,
    ):
        multiprocessing.set_start_method("fork", force=True)
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
        # cmd_queue is read only, so we can cancel the join thread.
        cmd_queue.cancel_join_thread()
        worker_exit_evt = threading.Event()
        parent_check_thread = threading.Thread(
            target=self._check_parent_process, args=(worker_exit_evt,), daemon=True
        )
        parent_check_thread.start()
        try:
            super()._worker_run(cmd_queue, result_queue)
        finally:
            print(f"[wrk={self._rank_worker_id}] shutting down\n", end="")
            worker_exit_evt.set()
            print(
                f"[wrk={self._rank_worker_id}] shutting down, wait for parent_check_thread\n",
                end="",
            )
            parent_check_thread.join()
            print(f"[wrk={self._rank_worker_id}] shutting down, close queues\n", end="")
            result_queue.close()
            result_queue.join_thread()
            cmd_queue.close()
            cmd_queue.cancel_join_thread()
            print(f"[wrk={self._rank_worker_id}] shutting down, done\n", end="")

    @override
    def _in_worker(self) -> bool:
        return multiprocessing.current_process() == self._process

    @override
    def start(self) -> None:
        self._process = multiprocessing.Process(
            target=self._worker_run,
            args=(self._cmd_queue, self._result_queue),
            daemon=True,
        )
        self._process.start()

    @override
    def shutdown(self, in_del: bool = False) -> None:
        if self._spawning_process != os.getpid():
            # Should avoid forked process containing a forked worker on exit.
            warnings.warn(
                "Shutting down worker from a different process than the one that spawned it, skipping"
            )
            return
        if self._process is not None:
            if in_del:
                # It seems that the ResourceWarning does not work in the gc loop? Also print a warning here.
                warnings.warn(
                    "Explicitly call DataLoader.shutdown() to avoid leaking processes. Terminating worker process.",
                    ResourceWarning,
                )
                print(
                    "WARNING: Explicitly call DataLoader.shutdown() to avoid leaking processes. Terminating worker process.\n",
                    end="",
                    file=sys.stderr,
                )
                self._cmd_queue.close()
                self._cmd_queue.cancel_join_thread()
                self._result_queue.close()
                self._result_queue.cancel_join_thread()
                # Kill the process, because we cannot communicate with it in the gc loop.
                self._process.terminate()
                self._process = None
            else:
                try:
                    self._shutdown_worker()
                except Exception:
                    self._process.join(10)
                    if self._process.is_alive():
                        self._process.terminate()
                else:
                    self._process.join()
                    assert self._process.exitcode == 0, (
                        f"Process exit code {self._process.exitcode}"
                    )
                self._process = None
                self._cmd_queue.close()
                self._cmd_queue.cancel_join_thread()
                self._result_queue.close()
                self._result_queue.cancel_join_thread()

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
    def _in_worker(self) -> bool:
        return threading.current_thread() == self._thread

    @override
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._worker_run,
            args=(self._cmd_queue, self._result_queue),
            daemon=True,
        )
        self._thread.start()

    @override
    def shutdown(self, in_del: bool = False) -> None:
        if self._thread is not None:
            if in_del:
                # It seems that the ResourceWarning does not work in the gc loop? Also print a warning here.
                warnings.warn(
                    "Explicitly call DataLoader.shutdown() to avoid leaking threads.",
                    ResourceWarning,
                )
                print(
                    "WARNING: Explicitly call DataLoader.shutdown() to avoid leaking threads.\n",
                    end="",
                    file=sys.stderr,
                )
                # Just try to enqueue the shutdown command to the thread and hope for the best. Ignore the result.
                self._cmd_queue.put(
                    self.WorkerCommand(
                        cmd=self._wrk_shutdown_worker.__name__, args=(), kwargs={}, future_id=-1
                    )
                )
                self._thread = None
            else:
                self._shutdown_worker()
                self._thread.join()
                self._thread = None

    @override
    def running(self) -> bool:
        return self._thread is not None

    @override
    def _assert_running(self) -> None:
        assert self._thread is not None, "Thread must be started first"
        assert self._thread.is_alive(), "Thread died"


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
        self._prefetching_samples = [[] for _ in range(self._worker_config.safe_num_workers)]
        self._exhausted_workers = [False] * self._worker_config.safe_num_workers

        if self._worker_config.num_workers == 0:
            assert prefetch_factor == 1, "prefetch_factor must be 1 for num_workers == 0"
        else:
            assert prefetch_factor > 0, "prefetch_factor must be > 0 for num_workers > 0"

        self._spawning_process = os.getpid()

    def shutdown(self, in_del: bool = False) -> None:
        if self._workers is not None:
            if in_del:
                warnings.warn(
                    "Explicitly call DataLoader.shutdown() to avoid leaking workers.",
                    ResourceWarning,
                )
                print(
                    "WARNING: Explicitly call DataLoader.shutdown() to avoid leaking workers.\n",
                    end="",
                    file=sys.stderr,
                )
            for worker in self._workers:
                worker.shutdown(in_del=in_del)
            self._workers = None

    def __del__(self) -> None:
        self.shutdown(in_del=True)

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
            self._exhausted_workers = [False] * self._worker_config.safe_num_workers

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
                self._prefetching_samples[worker_idx].append(worker.prefetch_next())
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
        yield from self._current_epoch_iter
        # Reset the epoch iterator, it was exhausted.
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

    def _start(self, initial_state: FlexState | None = None) -> None:
        self._workers = [
            self._worker_type(self._dataset, self._worker_config, local_worker_id, self._cache_pool)
            for local_worker_id in range(self._worker_config.safe_num_workers)
        ]
        for worker in self._workers:
            worker.start()

        if initial_state is None:
            if self._restore_state is not None:
                initial_state = self._restore_state
                self._restore_state = None

        if initial_state is None:
            worker_states = [None] * self._worker_config.safe_num_workers
        else:
            worker_states = initial_state["worker_states"]

        assert len(worker_states) == self._worker_config.safe_num_workers, (
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
