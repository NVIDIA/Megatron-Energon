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
    cast,
    override,
)

from megatron.energon.cache.base import CachePool
from megatron.energon.cache.no_cache import NoCachePool
from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import SavableDataset, add_sample_restore_key
from megatron.energon.rng import SystemRng
from megatron.energon.state import FlexState
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset, SampleIndex, get_sample_restore_key
from megatron.energon.wrappers.batch_dataset import BatchDataset
from megatron.energon.wrappers.gc_dataset import gc_init_worker

P = ParamSpec("P")
R = TypeVar("R", covariant=True)
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
    _cache_pool: CachePool

    exhausted: bool = True

    def __init__(
        self,
        dataset: SavableDataset[TSample],
        worker_config: WorkerConfig,
        rank_worker_id: int,
        cache_pool: CachePool,
    ):
        self.dataset = dataset
        self.worker_config = worker_config
        self._rank_worker_id = rank_worker_id
        self._global_worker_id = worker_config.global_worker_id(rank_worker_id)
        self._cache_pool = cache_pool

    # ------------------------------------------------------------------------------------------------
    # Section: Main control methods

    def start(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def running(self) -> bool:
        return True

    def _assert_running(self) -> None:
        assert self.running(), "Worker must be running"

    # ------------------------------------------------------------------------------------------------
    # Section: Worker methods

    def dataset_init(self, initial_state: FlexState | None) -> None:
        self._sample_index = SampleIndex(worker_config=self.worker_config, src=self)
        self._global_worker_id = self.worker_config.global_worker_id()
        if initial_state is None:
            self.dataset.reset_state_deep()
        else:
            assert initial_state["__class__"] == "DataLoaderWorker", "Worker type mismatch"
            self._sample_index.restore_state(initial_state["_sample_index"])
            self.dataset.restore_state(initial_state["datasets"][0])
            # TODO: exhausted

    def new_iter(self) -> None:
        self._dataset_iter = iter(self.dataset)
        self.exhausted = False

    def prefetch_next(self) -> Future[TSample]:
        assert self._dataset_iter is not None, "start_iter must be called before prefetch_next"
        with self._sample_index.ctx() as sample_idx:
            self.worker_config.worker_activate(sample_idx, cache_pool=self._cache_pool)
            try:
                next_sample = next(self._dataset_iter)
                add_sample_restore_key(next_sample, self._global_worker_id, sample_idx, src=self)
            except StopIteration as e:
                self.exhausted = True
                return ExceptionFuture(e)
            finally:
                self.worker_config.worker_deactivate()
        return DoneFuture(next_sample)

    def save_state(self) -> FlexState:
        return FlexState(
            __class__="DataLoaderWorker",
            rng=SystemRng.save_state(),
            dataset=self.dataset.save_state(),
            exhausted=self.exhausted,
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
        cache_pool: CachePool,
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

    @staticmethod
    def worker_call(fn: Callable[P, R]) -> Callable[P, R]:
        """Make the function be called in the worker process via the command and result queues.
        The function must be a method of the `DataLoaderWorker` class."""

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs) -> R:
            future = self._worker_call(fn.__name__, *args, **kwargs)
            return future.get()

        setattr(wrapper, "_orig", fn)
        return cast(Callable[P, R], wrapper)

    @staticmethod
    def worker_call_future(fn: Callable[P, R]) -> Callable[P, Future[R]]:
        """Make the function be called in the worker process via the command and result queues.
        The function must be a method of the `DataLoaderWorker` class."""

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs) -> Future[R]:
            return self._worker_call(fn.__name__, *args, **kwargs)

        setattr(wrapper, "_orig", fn)
        return cast(Callable[P, Future[R]], wrapper)

    def _wait_for_worker_result(self, future_id: int) -> None:
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

    def _worker_call(self, fn: str, *args: Any, **kwargs: Any) -> Future[Any]:
        self._assert_running()
        future_id = self._next_future_id
        self._next_future_id += 1

        self._futures[future_id] = self.FutureImpl(self, future_id)
        self._cmd_queue.put(
            self.WorkerCommand(cmd=fn, args=args, kwargs=kwargs, future_id=future_id)
        )
        return self._futures[future_id]

    def _worker_run(
        self, cmd_queue: QueueProtocol[WorkerCommand], result_queue: QueueProtocol[WorkerResult], seed: int
    ) -> None:
        SystemRng.seed(seed)
        import torch.utils.data._utils

        torch.utils.data._utils.worker._worker_info = torch.utils.data._utils.worker.WorkerInfo(
            id=self._rank_worker_id,
            num_workers=self.worker_config.num_workers,
            seed=seed,
            dataset=self.dataset,
        )
        self._global_worker_id = self.worker_config.global_worker_id()
        self.worker_config.assert_worker()
        while True:
            cmd = cmd_queue.get()
            try:
                fn = getattr(self, cmd.cmd)
                result = getattr(fn, "_orig")(self, *cmd.args, **cmd.kwargs)
            except Exception as e:
                result_queue.put(self.WorkerResult(future_id=cmd.future_id, exception=e))
            else:
                result_queue.put(self.WorkerResult(future_id=cmd.future_id, result=result))
                del result
            if cmd.cmd == "_shutdown_worker":
                break

    # ------------------------------------------------------------------------------------------------
    # Section: Worker methods - now calling to workers via queues.

    @worker_call
    def _shutdown_worker(self) -> None:
        """Shutdown the worker. The actual shutdown is handled in the _worker_run method."""
        pass

    @override
    @worker_call
    def dataset_init(self, initial_state: FlexState | None) -> None:
        super().dataset_init(initial_state)

    @override
    @worker_call
    def new_iter(self) -> None:
        super().new_iter()

    @override
    @worker_call_future
    def prefetch_next(self) -> TSample:
        # The super class implementation already returns a resolved future (to be interface compatible),
        # so immediately resolve the future to the result (get returns immediately).
        # The worker_call_future will wrap the result again in a future implicitly.
        return super().prefetch_next().get()

    @override
    @worker_call
    def save_state(self) -> FlexState:
        return super().save_state()


class ForkDataLoaderWorker(_DataLoaderAsynchronousWorker[TSample], Generic[TSample]):
    """
    Implements the `DataLoaderWorker` interface using processes.
    """

    _process: multiprocessing.Process | None = None

    _spawning_process: int

    def __init__(
        self,
        dataset: SavableDataset,
        worker_config: WorkerConfig,
        rank_worker_id: int,
        cache_pool: CachePool,
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
        """Check if the parent process is alive. If it is not, exit the worker process."""
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
        seed: int,
    ) -> None:
        gc_init_worker(self._rank_worker_id)
        worker_exit_evt = threading.Event()
        parent_check_thread = threading.Thread(
            target=self._check_parent_process, args=(worker_exit_evt,), daemon=True
        )
        parent_check_thread.start()
        try:
            super()._worker_run(cmd_queue, result_queue, seed)
        finally:
            worker_exit_evt.set()
            parent_check_thread.join()
            cmd_queue.cancel_join_thread()
            cmd_queue.close()
            result_queue.cancel_join_thread()
            result_queue.close()

    @override
    def start(self) -> None:
        # TODO: seed per worker
        self._process = multiprocessing.Process(
            target=self._worker_run,
            args=(self._cmd_queue, self._result_queue, seed),
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
            self._shutdown_worker()
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

    def __init__(
        self,
        dataset: SavableDataset,
        worker_config: WorkerConfig,
        rank_worker_id: int,
        cache_pool: CachePool,
    ):
        super().__init__(
            dataset,
            worker_config=worker_config,
            rank_worker_id=rank_worker_id,
            cmd_queue=queue.Queue(),
            result_queue=queue.Queue(),
            cache_pool=cache_pool,
        )

    def _worker_run(
        self, cmd_queue: queue.Queue, result_queue: queue.Queue, seed: int
    ) -> None:
        # TODO: Implement init_thread which should hook all randomness such that it's thread local.
        SystemRng.init_thread()
        super()._worker_run(cmd_queue, result_queue, seed)

    @override
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._worker_run,
            args=(self._rank_worker_id, self._cmd_queue, self._result_queue),
        )
        self._thread.start()

    @override
    def shutdown(self) -> None:
        if self._thread is not None:
            self._shutdown_worker()
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
        cache_pool: CachePool,
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
        prefetch_factor: int = 2,
        worker_type: WorkerType = ForkDataLoaderWorker,
        cache_pool: CachePool = NoCachePool(),
    ):
        if dataset.worker_config.num_workers == 0 and worker_type == ForkDataLoaderWorker:
            worker_type = DataLoaderWorker
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

        # TODO: Seed per worker from SavableDataLoader

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
            for worker, exhausted in zip(self._workers, self._exhausted_workers):
                if not exhausted:
                    worker.new_iter()

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
            workers_exhausted=self._exhausted_workers.copy(),
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
            initial_states = [None] * self._worker_config.num_workers
        else:
            initial_states = initial_state["worker_states"]

        assert len(initial_states) == self._worker_config.num_workers, (
            "Number of initial states must match number of workers"
        )

        for worker, initial_state in zip(self._workers, initial_states):
            worker.dataset_init(initial_state)

        if initial_state is not None:
            self._prefetching_samples = [
                [
                    # TODO: Use a callback future
                    DoneFuture(self.restore_sample(sample_key))
                    for sample_key in prefetched_samples_keys
                ]
                for prefetched_samples_keys in initial_state["prefetched_samples_keys"]
            ]
            self._next_worker_id = initial_state["next_worker_id"]
            self._exhausted_workers = initial_state["workers_exhausted"].copy()

    def restore_state_rank(self, state: FlexState | None) -> None:
        assert self._workers is None and self._current_epoch_iter is None, (
            "DataLoader already started"
        )

        if state is None:
            # Assume initial state.
            return

        assert isinstance(state, FlexState)
        assert state["__class__"] == type(self).__name__, "DataLoader type mismatch"

        old_micro_batch_size = state["micro_batch_size"]
        micro_batch_size = self._get_batch_size()

        if self._worker_config.num_workers == 0:
            assert micro_batch_size == old_micro_batch_size, "Micro batch size mismatch"
            assert len(state["worker_states"]) == 1
            assert isinstance(state["worker_states"][0], FlexState)
            self._dataset.restore_state(state["worker_states"][0])
        else:
            # Check batch sizes (before and after)
            if micro_batch_size != old_micro_batch_size:
                assert micro_batch_size is not None and old_micro_batch_size is not None, (
                    "Cannot resume with different batching mode "
                    "(batching to non-batching or vice versa)"
                )

                if micro_batch_size > old_micro_batch_size:
                    raise ValueError(
                        "Resuming with larger micro batch size is not allowed: "
                        f"{micro_batch_size} > {old_micro_batch_size}"
                    )
                elif (
                    micro_batch_size < old_micro_batch_size
                    and old_micro_batch_size % micro_batch_size != 0
                ):
                    raise ValueError(
                        "Resuming with smaller micro batch size only allowed if the old "
                        f"micro batch size is a multiple of the new one: {micro_batch_size} < {old_micro_batch_size}"
                    )

        self._restore_state = state

    def restore_sample(self, restore_key: tuple) -> TSample:
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
