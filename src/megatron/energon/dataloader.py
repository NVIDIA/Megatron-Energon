from abc import abstractmethod
from concurrent.futures import Future
import functools
import multiprocessing
import os
import queue
import sys
import threading
from typing import Any, Concatenate, Generic, Protocol, Self, Sequence, TypeVar, cast
from typing import Callable, TypeVar, ParamSpec
import warnings
from torch.utils.data import IterableDataset


from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.metadataset.dataset_loader import DatasetLoader
from megatron.energon.rng import SystemRng
from megatron.energon.state import FlexState
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import SampleIndex
from megatron.energon.wrappers.gc_dataset import gc_init_worker


class QueueProtocol(Protocol):
    def get(self, /) -> Any:
        ...

    def put(self, item: Any, /) -> None:
        ...


P = ParamSpec('P')
R = TypeVar('R', covariant=True)
TSelf = TypeVar('TSelf', bound="DataLoaderWorker")


class Future(Protocol[R]):
    def get(self) -> R:
        ...

TSample = TypeVar('TSample')


class DataLoaderWorker(Generic[TSample]):
    """
    A worker for a :class:`DataLoader`.

    The worker is responsible for executing the commands sent by the main process and returning the results.
    It also handles the communication with the main process via the command and result queues.

    There are different implementations of a worker:
    - :class:`ForkDataLoaderWorker` - A worker that forks a new process for each worker.
    - :class:`ThreadDataLoaderWorker` - A worker that uses threads to execute the commands.
    """
    def __init__(self, dataset: SavableDataset, worker_config: WorkerConfig, worker_id: int, cmd_queue: QueueProtocol, result_queue: QueueProtocol, data_queue: QueueProtocol):
        self.dataset = dataset
        self.worker_config = worker_config
        self._worker_id = worker_id
        self._cmd_queue = cmd_queue
        self._result_queue = result_queue
        self._data_queue = data_queue
        self._next_future_id = 0
        self._futures = {}

    # ------------------------------------------------------------------------------------------------
    # Section: Remote call implementation

    class FutureImpl(Future):
        _outerself: "DataLoaderWorker"
        _future_id: int
        _result: Any

        def __init__(self, outerself: "DataLoaderWorker", future_id: int):
            self._outerself = outerself
            self._future_id = future_id

        def get(self) -> Any:
            if not hasattr(self, "_result"):
                self._outerself._wait_for_worker_result(self._future_id)
            if isinstance(self._result, Exception):
                raise self._result
            return self._result
        
        def _set_result(self, result: Any) -> None:
            self._result = result

    @edataclass
    class WorkerCommand:
        cmd: str
        args: tuple[Any, ...]
        kwargs: dict[str, Any]
        future_id: int
    
    @staticmethod
    def worker_call(fn: Callable[P, R]) -> Callable[P, R]:
        """Make the function be called in the worker process via the command and result queues.
        The function must be a method of the `DataLoaderWorker` class."""
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs) -> R:
            future = self._worker_call(fn.__name__, *args, **kwargs)
            return future.get()
        setattr(wrapper, '_orig', fn)
        return cast(Callable[P, R], wrapper)

    @staticmethod
    def worker_call_async(fn: Callable[P, R]) -> Callable[P, Future[R]]:
        """Make the function be called in the worker process via the command and result queues.
        The function must be a method of the `DataLoaderWorker` class."""
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs) -> Future[R]:
            return self._worker_call(fn.__name__, *args, **kwargs)
        setattr(wrapper, '_orig', fn)
        return cast(Callable[P, Future[R]], wrapper)

    def _wait_for_worker_result(self, future_id: int) -> None:
        while True:
            future_id, res = self._result_queue.get()
            fut = self._futures.pop(future_id)
            fut._set_result(res)
            if future_id == future_id:
                return

    def _worker_call(self, fn: str, *args: Any, **kwargs: Any) -> Future[Any]:
        self._assert_running()
        future_id = self._next_future_id
        self._next_future_id += 1

        self._futures[future_id] = self.FutureImpl(self, future_id)
        self._cmd_queue.put(self.WorkerCommand(cmd=fn, args=args, kwargs=kwargs, future_id=future_id))
        return self._futures[future_id]

    def _worker_run(self, worker_id: int, cmd_queue: QueueProtocol, result_queue: QueueProtocol, data_queue: QueueProtocol, seed: int) -> None:
        SystemRng.seed(seed)
        self._worker_id = worker_id
        self._data_queue = data_queue
        while True:
            cmd: DataLoaderWorker.WorkerCommand | None = cmd_queue.get()
            if cmd is None:
                break
            try:
                fn = getattr(self, cmd.cmd)
                result = getattr(fn, '_orig')(self, *cmd.args, **cmd.kwargs)
                result_queue.put((cmd.future_id, result))
                del result
            except Exception as e:
                result_queue.put((cmd.future_id, e))

    # ------------------------------------------------------------------------------------------------
    # Section: Main control methods

    @abstractmethod
    def start(self) -> None:
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        pass

    @abstractmethod
    def running(self) -> bool:
        pass
    
    def _assert_running(self) -> None:
        assert self.running(), "Worker must be running"

    # ------------------------------------------------------------------------------------------------
    # Section: Worker methods

    @worker_call
    def dataset_init(self, initial_state: FlexState | None) -> None:
        self._sample_index = SampleIndex(worker_config=self.worker_config, src=self)
        if initial_state is None:
            self.dataset.reset_state_deep()
        else:
            assert initial_state["__class__"] == type(self).__name__, "Worker type mismatch"
            self._sample_index.restore_state(initial_state["_sample_index"])
            self.dataset.restore_state(initial_state["datasets"][0])
    
    @worker_call
    def start_iter(self) -> None:
        self._dataset_iter = iter(self.dataset)

    @worker_call_async
    def prefetch_next(self) -> tuple[int, TSample]:
        assert self._dataset_iter is not None, "start_iter must be called before prefetch_next"
        with self._sample_index.ctx() as sample_idx:
            next_sample = next(self._dataset_iter)
        return sample_idx, next_sample

    @worker_call
    def save_state(self) -> FlexState:
        return FlexState(
            __class__=type(self).__name__,
            rng=SystemRng.save_state(),
            datasets=[self.dataset.save_state()],
            _sample_index=self._sample_index.save_state(),
        )


class ForkDataLoaderWorker(DataLoaderWorker[TSample], Generic[TSample]):
    _cmd_queue: multiprocessing.Queue
    _result_queue: multiprocessing.Queue
    _data_queue: multiprocessing.Queue
    _process: multiprocessing.Process | None

    def __init__(self, dataset: SavableDataset, num_workers: int, worker_id: int):
        super().__init__(dataset, num_workers=num_workers, worker_id=worker_id, cmd_queue=multiprocessing.Queue(), result_queue=multiprocessing.Queue(), data_queue=multiprocessing.Queue())
        self._spawning_process = multiprocessing.current_process()
    
    def _check_parent_process(self, evt_exit: threading.Event) -> None:
        """Check if the parent process is alive. If it is not, exit the worker process."""
        parent_proc = multiprocessing.parent_process()
        if parent_proc is None:
            print("No parent process, exiting", file=sys.stderr)
            os._exit(-1)
        while not evt_exit.wait(1):
            if parent_proc.exitcode is not None:
                print("Parent process died, exiting", file=sys.stderr)
                os._exit(-1)

    def _worker_run(self, worker_id: int, cmd_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue, data_queue: multiprocessing.Queue, seed: int) -> None:
        gc_init_worker(worker_id)
        worker_exit_evt = threading.Event()
        parent_check_thread = threading.Thread(target=self._check_parent_process, args=(worker_exit_evt,), daemon=True)
        parent_check_thread.start()
        try:
            super()._worker_run(worker_id, cmd_queue, result_queue, data_queue, seed)
        finally:
            worker_exit_evt.set()
            parent_check_thread.join()
            cmd_queue.cancel_join_thread()
            cmd_queue.close()
            result_queue.cancel_join_thread()
            result_queue.close()
            data_queue.cancel_join_thread()
            data_queue.close()
    
    def start(self) -> None:
        self._process = multiprocessing.Process(target=self._worker_run, args=(self._worker_id, self._cmd_queue, self._result_queue, self._data_queue))
        self._process.start()

    def shutdown(self) -> None:
        if self._spawning_process != multiprocessing.current_process():
            # Should avoid forked process containing a forked worker on exit.
            warnings.warn("Shutting down worker from a different process than the one that spawned it, skipping")
            return
        if self._process is not None:
            self._cmd_queue.put(None)
            self._process.join()
            self._cmd_queue.cancel_join_thread()
            self._cmd_queue.close()
            self._result_queue.cancel_join_thread()
            self._result_queue.close()
            self._data_queue.cancel_join_thread()
            self._data_queue.close()

    def running(self) -> bool:
        return self._process is not None

    def _assert_running(self) -> None:
        assert self._process is not None, "Worker must be started first"
        assert self._process.is_alive(), "Worker died"


class ThreadDataLoaderWorker(DataLoaderWorker[TSample], Generic[TSample]):
    def __init__(self, dataset: SavableDataset, num_workers: int, worker_id: int):
        super().__init__(dataset, num_workers=num_workers, worker_id=worker_id, cmd_queue=queue.Queue(), result_queue=queue.Queue(), data_queue=queue.Queue())
    
    def _worker_run(self, worker_id: int, cmd_queue: queue.Queue, result_queue: queue.Queue, data_queue: queue.Queue, seed: int) -> None:
        # TODO: Implement init_thread which should hook all randomness such that it's thread local.
        SystemRng.init_thread()
        super()._worker_run(worker_id, cmd_queue, result_queue, data_queue, seed)
    
    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker_run, args=(self._worker_id, self._cmd_queue, self._result_queue, self._data_queue))
        self._thread.start()
    
    def shutdown(self) -> None:
        if self._thread is not None:
            self._cmd_queue.put(None)
            self._thread.join()
            self._thread = None

    def running(self) -> bool:
        return self._thread is not None


class WorkerType(Protocol[TSample]):
    def __call__(self, dataset: SavableDataset, num_workers: int, worker_id: int) -> DataLoaderWorker[TSample]:
        ...


class DataLoader(Generic[TSample]):
    _workers: list[DataLoaderWorker[TSample]]

    _dataset: SavableDataset
    _num_workers: int
    _worker_type: WorkerType

    def __init__(self, dataset: SavableDataset, num_workers: int = 0, worker_type: WorkerType = ForkDataLoaderWorker):
        self._dataset = dataset
        self._num_workers = num_workers
        self._worker_type = worker_type
        self._workers = []
    
    def start(self, initial_states: Sequence[FlexState | None]) -> None:
        if self._num_workers == 0:
            return
        self._workers = [self._worker_type(self._dataset, self._num_workers, local_worker_id) for local_worker_id in range(self._num_workers)]
        for worker in self._workers:
            worker.start()

        for worker, initial_state in zip(self._workers, initial_states):
            worker.dataset_init(initial_state, 0)
        
    def shutdown(self) -> None:
        for worker in self._workers:
            worker.shutdown()
        self._workers = []

    def _get_iterator(self):
        if self._num_workers == 0:
            # Easy case: no workers, just iterate over the dataset.
            yield from self._dataset
            return
        
        for worker in self._workers:
            worker.start_iter()
        
        while True:
            for worker in self._workers:
                yield worker.prefetch_next()
    
    def __iter__(self):
        return self._get_iterator()
    
    def __del__(self) -> None:
        self.shutdown()

    def __len__(self):
        return len(self._dataset)
