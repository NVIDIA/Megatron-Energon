# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import traceback
from abc import abstractmethod
from typing import Any, Callable, Generic, ParamSpec, Protocol, TypeVar, override

from megatron.energon.cache.base import CachePool
from megatron.energon.dataloader.future import CancelledError, Future
from megatron.energon.dataloader.workers.base_worker import DataLoaderWorker
from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.rng import SystemRng
from megatron.energon.state import FlexState
from megatron.energon.worker import WorkerConfig

TSample = TypeVar("TSample", covariant=True)

P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R", covariant=True)


class QueueProtocol(Protocol[T]):
    def get(self, /) -> T: ...

    def put(self, item: T, /) -> None: ...

    def qsize(self, /) -> int: ...


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

    __slots__ = ("_worker", "_future_id", "_result", "_exception", "_cancelled")

    _worker: "DataLoaderAsynchronousWorker"
    _future_id: int
    _result: Any
    _exception: Exception
    _cancelled: bool

    def __init__(self, worker: "DataLoaderAsynchronousWorker", future_id: int):
        self._worker = worker
        self._future_id = future_id

    def get(self) -> Any:
        if getattr(self, "_cancelled", False):
            raise CancelledError()
        if not hasattr(self, "_result") and not hasattr(self, "_exception"):
            self._worker._wait_for_worker_result(self._future_id)
        if hasattr(self, "_exception"):
            raise self._exception
        return self._result

    def cancel(self) -> bool:
        if getattr(self, "_cancelled", False):
            return True
        if hasattr(self, "_result") or hasattr(self, "_exception"):
            return False
        # In case the main process is waiting for thie future to complete, add the result
        self._worker._result_queue.put(
            WorkerResult(
                future_id=self._future_id, exception=CancelledError.with_current_traceback()
            )
        )
        self._cancelled = True
        return True

    def _set_result(self, result: Any) -> None:
        self._result = result

    def _set_exception(self, exception: Exception) -> None:
        self._exception = exception


class AsynchronousMixin:
    """Mixin for asynchronous workers."""

    _cmd_queue: QueueProtocol[WorkerCommand]
    _result_queue: QueueProtocol[WorkerResult]
    _next_future_id: int
    _futures: dict[int, FutureImpl]
    _name: str

    def _asynchronous_init(self, name: str) -> None:
        self._cmd_queue, self._result_queue = self._queues()
        self._next_future_id = 0
        self._futures = {}
        self._name = name

    @abstractmethod
    def _queues(self) -> tuple[QueueProtocol[WorkerCommand], QueueProtocol[WorkerResult]]: ...

    def _wait_for_worker_result(self, future_id: int) -> None:
        """
        Wait for the result of a future.
        If another result comes first, update the corresponding future.

        Args:
            future_id: The ID of the future to wait for.
        """
        while True:
            print(f"[{self._name}, fut={future_id}] waiting for result\n", end="")
            res = self._result_queue.get()
            fut = self._futures.pop(res.future_id)
            if res.exception is not None:
                fut._set_exception(res.exception)
            else:
                fut._set_result(res.result)
            # self._result_queue.task_done()
            if res.future_id == future_id:
                print(f"[{self._name}, fut={future_id}] got result, return\n", end="")
                return
            else:
                print(
                    f"[{self._name}, fut={future_id}] got result for {res.future_id=}, continue\n",
                    end="",
                )
                continue

    def _cancel_futures(self) -> None:
        """Cancel all futures after worker shutdown."""
        for fut in self._futures.values():
            fut.cancel()
        self._futures.clear()

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

        self._futures[future_id] = future = FutureImpl(self, future_id)
        print(
            f"[{self._name}] worker_call {fn.__name__=} {args=} {kwargs=} {future_id=}\n",
            end="",
        )
        self._cmd_queue.put(
            WorkerCommand(cmd=fn.__name__, args=args, kwargs=kwargs, future_id=future_id)
        )
        print(f"[{self._name}] queue: {self._cmd_queue.qsize()}\n", end="")
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
            while True:
                print(
                    f"[{self._name}] waiting for command {cmd_queue.qsize()=}\n",
                    end="",
                )
                cmd = cmd_queue.get()
                print(
                    f"[{self._name}, fut={cmd.future_id}] got command {cmd.cmd=} {cmd.args=} {cmd.kwargs=}\n",
                    end="",
                )
                try:
                    fn = getattr(self, cmd.cmd)
                    result = fn(*cmd.args, **cmd.kwargs)
                except Exception as e:
                    print(f"[{self._name}, fut={cmd.future_id}] send exception {e!r}\n", end="")
                    result_queue.put(WorkerResult(future_id=cmd.future_id, exception=e))
                else:
                    print(f"[{self._name}, fut={cmd.future_id}] send result {result!r}\n", end="")
                    result_queue.put(WorkerResult(future_id=cmd.future_id, result=result))
                    del result
                # cmd_queue.task_done()
                if cmd.cmd == self._wrk_shutdown_worker.__name__:
                    print(
                        f"[{self._name}, fut={cmd.future_id}] got shutdown command, exit\n", end=""
                    )
                    break
                print(
                    f"[{self._name}, fut={cmd.future_id}] processed, waiting for next command\n",
                    end="",
                )
        except:
            traceback.print_exc()
            raise

    @abstractmethod
    def _assert_running(self) -> bool:
        """Check if the execution is within the worker."""
        ...

    @abstractmethod
    def _in_worker(self) -> bool:
        """Check if the execution is within the worker."""
        ...

    def _wrk_shutdown_worker(self) -> None:
        """Does nothing. The actual shutdown is handled in the _worker_run method."""
        assert self._in_worker(), "_wrk_shutdown_worker must be called in the worker"

    def _shutdown_worker(self) -> None:
        """Shutdown the worker. The actual shutdown is handled in the _worker_run method."""
        assert not self._in_worker(), "shutdown_worker must not be called in the worker"
        # This is not actually a recursive call, because the worker loop will exit before calling this method.
        self._worker_call(self._wrk_shutdown_worker).get()
        self._cancel_futures()
        print(f"[{self._name}] shutdown\n", end="")

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def running(self) -> bool: ...


class DataLoaderAsynchronousWorker(DataLoaderWorker[TSample], AsynchronousMixin, Generic[TSample]):
    """
    Extension of the `DataLoaderWorker`, which implements commands via a command and results queue.

    There are different implementations of the async worker:
    - :class:`ForkDataLoaderWorker` - A worker that forks a new process for each worker.
    - :class:`ThreadDataLoaderWorker` - A worker that uses threads to execute the commands.
    """

    def __init__(
        self,
        dataset: SavableDataset,
        worker_config: WorkerConfig,
        rank_worker_id: int,
        cache_pool: CachePool | None,
    ):
        super().__init__(dataset, worker_config, rank_worker_id, cache_pool)
        assert worker_config.num_workers > 0, "Async workers require num_workers > 0"
        self._asynchronous_init(name=f"wkr-{rank_worker_id}")

    # ------------------------------------------------------------------------------------------------
    # Section: Worker methods - now calling to workers via queues.

    def _worker_run(
        self, cmd_queue: QueueProtocol[WorkerCommand], result_queue: QueueProtocol[WorkerResult]
    ) -> None:
        SystemRng.seed(self._seed)
        import torch.utils.data._utils

        torch.utils.data._utils.worker._worker_info = torch.utils.data._utils.worker.WorkerInfo(
            id=self._rank_worker_id,
            num_workers=self.worker_config.num_workers,
            seed=self._seed,
            dataset=self.dataset,
        )
        self._global_worker_id = self.worker_config.global_worker_id()

        super()._worker_run(cmd_queue, result_queue)

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
