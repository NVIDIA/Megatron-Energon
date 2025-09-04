# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import threading
import traceback
from abc import abstractmethod
from typing import Any, Callable, ParamSpec, Protocol, TypeVar

import structlog

from megatron.energon.dataloader.future import CancelledError, Future
from megatron.energon.edataclass import edataclass
from megatron.energon.logging import get_logger

logger = get_logger(__name__)


P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R", covariant=True)


class QueueProtocol(Protocol[T]):
    """Protocol for a queue."""

    def get(self, /) -> T: ...

    def put(self, item: T, /) -> None: ...

    def qsize(self, /) -> int: ...


@edataclass
class WorkerCommand:
    """Internal class for communicating a command to the worker via the command queue."""

    cmd: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    future_id: int


@edataclass
class WorkerResult:
    """Internal class for communicating a result from the worker via the result queue."""

    future_id: int
    result: Any = None
    exception: Exception | None = None


class FutureImpl(Future[Any]):
    """Class for returning a future result from the worker.."""

    __slots__ = ("_worker", "_future_id", "_result", "_exception", "_cancelled")

    _worker: "Asynchronous"
    _future_id: int
    _result: Any
    _exception: Exception

    def __init__(self, worker: "Asynchronous", future_id: int):
        self._worker = worker
        self._future_id = future_id
        self.logger = logger.bind(self=self, worker_name=worker._name, future_id=future_id)

    def get(self) -> Any:
        if not hasattr(self, "_result") and not hasattr(self, "_exception"):
            self._worker._wait_for_worker_result(self)
        if hasattr(self, "_exception"):
            raise self._exception
        return self._result

    def cancel(self) -> bool:
        if hasattr(self, "_result") or hasattr(self, "_exception"):
            self.logger.debug("cancel: already has result or exception")
            return False
        self._exception = CancelledError.with_current_traceback()
        self._worker._cancel_future(self._future_id)
        return True

    def done(self) -> bool:
        return hasattr(self, "_result") or hasattr(self, "_exception")

    def _set_result(self, result: Any) -> None:
        self._result = result

    def _set_exception(self, exception: Exception) -> None:
        self._exception = exception

    def __str__(self) -> str:
        return f"FutureImpl(worker={self._worker._name!r}, future_id={self._future_id!r}, done={self.done()!r}, exception={getattr(self, '_exception', '<no exception>')})"


class Asynchronous:
    """Asynchronous base class."""

    _cmd_queue: QueueProtocol[WorkerCommand]
    _result_queue: QueueProtocol[WorkerResult]
    _next_future_id: int
    _pending_futures: dict[int, FutureImpl]
    _name: str
    _result_lock: threading.Lock

    logger: structlog.BoundLogger

    def _asynchronous_init(self, name: str) -> None:
        self._cmd_queue, self._result_queue = self._queues()
        self._next_future_id = 0
        self._pending_futures = {}
        self._name = name
        self._result_lock = threading.Lock()
        self.logger = logger.bind(self=self, name=name)

    @abstractmethod
    def _queues(self) -> tuple[QueueProtocol[WorkerCommand], QueueProtocol[WorkerResult]]: ...

    def _wait_for_worker_result(self, future: FutureImpl) -> None:
        """
        Wait for the result of a future.
        If another result comes first, update the corresponding future.

        Args:
            future: The future to wait for.
        """

        self.logger.debug("waiting for result", future=future)
        with self._result_lock:
            if future.done():
                # If calling get() from multiple threads, the future may be done now, because
                # the other thread already set the result.
                return
            self.logger.trace("got future", future=future)
            while True:
                res = self._result_queue.get()
                fut = self._pending_futures.pop(res.future_id)
                if res.exception is not None:
                    fut._set_exception(res.exception)
                else:
                    fut._set_result(res.result)
                if res.future_id == future._future_id:
                    self.logger.trace("got result, return", future=future)
                    return
                else:
                    self.logger.trace(f"got result for {res.future_id=}, continue", future=future)
                    continue

    def _cancel_future(self, future_id: int) -> None:
        """Cancel a future."""
        self.logger.debug("cancelling future", future_id=future_id)
        # In case the main process is waiting for thie future to complete, add the result
        self._result_queue.put(
            WorkerResult(future_id=future_id, exception=CancelledError.with_current_traceback())
        )

    def _cancel_futures(self) -> None:
        """Cancel all futures after worker shutdown."""
        for fut in self._pending_futures.values():
            fut.cancel()
        self._pending_futures.clear()

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

        self._pending_futures[future_id] = future = FutureImpl(self, future_id)
        self.logger.trace("worker_call", fn=fn.__name__, future_id=future_id)
        self._cmd_queue.put(
            WorkerCommand(cmd=fn.__name__, args=args, kwargs=kwargs, future_id=future_id)
        )
        self.logger.trace("cmd_queue", cmd_queue_size=self._cmd_queue.qsize())
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
                self.logger.trace("waiting for command", cmd_queue_size=cmd_queue.qsize())
                cmd = cmd_queue.get()
                self.logger.trace("got command", cmd=cmd)
                try:
                    fn = getattr(self, cmd.cmd)
                    result = fn(*cmd.args, **cmd.kwargs)
                except Exception as e:
                    self.logger.trace("send exception", fut=cmd.future_id, exception=e)
                    result_queue.put(WorkerResult(future_id=cmd.future_id, exception=e))
                    self.logger.trace("result_queue", result_queue_size=result_queue.qsize())
                else:
                    self.logger.trace("send result", fut=cmd.future_id)
                    result_queue.put(WorkerResult(future_id=cmd.future_id, result=result))
                    self.logger.trace("result_queue", result_queue_size=result_queue.qsize())
                    del result
                # cmd_queue.task_done()
                if cmd.cmd == self._wrk_shutdown_worker.__name__:
                    self.logger.debug("got shutdown command, exit", fut=cmd.future_id)
                    break
                self.logger.trace("processed, waiting for next command", fut=cmd.future_id)
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
        self.logger.debug("shutdown")

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def running(self) -> bool: ...
