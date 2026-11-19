# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import threading
import traceback
from abc import abstractmethod
from queue import Empty
from time import monotonic
from typing import Any, Callable, ParamSpec, Protocol, TypeVar

from megatron.energon.dataloader.future import CancelledError, Future
from megatron.energon.edataclass import edataclass

P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R", covariant=True)


DEBUG_LEVEL = 0


class QueueProtocol(Protocol[T]):
    """Protocol for a queue."""

    def get(self, *, timeout: float | None = None) -> T: ...

    def put(self, item: T, /) -> None: ...


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
    shutdown: bool = False


class WorkerExitedError(RuntimeError):
    """A worker exited before completing its requests."""


class WorkerSerializationError(RuntimeError):
    """A worker command or result could not be serialized."""


class FutureImpl(Future[T]):
    """Class for returning a future result from the worker.."""

    __slots__ = ("_worker", "_future_id", "_result", "_exception", "_ready")

    _worker: "Asynchronous"
    _future_id: int
    _result: T
    _exception: Exception
    _ready: threading.Event

    def __init__(self, worker: "Asynchronous", future_id: int):
        self._worker = worker
        self._future_id = future_id
        self._ready = threading.Event()

    def get(self, timeout: float | None = None) -> T:
        if not self.done():
            self._worker._wait_for_worker_result(self, timeout)
        if hasattr(self, "_exception"):
            raise self._exception
        return self._result

    def cancel(self) -> bool:
        if hasattr(self, "_result") or hasattr(self, "_exception"):
            if DEBUG_LEVEL >= 1:
                print(
                    f"[{self._worker._name}, fut={self._future_id}] already has result or exception\n",
                    end="",
                )
            return False
        return self._worker._cancel_future(self._future_id)

    def done(self) -> bool:
        return self._ready.is_set()

    def _set_result(self, result: T) -> None:
        self._result = result
        self._ready.set()

    def _set_exception(self, exception: Exception) -> None:
        self._exception = exception
        self._ready.set()

    def __str__(self) -> str:
        return f"FutureImpl(worker={self._worker._name!r}, future_id={self._future_id!r}, done={self.done()!r}, exception={getattr(self, '_exception', '<no exception>')})"


class Asynchronous:
    """Asynchronous base class."""

    _cmd_queue: QueueProtocol[WorkerCommand]
    _result_queue: QueueProtocol[WorkerResult]
    _next_future_id: int
    _pending_futures: dict[int, FutureImpl[Any]]
    _name: str
    _result_lock: threading.Lock
    _pending_lock: threading.Lock

    def _asynchronous_init(self, name: str) -> None:
        self._next_future_id = 0
        self._pending_futures = {}
        self._name = name
        self._result_lock = threading.Lock()
        self._pending_lock = threading.Lock()

    def _wait_for_worker_result(self, future: FutureImpl[Any], timeout: float | None) -> None:
        """Dispatch queued replies until this request completes or its deadline expires."""
        deadline = None if timeout is None else monotonic() + timeout
        if not self._result_lock.acquire(timeout=-1 if timeout is None else timeout):
            raise TimeoutError(f"Timed out waiting for worker {self._name}")
        try:
            while not future.done():
                interval = 0.1
                if deadline is not None:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"Timed out waiting for worker {self._name}")
                    interval = min(interval, remaining)
                try:
                    reply = self._result_queue.get(timeout=interval)
                except Empty:
                    try:
                        self._assert_running()
                    except WorkerExitedError as error:
                        self._fail_pending_futures(error)
                    continue
                except (EOFError, OSError, ValueError, WorkerSerializationError):
                    self._fail_pending_futures(self._result_queue_error())
                    continue
                self._dispatch_reply(reply)
        finally:
            self._result_lock.release()

    def _dispatch_reply(self, reply: WorkerResult) -> None:
        with self._pending_lock:
            pending = self._pending_futures.pop(reply.future_id, None)
            if pending is not None:
                if reply.exception is not None:
                    pending._set_exception(reply.exception)
                else:
                    pending._set_result(reply.result)

    def _result_queue_error(self) -> WorkerExitedError:
        return WorkerExitedError(f"Result transport for worker {self._name} failed")

    def _fail_pending_futures(self, error: WorkerExitedError | WorkerSerializationError) -> None:
        with self._pending_lock:
            for future in self._pending_futures.values():
                future._set_exception(type(error)(*error.args))
            self._pending_futures.clear()

    def _cancel_future(self, future_id: int) -> bool:
        with self._pending_lock:
            future = self._pending_futures.pop(future_id, None)
            if future is None:
                return False
            future._set_exception(CancelledError.with_current_traceback())
            return True

    def _cancel_futures(self) -> None:
        """Cancel all futures after worker shutdown."""
        with self._pending_lock:
            for future in self._pending_futures.values():
                future._set_exception(CancelledError.with_current_traceback())
            self._pending_futures.clear()

    def _worker_call(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> FutureImpl[R]:
        """
        Call a function in the worker and return a future for getting the result.
        The function must be an instance method of `self`. Uses the name to identify the function in the worker
        instance.

        Args:
            fn: The function to call.
            *args: The arguments to pass to the function.
            **kwargs: The keyword arguments to pass to the function.
        """
        assert not self._in_worker(), "worker_call must not be called in the worker"
        try:
            with self._pending_lock:
                self._assert_running()
                future_id = self._next_future_id
                self._next_future_id += 1
                future: FutureImpl[R] = FutureImpl(self, future_id)
                self._pending_futures[future_id] = future
        except (WorkerExitedError, WorkerSerializationError) as error:
            self._fail_pending_futures(error)
            raise
        if DEBUG_LEVEL >= 2:
            print(
                f"[{self._name}] worker_call {fn.__name__=} {future_id=}\n",
                end="",
            )
        try:
            self._cmd_queue.put(
                WorkerCommand(cmd=fn.__name__, args=args, kwargs=kwargs, future_id=future_id)
            )
        except Exception:
            self._cancel_future(future_id)
            raise
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
                if DEBUG_LEVEL >= 2:
                    print(
                        f"[{self._name}] waiting for command {cmd_queue.qsize()=}\n",
                        end="",
                    )
                cmd = cmd_queue.get()
                if DEBUG_LEVEL >= 2:
                    print(
                        f"[{self._name}, fut={cmd.future_id}] got command {cmd.cmd=}\n",
                        end="",
                    )
                shutting_down = cmd.cmd == self._wrk_shutdown_worker.__name__
                try:
                    fn = getattr(self, cmd.cmd)
                    result = fn(*cmd.args, **cmd.kwargs)
                except Exception as e:
                    if DEBUG_LEVEL >= 2:
                        print(f"[{self._name}, fut={cmd.future_id}] send exception {e!r}\n", end="")
                    result_queue.put(
                        WorkerResult(future_id=cmd.future_id, exception=e, shutdown=shutting_down)
                    )
                else:
                    if DEBUG_LEVEL >= 2:
                        print(f"[{self._name}, fut={cmd.future_id}] send result\n", end="")
                    result_queue.put(
                        WorkerResult(future_id=cmd.future_id, result=result, shutdown=shutting_down)
                    )
                    del result
                # cmd_queue.task_done()
                if shutting_down:
                    if DEBUG_LEVEL >= 1:
                        print(
                            f"[{self._name}, fut={cmd.future_id}] got shutdown command, exit\n",
                            end="",
                        )
                    break
                if DEBUG_LEVEL >= 2:
                    print(
                        f"[{self._name}, fut={cmd.future_id}] processed, waiting for next command\n",
                        end="",
                    )
        except:
            traceback.print_exc()
            raise

    @abstractmethod
    def _assert_running(self) -> None:
        """Raise if the worker cannot execute requests."""
        ...

    @abstractmethod
    def _in_worker(self) -> bool:
        """Check if the execution is within the worker."""
        ...

    def _wrk_shutdown_worker(self) -> None:
        """Does nothing. The actual shutdown is handled in the _worker_run method."""
        assert self._in_worker(), "_wrk_shutdown_worker must be called in the worker"

    def _shutdown_worker(self, timeout: float | None = None) -> None:
        """Shutdown the worker. The actual shutdown is handled in the _worker_run method."""
        assert not self._in_worker(), "shutdown_worker must not be called in the worker"
        # This is not actually a recursive call, because the worker loop will exit before calling this method.
        self._worker_call(self._wrk_shutdown_worker).get(timeout)
        self._cancel_futures()
        if DEBUG_LEVEL >= 1:
            print(f"[{self._name}] shutdown\n", end="")

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def running(self) -> bool: ...
