# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import threading
import traceback
from abc import abstractmethod
from typing import Any, Callable, ClassVar, ParamSpec, Protocol, TypeVar

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


class ReplyWriter(Protocol):
    def put(self, reply: "WorkerResult", /) -> None: ...


class ShutdownEvent(Protocol):
    """Shutdown signal shared by a worker and its caller."""

    def set(self) -> None: ...

    def is_set(self) -> bool: ...


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

    def get_with_timeout(self, timeout: float | None) -> T:
        return self.get(timeout)

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
    _next_future_id: int
    _pending_futures: dict[int, FutureImpl[Any]]
    _name: str
    _pending_lock: threading.Lock
    _shutdown_future: FutureImpl[None] | None = None
    _shutdown_timeout: float = 15.0
    _shutdown_requested: ShutdownEvent
    _cancel_on_shutdown: ClassVar[tuple[str, ...]] = ()
    stop_concurrently: ClassVar[bool] = False

    @property
    def shutdown_grace_period(self) -> float:
        return self._shutdown_timeout

    def _asynchronous_init(self, name: str) -> None:
        self._next_future_id = 0
        self._pending_futures = {}
        self._name = name
        self._pending_lock = threading.Lock()

    def _wait_for_worker_result(self, future: FutureImpl[Any], timeout: float | None) -> None:
        if not future._ready.wait(timeout):
            raise TimeoutError(f"Timed out waiting for worker {self._name}")

    def _dispatch_reply(self, reply: WorkerResult) -> None:
        with self._pending_lock:
            pending = self._pending_futures.pop(reply.future_id, None)
            if pending is not None:
                if reply.exception is not None:
                    pending._set_exception(reply.exception)
                else:
                    pending._set_result(reply.result)

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
        self, cmd_queue: QueueProtocol[WorkerCommand], result_queue: ReplyWriter
    ) -> None:
        """
        The worker main loop.
        It waits for commands via the command queue and executes them.
        The functions to call are identified by their name.
        Replies are written to the backend's result sink.
        The worker exits when the command `_shutdown_worker` is received.
        """
        assert self._in_worker(), "_worker_run must be called in the worker"
        shutting_down = False
        try:
            while True:
                cmd = cmd_queue.get()
                if DEBUG_LEVEL >= 2:
                    print(
                        f"[{self._name}, fut={cmd.future_id}] got command {cmd.cmd=}\n",
                        end="",
                    )
                shutting_down = cmd.cmd == self._wrk_shutdown_worker.__name__
                try:
                    if cmd.cmd in self._cancel_on_shutdown and self._shutdown_requested.is_set():
                        raise CancelledError("Queued prefetch cancelled during shutdown")
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
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            if not shutting_down:
                self._finish_worker()

    @abstractmethod
    def _assert_running(self) -> None:
        """Raise if the worker cannot execute requests."""
        ...

    @abstractmethod
    def _in_worker(self) -> bool:
        """Check if the execution is within the worker."""
        ...

    def _finish_worker(self) -> None:
        """Release resources in the worker execution context."""

    def _wrk_shutdown_worker(self) -> None:
        assert self._in_worker(), "_wrk_shutdown_worker must be called in the worker"
        self._finish_worker()

    def begin_shutdown(self) -> None:
        """Ask the worker to release its resources before acknowledging shutdown."""
        if self._shutdown_future is None:
            self._shutdown_requested.set()
            self._shutdown_future = self._worker_call(self._wrk_shutdown_worker)

    def _shutdown_worker(self, timeout: float | None = None) -> None:
        self.begin_shutdown()
        assert self._shutdown_future is not None
        self._shutdown_future.get(timeout)
        self._cancel_futures()
        if DEBUG_LEVEL >= 1:
            print(f"[{self._name}] shutdown\n", end="")

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def shutdown(self, in_del: bool = False, *, deadline: float | None = None) -> None: ...

    @abstractmethod
    def running(self) -> bool: ...
