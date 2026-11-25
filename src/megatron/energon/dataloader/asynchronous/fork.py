# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import multiprocessing
import os
import sys
import threading
import warnings
from ctypes import c_bool
from multiprocessing.connection import Connection, wait
from multiprocessing.context import BaseContext
from multiprocessing.queues import Queue
from multiprocessing.reduction import ForkingPickler
from queue import Empty
from time import monotonic
from typing import Any

import torch.multiprocessing

from megatron.energon.dataloader.asynchronous.base import (
    Asynchronous,
    FutureImpl,
    QueueProtocol,
    ReplyWriter,
    WorkerCommand,
    WorkerExitedError,
    WorkerResult,
    WorkerSerializationError,
)

DEBUG_LEVEL = 0
_SHUTDOWN_ACK = "_acknowledge_shutdown"
_fork_lock = threading.Lock()


def _reset_fork_lock() -> None:
    global _fork_lock
    _fork_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_fork_lock)


class _CommandQueue:
    """Queue serialized commands so a failed pickle stays in the sending thread."""

    _queue: Queue

    def __init__(self, queue: Queue):
        self._queue = queue

    def get(self, *, timeout: float | None = None) -> WorkerCommand:
        return ForkingPickler.loads(self._queue.get(timeout=timeout))

    def put(self, command: WorkerCommand) -> None:
        try:
            payload = bytes(ForkingPickler.dumps(command))
        except Exception as error:
            raise WorkerSerializationError(
                f"Cannot serialize worker command {command.cmd}"
            ) from error
        self._queue.put(payload)

    def close(self) -> None:
        self._queue.close()
        self._queue.cancel_join_thread()


class _ReplyPipe:
    """Synchronous replies. A partial message ends at EOF when its sole writer exits."""

    _connection: Connection
    _close_lock: threading.Lock

    def __init__(self, connection: Connection):
        self._connection = connection
        self._close_lock = threading.Lock()

    def get(self, *, timeout: float | None = None) -> WorkerResult:
        if not self._connection.poll(timeout):
            raise Empty
        payload = self._connection.recv_bytes()
        try:
            return ForkingPickler.loads(payload)
        except Exception as error:
            raise WorkerSerializationError("Cannot decode worker reply") from error

    def put(self, reply: WorkerResult) -> None:
        try:
            payload = ForkingPickler.dumps(reply)
        except Exception as error:
            failure = WorkerSerializationError(
                f"Cannot serialize worker reply {reply.future_id}: {error}"
            )
            payload = ForkingPickler.dumps(
                WorkerResult(future_id=reply.future_id, exception=failure, shutdown=reply.shutdown)
            )
        self._connection.send_bytes(payload)

    def close(self) -> None:
        with self._close_lock:
            self._connection.close()


class _ForkShutdownEvent:
    """One-way flag that remains writable if the worker is killed while reading it."""

    _flag: c_bool

    def __init__(self, context: BaseContext):
        self._flag = context.RawValue(c_bool, False)

    def set(self) -> None:
        self._flag.value = True

    def is_set(self) -> bool:
        return self._flag.value


class ForkAsynchronous(Asynchronous):
    """Mixin for asynchronous workers that use processes."""

    stop_concurrently = True

    _process: multiprocessing.Process | None = None
    _cmd_queue: _CommandQueue
    _result_queue: _ReplyPipe
    _result_lock: threading.Lock
    _spawning_process: int
    _receiver: threading.Thread | None
    _receiver_stop: threading.Event
    _receive_error: WorkerExitedError | WorkerSerializationError | None

    def _asynchronous_init(self, name: str) -> None:
        super()._asynchronous_init(name)
        self._spawning_process = os.getpid()

    def _check_parent_process(self, evt_exit: threading.Event) -> None:
        """Check if the parent process is alive. If it is dead, exit the worker process."""
        parent_proc = torch.multiprocessing.parent_process()
        parent_pid = os.getppid()
        if parent_proc is None:
            print(f"[{self._name}] No parent process, exiting", file=sys.stderr)
            os._exit(-1)
        while not evt_exit.wait(1):
            if parent_proc.exitcode is not None or os.getppid() != parent_pid:
                print(f"[{self._name}] Parent process died, exiting", file=sys.stderr)
                os._exit(-1)

    def _worker_run(
        self, cmd_queue: QueueProtocol[WorkerCommand], result_queue: ReplyWriter
    ) -> None:
        try:
            from torch.utils.data._utils import signal_handling

            signal_handling._set_worker_signal_handlers()
        except (ImportError, AttributeError):
            pass

        try:
            torch.multiprocessing._set_thread_name("pt_data_worker")
        except (ImportError, AttributeError):
            pass

        # Disable torch internal multithreading, it may deadlock the forked process.
        torch.set_num_threads(1)

        worker_exit_evt = threading.Event()
        parent_check_thread = threading.Thread(
            target=self._check_parent_process, args=(worker_exit_evt,), daemon=True
        )
        parent_check_thread.start()
        try:
            super()._worker_run(cmd_queue, result_queue)
            # Torch may need this process to transfer tensor storage for replies still in flight.
            # Keep its descriptor broker alive until the parent reads the terminal reply.
            while cmd_queue.get().cmd != _SHUTDOWN_ACK:
                pass
        finally:
            if DEBUG_LEVEL >= 1:
                print(f"[{self._name}] shutting down\n", end="")
            worker_exit_evt.set()
            if DEBUG_LEVEL >= 1:
                print(
                    f"[{self._name}] shutting down, wait for parent_check_thread\n",
                    end="",
                )
            parent_check_thread.join()

    def _run_forked_worker(self, sender: _ReplyPipe) -> None:
        self._result_queue.close()
        try:
            self._worker_run(self._cmd_queue, sender)
        finally:
            sender.close()
            self._cmd_queue.close()

    def _in_worker(self) -> bool:
        return torch.multiprocessing.current_process() == self._process

    def start(self) -> None:
        # No other Energon fork may inherit a reply writer during this interval.
        with _fork_lock:
            self._start_process()

    def _start_process(self) -> None:
        if self._process is not None or getattr(self, "_receiver", None) is not None:
            raise RuntimeError(f"Worker {self._name} is already started or still shutting down")
        self._shutdown_future = None
        self._receiver = None
        self._receiver_stop = threading.Event()
        self._result_lock = threading.Lock()
        self._receive_error = None
        context = torch.multiprocessing.get_context("fork")
        self._shutdown_requested = _ForkShutdownEvent(context)
        # Allocate each worker's pipes right before forking. Later children must never
        # inherit another worker's reply writer, or a dead worker won't produce EOF.
        self._cmd_queue = _CommandQueue(context.Queue())
        reader, writer = context.Pipe(duplex=False)
        self._result_queue = _ReplyPipe(reader)
        sender = _ReplyPipe(writer)
        orig_num_threads = torch.get_num_threads()
        try:
            # Fork with Torch's internal multithreading disabled.
            torch.set_num_threads(1)
            self._process = context.Process(
                target=self._run_forked_worker,
                args=(sender,),
                daemon=True,
                name=f"ForkDataLoaderWorker-{self._name}",
            )
            self._process.start()
        except BaseException:
            self._cmd_queue.close()
            self._result_queue.close()
            self._process = None
            raise
        finally:
            sender.close()
            torch.set_num_threads(orig_num_threads)

    def _receive_replies(self, pipe: _ReplyPipe, stop: threading.Event) -> None:
        try:
            while not stop.is_set():
                try:
                    reply = pipe.get(timeout=0.1)
                except Empty:
                    self._assert_running()
                else:
                    if not stop.is_set():
                        self._dispatch_reply(reply)
                    if reply.shutdown:
                        return
        except BaseException as error:
            if not stop.is_set():
                if isinstance(error, (WorkerExitedError, WorkerSerializationError)):
                    failure = error
                elif isinstance(error, (EOFError, OSError, ValueError)):
                    failure = self._result_queue_error()
                else:
                    failure = WorkerSerializationError(
                        f"Worker {self._name} reply receiver failed: {type(error).__name__}"
                    )
                self._receive_error = failure
                self._fail_pending_futures(failure)
        finally:
            pipe.close()

    def _wait_for_worker_result(self, future: FutureImpl[Any], timeout: float | None) -> None:
        deadline = None if timeout is None else monotonic() + timeout
        if not self._result_lock.acquire(timeout=-1 if timeout is None else timeout):
            raise TimeoutError(f"Timed out waiting for worker {self._name}")
        try:
            if not future.done() and self._receiver is None and not self._receiver_stop.is_set():
                self._receiver = threading.Thread(
                    target=self._receive_replies,
                    args=(self._result_queue, self._receiver_stop),
                    daemon=True,
                    name=f"{self._name}-replies",
                )
                try:
                    self._receiver.start()
                except Exception as error:
                    self._receiver = None
                    self._receive_error = WorkerExitedError(
                        f"Could not start reply receiver for worker {self._name}: {error}"
                    )
                    self._fail_pending_futures(self._receive_error)
        finally:
            self._result_lock.release()
        remaining = None if deadline is None else max(0.0, deadline - monotonic())
        super()._wait_for_worker_result(future, remaining)

    def shutdown(self, in_del: bool = False, *, deadline: float | None = None) -> None:
        if self._spawning_process != os.getpid():
            return
        process = self._process
        receiver = getattr(self, "_receiver", None)
        if process is None and receiver is None:
            return

        if deadline is None:
            deadline = monotonic() + self._shutdown_timeout
        # A short, shared allowance lets killed children and reply receivers exit.
        reap_deadline = deadline + 0.25
        timed_out = False
        cleanup_error = None
        if process is not None:
            try:
                if in_del:
                    warnings.warn(f"Terminating unclosed worker {self._name}", ResourceWarning)
                    if process.is_alive():
                        process.kill()
                elif self._shutdown_future is not None or process.is_alive():
                    try:
                        self._shutdown_worker(timeout=max(0.0, deadline - monotonic()))
                    except TimeoutError:
                        timed_out = True
                    except WorkerExitedError:
                        pass
                    except Exception as error:
                        cleanup_error = error
                    if not timed_out and process.is_alive():
                        try:
                            self._cmd_queue.put(
                                WorkerCommand(cmd=_SHUTDOWN_ACK, args=(), kwargs={}, future_id=-1)
                            )
                        except Exception as error:
                            if cleanup_error is None:
                                cleanup_error = error
                            else:
                                cleanup_error.add_note(f"Could not acknowledge shutdown: {error}")
                process.join(max(0.0, deadline - monotonic()))
                if process.is_alive():
                    timed_out = True
                    process.kill()
                    process.join(max(0, reap_deadline - monotonic()))
            finally:
                self._receiver_stop.set()
                self._cancel_futures()
                self._cmd_queue.close()
                self._result_queue.close()
                if not process.is_alive():
                    self._process = None

        receiver = self._receiver
        if receiver is not None:
            # Let EOF wake the receiver after a forced process exit, even
            # when the acknowledgement already used the full grace period.
            receiver.join(max(0, reap_deadline - monotonic()))
            if receiver.is_alive():
                timed_out = True
            else:
                self._receiver = None
        if timed_out:
            raise TimeoutError(
                f"Worker {self._name} did not shut down within the deadline"
            ) from cleanup_error
        if cleanup_error is not None:
            raise cleanup_error

    def running(self) -> bool:
        return self._process is not None or (
            getattr(self, "_receiver", None) is not None and self._receiver.is_alive()
        )

    def _assert_running(self) -> None:
        process = self._process
        if self._receive_error is not None:
            raise type(self._receive_error)(*self._receive_error.args)
        if process is None or self._receiver_stop.is_set():
            raise WorkerExitedError(f"Worker {self._name} is not running")
        if not process.is_alive():
            raise WorkerExitedError(f"Worker {self._name} exited with code {process.exitcode}")

    def _result_queue_error(self) -> WorkerExitedError:
        process = self._process
        if process is not None:
            # EOF can be observed just before the process exit status is available.
            wait([process.sentinel], timeout=0.1)
        try:
            self._assert_running()
        except WorkerExitedError as error:
            return error
        return WorkerExitedError(f"Result transport for worker {self._name} failed")
