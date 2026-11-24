# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import queue
import sys
import threading
import warnings
from time import monotonic
from typing import Callable

from megatron.energon.dataloader.asynchronous.base import (
    Asynchronous,
    WorkerCommand,
    WorkerExitedError,
    WorkerResult,
)


class _ThreadReplies:
    _dispatch: Callable[[WorkerResult], None]

    def __init__(self, dispatch: Callable[[WorkerResult], None]):
        self._dispatch = dispatch

    def put(self, reply: WorkerResult, /) -> None:
        self._dispatch(reply)


class ThreadAsynchronous(Asynchronous):
    """Mixin for asynchronous workers that use threads."""

    _thread: threading.Thread | None = None
    _worker_exited: threading.Event

    def _run_thread(self) -> None:
        try:
            self._worker_run(self._cmd_queue, _ThreadReplies(self._dispatch_reply))
        finally:
            self._worker_exited.set()
            self._fail_pending_futures(WorkerExitedError(f"Worker {self._name} thread exited"))

    def _in_worker(self) -> bool:
        return threading.current_thread() == self._thread

    def start(self) -> None:
        self._shutdown_future = None
        self._shutdown_requested = threading.Event()
        self._worker_exited = threading.Event()
        self._cmd_queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run_thread,
            daemon=True,
            name=f"{self._name}",
        )
        try:
            self._thread.start()
        except BaseException:
            self._thread = None
            raise

    def shutdown(self, in_del: bool = False, *, deadline: float | None = None) -> None:
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
                self._shutdown_requested.set()
                self._cmd_queue.put(
                    WorkerCommand(
                        cmd=self._wrk_shutdown_worker.__name__, args=(), kwargs={}, future_id=-1
                    )
                )
                self._cancel_futures()
            else:
                if deadline is None:
                    deadline = monotonic() + self._shutdown_timeout
                try:
                    self._shutdown_worker(timeout=max(0, deadline - monotonic()))
                except WorkerExitedError:
                    pass
                finally:
                    self._cancel_futures()
                    self._thread.join(max(0, deadline - monotonic()))
                    if not self._thread.is_alive():
                        self._thread = None
                if self._thread is not None:
                    raise TimeoutError(f"Worker {self._name} thread did not shut down")

    def running(self) -> bool:
        return self._thread is not None

    def _assert_running(self) -> None:
        if self._thread is None or self._worker_exited.is_set() or not self._thread.is_alive():
            raise WorkerExitedError(f"Worker {self._name} thread exited")
