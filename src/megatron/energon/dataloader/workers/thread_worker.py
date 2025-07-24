# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import queue
import sys
import threading
import warnings
from typing import Generic, TypeVar, override

from megatron.energon.dataloader.workers.async_worker import (
    AsynchronousMixin,
    DataLoaderAsynchronousWorker,
    QueueProtocol,
    WorkerCommand,
    WorkerResult,
)

TSample = TypeVar("TSample", covariant=True)


class ThreadAsynchronousMixin(AsynchronousMixin):
    """Mixin for asynchronous workers that use threads."""

    _thread: threading.Thread | None = None

    @override
    def _queues(self) -> tuple[QueueProtocol[WorkerCommand], QueueProtocol[WorkerResult]]:
        return queue.Queue(), queue.Queue()

    @override
    def _in_worker(self) -> bool:
        return threading.current_thread() == self._thread

    @override
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._worker_run,
            args=(self._cmd_queue, self._result_queue),
            daemon=True,
            name=f"{self._name}",
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
                    WorkerCommand(
                        cmd=self._wrk_shutdown_worker.__name__, args=(), kwargs={}, future_id=-1
                    )
                )
                self._cancel_futures()
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


class ThreadDataLoaderWorker(
    ThreadAsynchronousMixin, DataLoaderAsynchronousWorker[TSample], Generic[TSample]
):
    """
    Implements the `DataLoaderWorker` interface using threads.
    """
