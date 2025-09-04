# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import multiprocessing
import os
import sys
import threading
import warnings

import structlog
import torch.multiprocessing

from megatron.energon.dataloader.asynchronous.base import (
    Asynchronous,
    QueueProtocol,
    WorkerCommand,
    WorkerResult,
)
from megatron.energon.logging import get_logger

logger = get_logger(__name__)


class ForkAsynchronous(Asynchronous):
    """Mixin for asynchronous workers that use processes."""

    _process: multiprocessing.Process | None = None
    _cmd_queue: multiprocessing.Queue
    _result_queue: multiprocessing.Queue

    _threaded_shutdown: threading.Thread | None = None

    _spawning_process: int

    def _asynchronous_init(self, name: str) -> None:
        super()._asynchronous_init(name)
        self._spawning_process = os.getpid()

    def _queues(self) -> tuple[QueueProtocol[WorkerCommand], QueueProtocol[WorkerResult]]:
        return torch.multiprocessing.Queue(), torch.multiprocessing.Queue()

    def _check_parent_process(self, evt_exit: threading.Event) -> None:
        """Check if the parent process is alive. If it is dead, exit the worker process."""
        parent_proc = torch.multiprocessing.parent_process()
        parent_pid = os.getppid()
        if parent_proc is None:
            self.logger.error("No parent process, exiting")
            os._exit(-1)
        while not evt_exit.wait(1):
            if parent_proc.exitcode is not None or os.getppid() != parent_pid:
                self.logger.error("Parent process died, exiting")
                os._exit(-1)

    def _worker_run(
        self,
        cmd_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
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
            logger.set
            self.logger.info("shutting down")
            worker_exit_evt.set()
            self.logger.info("shutting down, wait for parent_check_thread")
            parent_check_thread.join()
            self.logger.info("shutting down, close queues")
            result_queue.close()
            result_queue.join_thread()
            cmd_queue.close()
            cmd_queue.cancel_join_thread()
            self.logger.info("shutting down, done")

    def _in_worker(self) -> bool:
        return torch.multiprocessing.current_process() == self._process

    def start(self) -> None:
        torch.multiprocessing.set_start_method("fork", force=True)
        orig_num_threads = torch.get_num_threads()
        # Disable torch internal multithreading, it may deadlock the forked process.
        torch.set_num_threads(1)
        self._process = torch.multiprocessing.Process(
            target=self._worker_run,
            args=(self._cmd_queue, self._result_queue),
            daemon=True,
            name=f"ForkDataLoaderWorker-{self._name}",
        )
        self._process.start()
        # Revert the original number of threads in the main process.
        torch.set_num_threads(orig_num_threads)

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
                self._cmd_queue.close()
                self._cmd_queue.cancel_join_thread()
                self._result_queue.close()
                self._result_queue.cancel_join_thread()
                # Kill the process, because we cannot communicate with it in the gc loop.
                self._process.terminate()
                self._process = None
                self._cancel_futures()
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

    def running(self) -> bool:
        return self._process is not None

    def _assert_running(self) -> None:
        assert self._process is not None, "Worker must be started first"
        assert self._process.is_alive(), "Worker died"
