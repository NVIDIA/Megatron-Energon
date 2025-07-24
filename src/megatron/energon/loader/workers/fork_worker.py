# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import multiprocessing
import os
import sys
import threading
import warnings
from typing import Generic, TypeVar, override

from megatron.energon.cache.base import CachePool
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.loader.workers.async_worker import DataLoaderAsynchronousWorker
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.gc_dataset import gc_init_worker

TSample = TypeVar("TSample", covariant=True)


class ForkDataLoaderWorker(DataLoaderAsynchronousWorker[TSample], Generic[TSample]):
    """
    Implements the `DataLoaderWorker` interface using processes.
    """

    _process: multiprocessing.Process | None = None
    _cmd_queue: multiprocessing.Queue
    _result_queue: multiprocessing.Queue

    _threaded_shutdown: threading.Thread | None = None

    _spawning_process: int

    def __init__(
        self,
        dataset: SavableDataset,
        worker_config: WorkerConfig,
        rank_worker_id: int,
        cache_pool: CachePool | None,
    ):
        multiprocessing.set_start_method("fork", force=True)
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
        """Check if the parent process is alive. If it is dead, exit the worker process."""
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
    ) -> None:
        gc_init_worker(self._rank_worker_id)
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
            print(f"[wrk={self._rank_worker_id}] shutting down\n", end="")
            worker_exit_evt.set()
            print(
                f"[wrk={self._rank_worker_id}] shutting down, wait for parent_check_thread\n",
                end="",
            )
            parent_check_thread.join()
            print(f"[wrk={self._rank_worker_id}] shutting down, close queues\n", end="")
            result_queue.close()
            result_queue.join_thread()
            cmd_queue.close()
            cmd_queue.cancel_join_thread()
            print(f"[wrk={self._rank_worker_id}] shutting down, done\n", end="")

    @override
    def _in_worker(self) -> bool:
        return multiprocessing.current_process() == self._process

    @override
    def start(self) -> None:
        self._process = multiprocessing.Process(
            target=self._worker_run,
            args=(self._cmd_queue, self._result_queue),
            daemon=True,
            name=f"ForkDataLoaderWorker-{self._rank_worker_id}",
        )
        self._process.start()

    @override
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
                print(
                    "WARNING: Explicitly call DataLoader.shutdown() to avoid leaking processes. Terminating worker process.\n",
                    end="",
                    file=sys.stderr,
                )
                self._cmd_queue.close()
                self._cmd_queue.cancel_join_thread()
                self._result_queue.close()
                self._result_queue.cancel_join_thread()
                # Kill the process, because we cannot communicate with it in the gc loop.
                self._process.terminate()
                self._process = None
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

    @override
    def running(self) -> bool:
        return self._process is not None

    def _assert_running(self) -> None:
        assert self._process is not None, "Worker must be started first"
        assert self._process.is_alive(), "Worker died"
