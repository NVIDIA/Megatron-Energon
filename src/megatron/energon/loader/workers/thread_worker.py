# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import queue
import sys
import threading
import warnings
from typing import Generic, TypeVar, override

from megatron.energon.cache.base import CachePool
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.loader.workers.async_worker import DataLoaderAsynchronousWorker
from megatron.energon.worker import WorkerConfig

TSample = TypeVar("TSample", covariant=True)


class ThreadDataLoaderWorker(DataLoaderAsynchronousWorker[TSample], Generic[TSample]):
    """
    Implements the `DataLoaderWorker` interface using threads.
    """

    _thread: threading.Thread | None = None
    _cmd_queue: queue.Queue
    _result_queue: queue.Queue

    def __init__(
        self,
        dataset: SavableDataset,
        worker_config: WorkerConfig,
        rank_worker_id: int,
        cache_pool: CachePool | None,
    ):
        super().__init__(
            dataset,
            worker_config=worker_config,
            rank_worker_id=rank_worker_id,
            cmd_queue=queue.Queue(),
            result_queue=queue.Queue(),
            cache_pool=cache_pool,
        )

    def _worker_run(self, cmd_queue: queue.Queue, result_queue: queue.Queue) -> None:
        super()._worker_run(cmd_queue, result_queue)

    @override
    def _in_worker(self) -> bool:
        return threading.current_thread() == self._thread

    @override
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._worker_run,
            args=(self._cmd_queue, self._result_queue),
            daemon=True,
            name=f"ThreadDataLoaderWorker-{self._rank_worker_id}",
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
                    self.WorkerCommand(
                        cmd=self._wrk_shutdown_worker.__name__, args=(), kwargs={}, future_id=-1
                    )
                )
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
