# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import multiprocessing
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List, Optional


class BaseAggregator(ABC):
    """
    Base class for a user-defined aggregator.
    Implement on_start, on_item, and on_finish to handle aggregator logic.
    """

    @abstractmethod
    def on_start(self, aggregator_pool: AggregatorPool) -> None:
        """
        Called exactly once in the aggregator process before receiving any items.
        """
        pass

    @abstractmethod
    def on_item(self, item: Any, aggregator_pool: AggregatorPool) -> None:
        """
        Called for each item produced by the workers.
        """
        pass

    @abstractmethod
    def on_finish(self, aggregator_pool: AggregatorPool) -> None:
        """
        Called once when all workers have signaled completion (i.e. all items are processed).
        """
        pass

    def get_final_result_data(self) -> Any:
        """
        Called after on_finish to retrieve any final data produced by the aggregator.
        """
        return None


class AggregatorPool:
    """
    A pool that manages multiple worker processes sending results to
    a single aggregator process.

    The user must provide:
      - user_produce_data(task) -> yields items (streaming results)
      - aggregator: an instance of a class derived from BaseAggregator
                    which implements on_start, on_item, on_finish, etc.
    """

    def __init__(
        self,
        num_workers: int,
        user_produce_data: Callable[[Any], Iterable[Any]],
        aggregator: BaseAggregator,
    ) -> None:
        """
        :param num_workers: number of worker processes
        :param user_produce_data: function(task) -> yields items (the "large" data stream)
        :param aggregator: an instance of a user-defined class for handling aggregator logic
        """
        self.num_workers: int = num_workers
        self.user_produce_data: Callable[[Any], Iterable[Any]] = user_produce_data
        self.aggregator: BaseAggregator = aggregator

        # Queues for tasks and results
        self.task_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
        self.result_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()

        # Queue to pass final aggregator data back to the main process
        self._final_result_data_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()

        # Keep track of how many workers have finished
        self._finished_workers = multiprocessing.Value("i", 0)

        # Processes
        self.workers: List[multiprocessing.Process] = []
        self.aggregator_process: Optional[multiprocessing.Process] = None

        # Will store whatever is pulled from _final_data_queue in close()
        self._aggregator_final_result_data: Optional[Any] = None

    def _worker(self, worker_id: int) -> None:
        """Function that runs inside each worker process."""
        while True:
            task = self.task_queue.get()
            if task is None:
                # No more tasks, signal aggregator that this worker is done
                break

            # Produce data in a streaming fashion
            for item in self.user_produce_data(task):
                self.result_queue.put(item)

        # After finishing all tasks, send a sentinel to the aggregator
        self.result_queue.put(None)

    def _aggregator_run(self) -> None:
        """
        Function that runs in the aggregator process.
        Keeps reading items from result_queue.
        - If an item is None, that means a worker finished all of its tasks.
        - Otherwise, call aggregator.on_item(...) with that item.
        """
        # Let the aggregator do any initialization it needs
        self.aggregator.on_start(self)

        local_finished_workers = 0

        while True:
            item = self.result_queue.get()
            if item is None:
                # A worker has finished all of its tasks
                local_finished_workers += 1
                with self._finished_workers.get_lock():
                    self._finished_workers.value = local_finished_workers

                # Check if all workers are done
                if local_finished_workers == self.num_workers:
                    break
            else:
                # Process the item in the aggregator
                self.aggregator.on_item(item, self)

        # All workers done, aggregator can finalize
        self.aggregator.on_finish(self)

        # After finishing, serialize the aggregator's final data
        final_result_data = self.aggregator.get_final_result_data()
        self._send_final_aggregator_data(final_result_data)

    def _send_final_aggregator_data(self, data: Any) -> None:
        """
        Called in the aggregator process to push the aggregator's final
        data back to the main process.
        """
        self._final_result_data_queue.put(data)

    def is_done(self) -> bool:
        """
        For optional use inside aggregator methods to check
        if all workers have signaled completion.
        """
        with self._finished_workers.get_lock():
            return self._finished_workers.value == self.num_workers

    def start(self) -> None:
        """
        Start the aggregator process and the worker processes.
        """
        # Start aggregator process
        self.aggregator_process = multiprocessing.Process(target=self._aggregator_run, daemon=True)
        self.aggregator_process.start()

        # Start worker processes
        for w_id in range(self.num_workers):
            p = multiprocessing.Process(target=self._worker, args=(w_id,), daemon=True)
            p.start()
            self.workers.append(p)

    def submit_task(self, task: Any) -> None:
        """
        Submit a task to be processed by a worker.
        """
        self.task_queue.put(task)

    def close(self) -> None:
        """
        Signal all workers to exit, wait for them, and then
        wait for the aggregator to finish. Finally, retrieve
        the aggregator's final data from the queue (if any).
        """
        # Send the sentinel (None) to each worker
        for _ in range(self.num_workers):
            self.task_queue.put(None)

        # Wait for all workers to finish
        for p in self.workers:
            p.join()

        # Now wait for aggregator to finish reading the queue
        if self.aggregator_process is not None:
            self.aggregator_process.join()

        # Retrieve aggregator final data from the queue (if aggregator posted any)
        # We only expect one item in this queue: the aggregator's final data.
        if not self._final_result_data_queue.empty():
            self._aggregator_final_result_data = self._final_result_data_queue.get()

    def get_final_aggregator_data(self) -> Any:
        """
        Access whatever final data the aggregator produced, after close() is called.
        """
        return self._aggregator_final_result_data
