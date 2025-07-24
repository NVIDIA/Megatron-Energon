# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Generic, TypeVar, override

from megatron.energon.cache.base import CachePool
from megatron.energon.dataloader.asynchronous import (
    Asynchronous,
    QueueProtocol,
    WorkerCommand,
    WorkerResult,
)
from megatron.energon.dataloader.future import Future
from megatron.energon.dataloader.workers.base_worker import DataLoaderWorker
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.rng import SystemRng
from megatron.energon.state import FlexState
from megatron.energon.worker import WorkerConfig

TSample = TypeVar("TSample", covariant=True)


class DataLoaderAsynchronousWorker(DataLoaderWorker[TSample], Asynchronous, Generic[TSample]):
    """
    Extension of the `DataLoaderWorker`, which implements commands via a command and results queue.

    There are different implementations of the async worker:
    - :class:`ForkDataLoaderWorker` - A worker that forks a new process for each worker.
    - :class:`ThreadDataLoaderWorker` - A worker that uses threads to execute the commands.
    """

    def __init__(
        self,
        dataset: SavableDataset,
        worker_config: WorkerConfig,
        rank_worker_id: int,
        cache_pool: CachePool | None,
    ):
        super().__init__(dataset, worker_config, rank_worker_id, cache_pool)
        assert worker_config.num_workers > 0, "Async workers require num_workers > 0"
        self._asynchronous_init(name=f"wkr-{rank_worker_id}")

    # ------------------------------------------------------------------------------------------------
    # Section: Worker methods - now calling to workers via queues.

    def _worker_run(
        self, cmd_queue: QueueProtocol[WorkerCommand], result_queue: QueueProtocol[WorkerResult]
    ) -> None:
        SystemRng.seed(self._seed)
        import torch.utils.data._utils

        torch.utils.data._utils.worker._worker_info = torch.utils.data._utils.worker.WorkerInfo(
            id=self._rank_worker_id,
            num_workers=self.worker_config.num_workers,
            seed=self._seed,
            dataset=self.dataset,
        )
        self._global_worker_id = self.worker_config.global_worker_id()

        super()._worker_run(cmd_queue, result_queue)

    def _wrk_prefetch_next(self) -> TSample:
        """Wraps the super class method to call it in the worker process."""
        # The super class implementation already returns a resolved future (to be interface compatible),
        # so immediately resolve the future to the result (get returns immediately).
        return super().prefetch_next().get()

    @override
    def dataset_init(self, initial_state: FlexState | None) -> None:
        if self._in_worker():
            return super().dataset_init(initial_state)
        else:
            return self._worker_call(self.dataset_init, initial_state).get()

    @override
    def new_iter(self) -> None:
        if self._in_worker():
            return super().new_iter()
        else:
            return self._worker_call(self.new_iter).get()

    @override
    def prefetch_next(self) -> Future[TSample]:
        # Do not resolve the future here, but return it.
        if self._in_worker():
            return super().prefetch_next()
        return self._worker_call(self._wrk_prefetch_next)

    @override
    def save_state(self) -> FlexState:
        if self._in_worker():
            return super().save_state()
        else:
            return self._worker_call(self.save_state).get()
