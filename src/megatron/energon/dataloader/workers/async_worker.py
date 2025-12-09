# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import threading
from typing import Generic, TypeVar

import torch.utils.data._utils

from megatron.energon.cache.base import CachePool
from megatron.energon.dataloader.asynchronous import (
    Asynchronous,
    QueueProtocol,
    WorkerCommand,
    WorkerResult,
)
from megatron.energon.dataloader.future import Future
from megatron.energon.dataloader.workers.base_worker import (
    DataLoaderWorker,
    WorkerSampleRestoreKey,
    WorkerState,
)
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.rng import SystemRng
from megatron.energon.worker import WorkerConfig

TSample = TypeVar("TSample", covariant=True)

# Patching the torch worker info.

_thread_local_worker_info = threading.local()


def torch_set_worker_info(id: int, num_workers: int, seed: int, dataset: SavableDataset):
    _thread_local_worker_info._worker_info = torch.utils.data._utils.worker.WorkerInfo(
        id=id,
        num_workers=num_workers,
        seed=seed,
        dataset=dataset,
    )


def _patch_get_worker_info():
    return getattr(_thread_local_worker_info, "_worker_info", None)


torch.utils.data.get_worker_info = _patch_get_worker_info


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
        self._global_worker_id = self.worker_config.global_worker_id()

        super()._worker_run(cmd_queue, result_queue)

    def _wrk_prefetch_next(self) -> TSample:
        """Wraps the super class method to call it in the worker process."""
        # The super class implementation already returns a resolved future (to be interface compatible),
        # so immediately resolve the future to the result (get returns immediately).
        return super().prefetch_next().get()

    def _wrk_restore_sample(self, restore_key: WorkerSampleRestoreKey) -> TSample:
        """Wraps the super class method to call it in the worker process."""
        # The super class implementation already returns a resolved future (to be interface compatible),
        # so immediately resolve the future to the result (get returns immediately).
        return super().restore_sample(restore_key).get()

    def dataset_init(self, initial_state: WorkerState | None) -> None:
        if self._in_worker():
            return super().dataset_init(initial_state)
        else:
            return self._worker_call(self.dataset_init, initial_state).get()

    def new_iter(self) -> None:
        if self._in_worker():
            return super().new_iter()
        else:
            return self._worker_call(self.new_iter).get()

    def prefetch_next(self) -> Future[TSample]:
        # Do not resolve the future here, but return it.
        if self._in_worker():
            return super().prefetch_next()
        return self._worker_call(self._wrk_prefetch_next)

    def restore_sample(self, restore_key: WorkerSampleRestoreKey) -> Future[TSample]:
        # Do not resolve the future here, but return it.
        assert isinstance(restore_key, WorkerSampleRestoreKey)
        if self._in_worker():
            return super().restore_sample(restore_key)
        return self._worker_call(self._wrk_restore_sample, restore_key)

    def save_state(self) -> WorkerState:
        if self._in_worker():
            return super().save_state()
        else:
            return self._worker_call(self.save_state).get()
