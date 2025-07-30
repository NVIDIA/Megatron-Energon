# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import threading
from typing import Generic, TypeVar

import torch.utils.data

from megatron.energon.dataloader.asynchronous import ThreadAsynchronous, WorkerCommand, WorkerResult
from megatron.energon.dataloader.asynchronous.base import QueueProtocol
from megatron.energon.dataloader.workers.async_worker import (
    DataLoaderAsynchronousWorker,
)

TSample = TypeVar("TSample", covariant=True)


_orig_get_worker_info = torch.utils.data.get_worker_info

_thread_local_worker_info = threading.local()


def _patch_get_worker_info():
    if not hasattr(_thread_local_worker_info, "_worker_info"):
        _thread_local_worker_info._worker_info = _orig_get_worker_info()
    return _orig_get_worker_info()


torch.utils.data.get_worker_info = _patch_get_worker_info


class ThreadDataLoaderWorker(
    ThreadAsynchronous, DataLoaderAsynchronousWorker[TSample], Generic[TSample]
):
    """
    Implements the `DataLoaderWorker` interface using threads.
    """

    def _worker_run(
        self, cmd_queue: QueueProtocol[WorkerCommand], result_queue: QueueProtocol[WorkerResult]
    ) -> None:
        _thread_local_worker_info._worker_info = torch.utils.data._utils.worker.WorkerInfo(
            id=self._rank_worker_id,
            num_workers=self.worker_config.num_workers,
            seed=self._seed,
            dataset=self.dataset,
        )
        return super()._worker_run(cmd_queue, result_queue)
