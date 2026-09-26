# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Generic, TypeVar

from megatron.energon.dataloader.asynchronous import ThreadAsynchronous, WorkerCommand, WorkerResult
from megatron.energon.dataloader.asynchronous.base import QueueProtocol
from megatron.energon.dataloader.workers.async_worker import (
    DataLoaderAsynchronousWorker,
    torch_set_worker_info,
)

TSample = TypeVar("TSample", covariant=True)


class ThreadDataLoaderWorker(
    ThreadAsynchronous, DataLoaderAsynchronousWorker[TSample], Generic[TSample]
):
    """
    Implements the `DataLoaderWorker` interface using threads.
    """

    def _worker_run(
        self, cmd_queue: QueueProtocol[WorkerCommand], result_queue: QueueProtocol[WorkerResult]
    ) -> None:
        torch_set_worker_info(
            id=self._rank_worker_id,
            num_workers=self.worker_config.num_workers,
            seed=self._seed,
            dataset=self.dataset,
        )
        try:
            return super()._worker_run(cmd_queue, result_queue)
        finally:
            self.dataset.worker_close()
