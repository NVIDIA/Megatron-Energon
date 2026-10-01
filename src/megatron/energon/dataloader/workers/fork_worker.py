# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import multiprocessing
from typing import Generic, TypeVar

from megatron.energon.dataloader.asynchronous import ForkAsynchronous
from megatron.energon.dataloader.workers.async_worker import (
    DataLoaderAsynchronousWorker,
    torch_set_worker_info,
)
from megatron.energon.wrappers.gc_dataset import gc_init_worker

TSample = TypeVar("TSample", covariant=True)


class ForkDataLoaderWorker(
    ForkAsynchronous, DataLoaderAsynchronousWorker[TSample], Generic[TSample]
):
    """
    Implements the `DataLoaderWorker` interface using processes.
    """

    def _worker_run(
        self,
        cmd_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
    ) -> None:
        gc_init_worker(self._rank_worker_id)

        torch_set_worker_info(
            id=self._rank_worker_id,
            num_workers=self.worker_config.num_workers,
            seed=self._seed,
            dataset=self.dataset,
        )

        try:
            super()._worker_run(cmd_queue, result_queue)
        finally:
            self.dataset.worker_close()
            self.dataset.close()
