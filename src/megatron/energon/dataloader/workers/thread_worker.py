# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Generic, TypeVar

from megatron.energon.dataloader.asynchronous import ThreadAsynchronous
from megatron.energon.dataloader.workers.async_worker import (
    DataLoaderAsynchronousWorker,
)

TSample = TypeVar("TSample", covariant=True)


class ThreadDataLoaderWorker(
    ThreadAsynchronous, DataLoaderAsynchronousWorker[TSample], Generic[TSample]
):
    """
    Implements the `DataLoaderWorker` interface using threads.
    """
