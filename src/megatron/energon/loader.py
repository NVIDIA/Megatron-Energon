# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal, Optional, TypeVar

from megatron.energon.cache import CachePool
from megatron.energon.dataloader import (
    DataLoader,
    DataLoaderWorker,
    ForkDataLoaderWorker,
    ThreadDataLoaderWorker,
)
from megatron.energon.errors import warn_deprecated
from megatron.energon.flavors import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.gc_dataset import GC_DEFAULT_EVERY_N_ITER

T = TypeVar("T")


def get_savable_loader(
    dataset: SavableDataset[T],
    *,
    worker_config: Optional[WorkerConfig] = None,
    worker_type: Literal["main", "fork", "thread"] | type[DataLoaderWorker] = "fork",
    gc_freeze_at_start: bool = True,
    gc_collect_every_n_steps: int = GC_DEFAULT_EVERY_N_ITER,
    prefetch_factor: int = 2,
    cache_pool: Optional[CachePool] = None,
    watchdog_timeout_seconds: Optional[float] = 60,
    watchdog_initial_timeout_seconds: Optional[float] = None,
    fail_on_timeout: bool = False,
    pin_memory: bool = True,
) -> DataLoader[T]:
    """

    Get a dataloader for the given dataset.

    Args:
        dataset: The dataset to create a loader for.
        worker_config: Deprecated. Please pass this to the dataset instead.
        worker_type: The type of worker to use.
        gc_freeze_at_start: If True, the garbage collector is frozen at the start of the loader.
        gc_collect_every_n_steps: The number of steps after which the garbage collector is called.
        prefetch_factor: The factor by which to prefetch the dataset.
        cache_pool: If set, the cache pool to use for the dataset.
        watchdog_timeout_seconds: The timeout in seconds. If None, the watchdog is disabled.
        watchdog_initial_timeout_seconds: The initial timeout in seconds. If None, the timeout is the same as watchdog_timeout_seconds.
        fail_on_timeout: If True, stops the whole process upon timeout, after printing a stack trace.
        pin_memory: If True, the dataset is pinned to memory.

    Returns:
        The instantiated :class:`megatron.energon.DataLoader`, yielding batches from the dataset.
    """
    if worker_config is not None:
        if worker_config != dataset.worker_config:
            raise AssertionError(
                "The worker_config passed to get_savable_loader() does not match the one of the dataset. "
                "Also note, it is deprecated to pass one to get_savable_loader() and it will have no effect."
            )
        else:
            warn_deprecated(
                "Passing a worker_config to get_savable_loader() is deprecated and will have no effect."
            )

    if worker_type == "fork":
        worker_type = ForkDataLoaderWorker
    elif worker_type == "thread":
        worker_type = ThreadDataLoaderWorker
    elif worker_type == "main":
        worker_type = DataLoaderWorker
    elif not issubclass(worker_type, DataLoaderWorker):
        raise ValueError(f"Invalid worker type: {worker_type}")
    if dataset.worker_config.num_workers == 0:
        assert prefetch_factor == 2
        prefetch_factor = 1
        pin_memory_arg = None
    else:
        pin_memory_arg = "automatic" if pin_memory else None

    return DataLoader(
        dataset,
        prefetch_factor=prefetch_factor,
        worker_type=worker_type,
        cache_pool=cache_pool,
        gc_collect_every_n_steps=gc_collect_every_n_steps,
        gc_freeze_at_start=gc_freeze_at_start,
        watchdog_timeout_seconds=watchdog_timeout_seconds,
        watchdog_initial_timeout_seconds=watchdog_initial_timeout_seconds,
        fail_on_timeout=fail_on_timeout,
        pin_memory=pin_memory_arg,
    )


def get_loader(
    dataset: SavableDataset[T],
    *,
    worker_config: Optional[WorkerConfig] = None,
    prefetch_factor: int = 2,
    cache_pool: Optional[CachePool] = None,
    watchdog_timeout_seconds: Optional[float] = 60,
    watchdog_initial_timeout_seconds: Optional[float] = None,
    fail_on_timeout: bool = False,
) -> DataLoader[T]:
    """
    Get a dataloader for the given dataset.

    Args:
        dataset: The dataset to create a loader for.
        worker_config: Deprecated. Please pass this to the dataset instead.
        cache_pool: If set, the cache pool to use for the dataset.
        watchdog_timeout_seconds: The timeout in seconds. If None, the watchdog is disabled.
        watchdog_initial_timeout_seconds: The initial timeout in seconds. If None, the timeout is the same as watchdog_timeout_seconds.
        fail_on_timeout: If True, stops the whole process upon timeout, after printing a stack trace.

    Returns:
        The instantiated :class:`DataLoader`, yielding batches from the dataset.
    """
    if worker_config is not None:
        if worker_config != dataset.worker_config:
            raise AssertionError(
                "The worker_config passed to get_loader() does not match the one of the dataset. "
                "Also note, it is deprecated to pass one to get_loader() and it will have no effect."
            )
        else:
            warn_deprecated(
                "Passing a worker_config to get_loader() is deprecated and will have no effect."
            )

    if dataset.worker_config.num_workers == 0:
        assert prefetch_factor == 2
        prefetch_factor = 1
        pin_memory = None
    else:
        pin_memory = "automatic"

    return DataLoader(
        dataset,
        prefetch_factor=prefetch_factor,
        cache_pool=cache_pool,
        watchdog_timeout_seconds=watchdog_timeout_seconds,
        watchdog_initial_timeout_seconds=watchdog_initial_timeout_seconds,
        fail_on_timeout=fail_on_timeout,
        pin_memory=pin_memory,
    )
