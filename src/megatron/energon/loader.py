# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import re
from typing import Callable, Optional, TypeVar

from megatron.energon.cache import CachePool, FileStore, SystemFileStore
from megatron.energon.cache.file_store import JsonlFileStore, WebdatasetFileStore
from megatron.energon.deprecation import warn_deprecated
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import SavableDataset, WebdatasetMeta
from megatron.energon.flavors.base_dataset import DatasetSampleReader
from megatron.energon.flavors.jsonl.ijsonl_reader import IJsonlReader
from megatron.energon.flavors.webdataset.itar_reader import ShardInfosITarReader
from megatron.energon.flavors.webdataset.metadata import EnergonDatasetType, get_dataset_type
from megatron.energon.savable_loader import BasicDataLoader, SavableDataLoader
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.gc_dataset import GC_DEFAULT_EVERY_N_ITER

T = TypeVar("T")


def get_savable_loader(
    dataset: SavableDataset[T],
    *,
    worker_config: Optional[WorkerConfig] = None,
    checkpoint_every_sec: float = 60,
    checkpoint_every_min_n_samples: Optional[int] = None,
    n_checkpoints: Optional[int] = None,
    gc_collect_every_n_steps: int = GC_DEFAULT_EVERY_N_ITER,
    prefetch_factor: int = 2,
    cache_pool: Optional[CachePool] = None,
    watchdog_timeout_seconds: Optional[float] = 60,
    watchdog_initial_timeout_seconds: Optional[float] = None,
    fail_on_timeout: bool = False,
) -> SavableDataLoader[T]:
    """

    Get a dataloader for the given dataset.

    Args:
        dataset: The dataset to create a loader for.
        worker_config: Deprecated. Please pass this to the dataset instead.
        checkpoint_every_sec: This is the time in seconds after which an internal checkpoint is
            saved. It may take the same duration to restore a checkpoint, but introduces additional
            overhead during reading data from the dataset, so this should be chosen accordingly.
            Only applies if using workers.
        checkpoint_every_min_n_samples: Overwrites the minimum number of samples between
            checkpoints. Defaults to `number of workers * 2`. Only applies if using workers.
        n_checkpoints: The number of internal checkpoints to keep. Only applies if using workers.
            If None, computes a suitable value.
        cache_pool: If set, the cache pool to use for the dataset.
        watchdog_timeout_seconds: The timeout in seconds. If None, the watchdog is disabled.
        watchdog_initial_timeout_seconds: The initial timeout in seconds. If None, the timeout is the same as watchdog_timeout_seconds.
        fail_on_timeout: If True, stops the whole process upon timeout, after printing a stack trace.
    Returns:
        The instantiated :class:`megatron.energon.SavableDataLoader`, yielding batches from the dataset,
        allowing to save the state of the dataset.
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

    return SavableDataLoader(
        dataset,
        checkpoint_every_sec=checkpoint_every_sec,
        checkpoint_every_min_n_samples=checkpoint_every_min_n_samples,
        n_checkpoints=n_checkpoints,
        gc_collect_every_n_steps=gc_collect_every_n_steps,
        prefetch_factor=prefetch_factor,
        cache_pool=cache_pool,
        watchdog_timeout_seconds=watchdog_timeout_seconds,
        watchdog_initial_timeout_seconds=watchdog_initial_timeout_seconds,
        fail_on_timeout=fail_on_timeout,
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
) -> BasicDataLoader[T]:
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
        The instantiated :class:`torch.data.DataLoader`, yielding batches from the dataset.
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

    return BasicDataLoader(
        dataset,
        prefetch_factor=prefetch_factor,
        cache_pool=cache_pool,
        watchdog_timeout_seconds=watchdog_timeout_seconds,
        watchdog_initial_timeout_seconds=watchdog_initial_timeout_seconds,
        fail_on_timeout=fail_on_timeout,
    )


# Regex for any URL-like string (any protocol)
_url_regex = re.compile(r"^(?P<protocol>[a-z][a-z0-9+.-]*)://(?P<path>.*)", re.IGNORECASE)


def get_file_store(
    path: str | EPath,
) -> FileStore[bytes]:
    """
    Get a file store for the given path.

    Args:
        path: The path to the file store.

    Returns:
        The instantiated :class:`megatron.energon.FileStore`.
    """
    if isinstance(path, str) and (m := _url_regex.match(path)):
        prot = m.group("protocol")
        if prot.count("+") == 1:
            # filesystem+fs_prot://
            fs_type, fs_prot = prot.split("+")
            assert fs_type == "filesystem"
            return SystemFileStore(f"{fs_prot}://{m.group('path')}")
        elif prot == "filesystem":
            # filesystem:// (may be relative or absolute)
            fs_type = "filesystem"
            return SystemFileStore(m.group("path"))
    path = EPath(path)
    ds_type = get_dataset_type(path)
    if ds_type == EnergonDatasetType.WEBDATASET:
        return WebdatasetFileStore(path)
    if ds_type == EnergonDatasetType.JSONL:
        return JsonlFileStore(path)
    if ds_type == EnergonDatasetType.FILESYSTEM:
        return SystemFileStore(path)
    raise NotImplementedError(f"Unsupported dataset type: {ds_type}")


def get_dataset_reader(
    path: str | EPath,
    *,
    split_part: str = "train",
    split_config: str | None = None,
    part_filter: Callable[[str], bool] | None = None,
) -> DatasetSampleReader:
    """
    Get a dataset reader for the given (core) dataset path.

    Args:
        path: The path to the (core) dataset.

    Returns:
        The instantiated :class:`megatron.energon.DatasetSampleReader`.
    """
    path = EPath(path)
    ds_type = get_dataset_type(path)
    if ds_type == EnergonDatasetType.WEBDATASET:
        wds_meta = WebdatasetMeta.from_config(
            path=path,
            split_part=split_part,
            split_config=split_config,
        )
        return ShardInfosITarReader(
            path,
            shard_infos=wds_meta.shards,
            part_filter=part_filter,
            itar_cache_size=1,
        )
    if ds_type == EnergonDatasetType.JSONL:
        return IJsonlReader(
            path,
            index_cache_size=1,
        )
    raise NotImplementedError(f"Unsupported dataset type for indexed access: {ds_type}")
