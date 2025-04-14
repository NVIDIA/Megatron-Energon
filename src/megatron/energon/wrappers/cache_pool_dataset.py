# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Iterator, TypeVar

from megatron.energon.cache import CachePool
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class CachePoolDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """Sets the currently active cache pool for the worker."""

    cache_pool: CachePool

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        *,
        worker_config: WorkerConfig,
        cache_pool: CachePool,
    ):
        """Construct a CachePoolDataset, which sets the cache pool for the worker.

        Args:
            dataset: The input dataset to wrap
            cache_pool: The cache pool to set for the worker
        """
        super().__init__(dataset, worker_config=worker_config)
        self.cache_pool = cache_pool

    def reset_state_own(self) -> None:
        return

    def __len__(self):
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_sample]:
        self.worker_config._cache_pool = self.cache_pool
        try:
            for sample in self.dataset:
                self.worker_config._cache_pool = None
                yield sample
                self.worker_config._cache_pool = self.cache_pool

        finally:
            self.worker_config._cache_pool = None

    def config(self) -> Dict[str, Any]:
        # This is transparent, no config to be saved (it does not affect the dataset)
        return self.dataset.config()

    def __str__(self):
        return f"CachePoolDataset(cache_pool={self.cache_pool}, dataset={self.dataset})"
