# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Iterator, Sequence, TypeVar

from megatron.energon.cache.base import FileStore
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.state import FlexState
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class FileStoreInitWrapper(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """This dataset wraps a primary dataset, and additional auxiliary datasets for (de)initialization."""

    auxiliary_datasets: Sequence[FileStore]

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        *,
        auxiliary_datasets: Sequence[FileStore] = (),
        worker_config: WorkerConfig,
    ):
        """Construct the auxiliary datasets dataset, which wraps a primary dataset and additional
        auxiliary datasets for initialization.

        Args:
            dataset: The input dataset to wrap
            auxiliary_datasets: The additional datasets to (de)initialize
            worker_config: The worker configuration
        """
        super().__init__(dataset, worker_config=worker_config)
        self.auxiliary_datasets = auxiliary_datasets

    def reset_state_own(self) -> None:
        for ds in self.auxiliary_datasets:
            ds.worker_init()
        cache_pool = self.worker_config.active_worker_cache_pool()
        if cache_pool is not None:
            cache_pool.worker_init()

    def worker_close(self) -> None:
        for ds in self.auxiliary_datasets:
            ds.worker_close()
        cache_pool = self.worker_config.active_worker_cache_pool()
        if cache_pool is not None:
            cache_pool.worker_close()
        super().worker_close()

    def close(self) -> None:
        for ds in self.auxiliary_datasets:
            ds.close()
        cache_pool = self.worker_config.active_worker_cache_pool()
        if cache_pool is not None:
            cache_pool.close()
        super().close()

    def __iter__(self) -> Iterator[T_sample]:
        yield from self.dataset

    def save_state(self) -> FlexState:
        # Just delegate, make self transparent
        return self.dataset.save_state()

    def restore_state(self, state: FlexState):
        # Just delegate, make self transparent
        return self.dataset.restore_state(state)

    def config(self) -> Dict[str, Any]:
        # Transparent logger, it won't change the samples
        return self.dataset.config()

    def __str__(self):
        return f"FileStoreInitWrapper(auxiliary_datasets={self.auxiliary_datasets}, dataset={self.dataset})"
