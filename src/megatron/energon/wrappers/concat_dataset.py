# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterator, TypeVar

from megatron.energon.flavors.base_dataset import RestoreKey, SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import (
    BaseWrapperDataset,
    WrappedRestoreKey,
    wrap_sample_restore_key,
)

T_sample = TypeVar("T_sample")


@dataclass(kw_only=True, slots=True, frozen=True)
class ConcatRestoreKey(WrappedRestoreKey):
    dataset_idx: int


class ConcatDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """
    This dataset wrapper concatenates multiple iterable datasets together. The datasets must be
    finite, otherwise not all datasets can be sampled. This is only useful for validation / test
    datasets.
    """

    def __init__(
        self,
        *datasets: SavableDataset[T_sample],
        worker_config: WorkerConfig,
    ):
        """Construct a concatenated dataset."""
        super().__init__(datasets, worker_config=worker_config)
        assert len(self) >= 0, "Datasets must be finite."

    def reset_state_own(self) -> None:
        return

    def len_worker(self, worker_idx: int | None = None) -> int:
        return sum(dataset.len_worker(worker_idx) for dataset in self.datasets)

    def __iter__(self) -> Iterator[T_sample]:
        for ds_idx, dataset in enumerate(self.datasets):
            for sample in dataset:
                yield wrap_sample_restore_key(
                    sample,
                    ConcatRestoreKey,
                    dataset_idx=ds_idx,
                )

    def restore_sample(self, restore_key: RestoreKey) -> T_sample:
        assert isinstance(restore_key, ConcatRestoreKey)
        return self.datasets[restore_key.dataset_idx].restore_sample(restore_key.inner)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "datasets": [dataset.config() for dataset in self.datasets],
        }

    def __str__(self):
        return f"ConcatDataset(datasets={self.datasets})"
