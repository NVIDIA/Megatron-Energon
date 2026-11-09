# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Iterator, TypeVar

from megatron.energon.flavors.base_dataset import (
    FlexState,
    SavableDataset,
    add_sample_restore_key,
)
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class ConcatDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """
    This dataset wrapper concatenates multiple iterable datasets together. The datasets must be
    finite, otherwise not all datasets can be sampled. This is only useful for validation / test
    datasets.
    """

    #: Index of the dataset currently iterated, so that a restored state continues there.
    _dataset_idx: int
    #: Whether the state was restored since the last iteration started (not saved).
    _state_restored: bool

    _savable_fields = ("_dataset_idx",)

    def __init__(
        self,
        *datasets: SavableDataset[T_sample],
        worker_config: WorkerConfig,
    ):
        """Construct a concatenated dataset."""
        super().__init__(datasets, worker_config=worker_config)
        assert len(self) >= 0, "Datasets must be finite."
        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._dataset_idx = 0
        self._state_restored = False

    def restore_state(self, state: FlexState) -> None:
        super().restore_state(state)
        self._state_restored = True

    def len_worker(self, worker_idx: int | None = None) -> int:
        return sum(dataset.len_worker(worker_idx) for dataset in self.datasets)

    def __iter__(self) -> Iterator[T_sample]:
        if not self._state_restored:
            # A new iteration starts with the first dataset, even if the previous one was not
            # iterated to its end.
            self._dataset_idx = 0
        self._state_restored = False

        # Finished datasets reset themselves for their next pass, so continue with the current one.
        while self._dataset_idx < len(self.datasets):
            ds_idx = self._dataset_idx
            for sample in self.datasets[ds_idx]:
                yield add_sample_restore_key(
                    sample,
                    ds_idx,
                    src=self,
                )
            self._dataset_idx += 1
        self._dataset_idx = 0

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "datasets": [dataset.config() for dataset in self.datasets],
        }

    def __str__(self):
        return f"ConcatDataset(datasets={self.datasets})"
