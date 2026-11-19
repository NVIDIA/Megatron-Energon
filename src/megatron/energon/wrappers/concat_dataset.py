# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterator, TypeVar

from megatron.energon.flavors.base_dataset import (
    FlexState,
    RestoreKey,
    SavableDataset,
    set_sample_restore_key,
)
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import (
    BaseWrapperDataset,
    WrappedRestoreKey,
    get_sample_restore_key,
)

T_sample = TypeVar("T_sample")


@dataclass(frozen=True, slots=True, kw_only=True)
class ConcatRestoreKey(WrappedRestoreKey):
    """A sample selected from one child of ConcatDataset."""

    dataset_idx: int


class ConcatDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """
    This dataset wrapper concatenates multiple iterable datasets together. The datasets must be
    finite, otherwise not all datasets can be sampled. This is only useful for validation / test
    datasets.
    """

    _worker_local_fields = {"_state_restored": bool}

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
                inner = get_sample_restore_key(sample)
                yield set_sample_restore_key(
                    sample,
                    None if inner is None else ConcatRestoreKey(dataset_idx=ds_idx, inner=inner),
                )
            self._dataset_idx += 1
        self._dataset_idx = 0

    def restore_sample(self, restore_key: RestoreKey) -> T_sample:
        assert isinstance(restore_key, ConcatRestoreKey)
        key = restore_key
        sample = self.datasets[key.dataset_idx].restore_sample(key.inner)
        return set_sample_restore_key(sample, key)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "datasets": [dataset.config() for dataset in self.datasets],
        }

    def __str__(self):
        return f"ConcatDataset(datasets={self.datasets})"
