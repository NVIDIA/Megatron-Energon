# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Iterator, TypeVar

from megatron.energon.flavors.base_dataset import (
    SavableDataset,
    add_sample_restore_key,
)
from megatron.energon.flavors.trace import TraceIter, trace_iter
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


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

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    @trace_iter(
        name=lambda self: "ConcatDataset",
        call_args={
            "config": lambda self: self._own_config(),
        },
    )
    def __iter__(self, trace_iter: TraceIter) -> Iterator[T_sample]:
        for ds_idx, dataset in enumerate(self.datasets):
            for sample in dataset:
                yield add_sample_restore_key(
                    sample,
                    ds_idx,
                    src=self,
                )

    def _own_config(self) -> Dict[str, Any]:
        return {}

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "datasets": [dataset.config() for dataset in self.datasets],
        }

    def __str__(self):
        return f"ConcatDataset(datasets={self.datasets})"
