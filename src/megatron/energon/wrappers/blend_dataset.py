# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar, Union

import torch

from megatron.energon.dataclass_slots import dataclass_slots
from megatron.energon.flavors.base_dataset import (
    MergedState,
    SavableDataset,
    State,
    add_sample_restore_key,
)
from megatron.energon.rng import WorkerRng, WorkerRngMergedState, WorkerRngState
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


@dataclass_slots
class BlendDatasetState(State):
    #: States of the sub datasets
    datasets: List[State]
    #: Whether the subdatasets are done
    exhausted: List[bool]
    #: State of the worker rng
    rng: WorkerRngState


"""
exhausted=list(self.exhausted[self.worker_config.rank_worker_id()]),
rng=self._worker_rng.save_state(),
"""


class BlendDataset(BaseWrapperDataset[T_sample], Generic[T_sample]):
    """
    This dataset wrapper blends multiple iterable datasets together give a weighting.
    The datasets may be infinite. This dataset is always infinite.
    """

    dataset_weights: Tuple[Tuple[SavableDataset[T_sample], float], ...]
    exhausted: List[List[bool]]
    _worker_rng: WorkerRng

    def __init__(
        self,
        *dataset_weights: Tuple[SavableDataset[T_sample], float],
        worker_config: WorkerConfig,
    ):
        """Construct a BlendDataset.

        Args:
            dataset_weights: Each argument should be a tuple of (dataset, weight) with a weight
                between 0 and 1. The output samples are sampled from the input datasets with the
                given probabilities.
            worker_config: Configuration for the workers.
        """
        datasets = [dataset for dataset, _weight in dataset_weights]
        super().__init__(datasets, worker_config=worker_config)

        self.dataset_weights = dataset_weights
        self._worker_rng = WorkerRng(self.worker_config)
        self.exhausted = [
            [False] * len(dataset_weights) for _ in range(max(self.worker_config.num_workers, 1))
        ]

    def __len__(self) -> int:
        # Give the number of samples in inner datasets, disregarding the weight
        return sum(len(dataset) for dataset, weight in self.dataset_weights)

    def __iter__(self) -> Iterator[T_sample]:
        assert self.worker_has_samples(), "Cannot blend all empty datasets"

        # Create a list of datasets and their weights, but
        # set the weight to 0 if the dataset has no samples on this worker.

        dataset_iters = []
        weights = []
        for idx, (dataset, weight) in enumerate(self.dataset_weights):
            assert weight > 0, "All blending weights must be > 0"

            if dataset.worker_has_samples():
                dataset_iters.append(iter(dataset))
                weights.append(weight)
            else:
                dataset_iters.append(None)
                weights.append(0)

        weights = torch.tensor(weights, dtype=torch.float32)
        if weights.sum() == 0:
            raise RuntimeError(
                "There is a worker with no samples in any of the blended datasets. "
                "This can happen if you have a lot of workers and your dataset is too small. "
                "Currently this case is not supported."
            )

        # Some may already be exhausted on this worker when restoring a state.
        for idx, exhausted in enumerate(self.exhausted[self.worker_config.rank_worker_id()]):
            if exhausted:
                weights[idx] = 0
                dataset_iters[idx] = None

        while True:
            ds_idx = self._worker_rng.choice_idx(probs=weights)

            if dataset_iters[ds_idx] is None:
                if all(dataset_iter is None for dataset_iter in dataset_iters):
                    break
                continue
            try:
                sample = next(dataset_iters[ds_idx])
            except StopIteration:
                dataset_iters[ds_idx] = None
                weights[ds_idx] = 0
                self.exhausted[self.worker_config.rank_worker_id()][ds_idx] = True
                if all(dataset_iter is None for dataset_iter in dataset_iters):
                    break
            else:
                yield add_sample_restore_key(sample, ds_idx, src=self)

        self.exhausted[self.worker_config.rank_worker_id()] = [False] * len(self.dataset_weights)

    def worker_has_samples(self) -> bool:
        return any(dataset.worker_has_samples() for dataset, _weight in self.dataset_weights)

    def restore_state(self, state: Optional[BlendDatasetMergedState]) -> None:
        if state is None:
            for dataset, _weight in self.dataset_weights:
                dataset.restore_state(None)
            self._worker_rng.restore_state(None)
            self.exhausted = [
                [False] * len(self.dataset_weights)
                for _ in range(max(self.worker_config.num_workers, 1))
            ]
        else:
            assert isinstance(state, BlendDatasetMergedState)
            assert len(state.datasets) == len(self.dataset_weights), (
                f"The {len(state.datasets)} datasets to be restored do not match the configured {len(self.dataset_weights)} datasets"
            )
            for (dataset, _weight), dstate in zip(self.dataset_weights, state.datasets):
                dataset.restore_state(dstate)
            self._worker_rng.restore_state(state.rng)
            # Need [list() for ...] to create a deep copy
            self.exhausted = [list(sub) for sub in state.exhausted]

    def can_restore_sample(self) -> bool:
        return all(dataset.can_restore_sample() for dataset, _weight in self.dataset_weights)

    def assert_can_restore(self) -> None:
        for dataset, _weight in self.dataset_weights:
            dataset.assert_can_restore()

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        id, ds_idx = index[:2]
        assert id == type(self).__name__
        index = index[2:]
        assert isinstance(ds_idx, int)
        return add_sample_restore_key(
            self.dataset_weights[ds_idx][0].restore_sample(index),
            ds_idx,
            src=self,
        )

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset_weights": [
                (dataset.config(), weight) for dataset, weight in self.dataset_weights
            ],
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"BlendDataset(dataset_weights={self.dataset_weights})"
