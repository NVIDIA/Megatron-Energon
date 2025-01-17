# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar, Union

import torch

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


@dataclass
class BlendDatasetState(State):
    #: States of the sub datasets
    datasets: List[State]
    #: State of the worker rng
    rng: WorkerRngState


@dataclass
class BlendDatasetMergedState(MergedState):
    #: States of the sub datasets
    datasets: List[MergedState]
    #: State of the worker rng
    rng: WorkerRngMergedState


class BlendDataset(BaseWrapperDataset[T_sample], Generic[T_sample]):
    """
    This dataset wrapper blends multiple iterable datasets together give a weighting.
    The datasets may be infinite. This dataset is always infinite.
    """

    dataset_weights: Tuple[Tuple[SavableDataset[T_sample], float], ...]

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

    def __len__(self) -> int:
        # Give the number of samples in inner datasets, disregarding the weight
        return sum(len(dataset) for dataset, weight in self.dataset_weights)

    def __iter__(self) -> Iterator[T_sample]:
        assert self.worker_has_samples(), "Cannot blend all empty datasets"
        datasets, weights = zip(
            *[
                (dataset, weight)
                for dataset, weight in self.dataset_weights
                if dataset.worker_has_samples()
            ]
        )
        dataset_iters = [iter(dataset) for dataset in datasets]
        weights = torch.tensor(weights, dtype=torch.float32)
        probs = weights / weights.sum()
        assert torch.all(probs > 0), "Negative weights are not allowed"

        while True:
            ds_idx = self._worker_rng.choice_idx(probs=probs)
            if dataset_iters[ds_idx] is None:
                if all(dataset_iter is None for dataset_iter in dataset_iters):
                    break
                continue
            try:
                sample = next(dataset_iters[ds_idx])
            except StopIteration:
                dataset_iters[ds_idx] = None
                probs[ds_idx] = 0
                if all(dataset_iter is None for dataset_iter in dataset_iters):
                    break
            else:
                yield add_sample_restore_key(sample, ds_idx, src=self)

    def worker_has_samples(self) -> bool:
        return any(dataset.worker_has_samples() for dataset, _weight in self.dataset_weights)

    def save_state(self) -> BlendDatasetState:
        return BlendDatasetState(
            datasets=[d.save_state() for d, _weight in self.dataset_weights],
            rng=self._worker_rng.save_state(),
        )

    def merge_states(self, states: List[BlendDatasetState]) -> BlendDatasetMergedState:
        assert all(s is None or isinstance(s, BlendDatasetState) for s in states)
        assert all(s is None or len(s.datasets) == len(self.dataset_weights) for s in states)
        return BlendDatasetMergedState(
            datasets=[
                d.merge_states([None if s is None else s.datasets[ds_idx] for s in states])
                for ds_idx, (d, _) in enumerate(self.dataset_weights)
            ],
            rng=self._worker_rng.merge_states([None if s is None else s.rng for s in states]),
        )

    def restore_state(self, state: Optional[BlendDatasetMergedState]) -> None:
        if state is None:
            for dataset, _weight in self.dataset_weights:
                dataset.restore_state(None)
            self._worker_rng.restore_state(None)
        else:
            assert isinstance(state, BlendDatasetMergedState)
            assert len(state.datasets) == len(
                self.dataset_weights
            ), f"The {len(state.datasets)} datasets to be restored do not match the configured {len(self.dataset_weights)} datasets"
            for (dataset, _weight), dstate in zip(self.dataset_weights, state.datasets):
                dataset.restore_state(dstate)
            self._worker_rng.restore_state(state.rng)

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
