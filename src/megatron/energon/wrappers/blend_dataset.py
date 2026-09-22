# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, Iterator, Sequence, Tuple, TypeVar

import torch

from megatron.energon.flavors.base_dataset import SavableDataset, add_sample_restore_key
from megatron.energon.rng import WorkerRng
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class BlendDataset(BaseWrapperDataset[T_sample, T_sample]):
    """
    This dataset wrapper blends multiple iterable datasets together give a weighting.
    The datasets may be infinite. This dataset is always infinite.
    """

    datasets: tuple[SavableDataset[T_sample], ...]
    weights: tuple[float, ...]
    dataset_weights: Sequence[tuple[SavableDataset[T_sample], float]]
    exhausted: list[bool]
    _worker_rng: WorkerRng
    sample_size_fn: Callable[[T_sample], int | float] | None
    sample_size_alpha: float
    sample_size_epsilon: float
    _emitted_sizes: list[float]

    _savable_fields = ("exhausted", "_worker_rng", "_emitted_sizes")

    def __init__(
        self,
        *dataset_weights: Tuple[SavableDataset[T_sample], float],
        worker_config: WorkerConfig,
        sample_size_fn: Callable[[T_sample], int | float] | None = None,
        sample_size_alpha: float = 1.0,
        sample_size_epsilon: float = 1.0,
    ):
        """Construct a BlendDataset.

        Args:
            dataset_weights: Each argument should be a tuple of (dataset, weight) with a weight
                between 0 and 1. The output samples are sampled from the input datasets with the
                given probabilities.
            worker_config: Configuration for the workers.
            sample_size_fn: Optional function returning the accounting size of a sample. When set,
                sampling uses soft size-deficit regulation toward the target weight distribution.
            sample_size_alpha: Exponent for size-deficit credits. Higher values favor datasets
                furthest behind their target size share.
            sample_size_epsilon: Floor added to credits so every active dataset keeps non-zero
                selection probability.
        """
        self.datasets, self.weights = zip(*dataset_weights)

        super().__init__(self.datasets, worker_config=worker_config)

        self.dataset_weights = dataset_weights
        self.sample_size_fn = sample_size_fn
        self.sample_size_alpha = sample_size_alpha
        self.sample_size_epsilon = sample_size_epsilon
        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._worker_rng = WorkerRng(self.worker_config)
        n = len(self.weights)
        self.exhausted = [False] * n
        self._emitted_sizes = [0.0] * n

    def len_worker(self, worker_idx: int | None = None) -> int:
        # Give the number of samples in inner datasets, disregarding the weight
        return sum(dataset.len_worker(worker_idx) for dataset in self.datasets)

    def _compute_size_probs(
        self,
        weights: torch.Tensor,
        emitted: torch.Tensor,
    ) -> torch.Tensor:
        target_weights = weights / weights.sum()
        deficits = target_weights * emitted.sum() - emitted
        credits = torch.clamp(deficits, min=0.0)
        probs = target_weights * (self.sample_size_epsilon + credits) ** self.sample_size_alpha
        if probs.sum() == 0:
            probs = weights
        return probs

    def __iter__(self) -> Iterator[T_sample]:
        assert self.worker_has_samples(), "Cannot blend all empty datasets"

        # Create a list of datasets and their weights, but
        # set the weight to 0 if the dataset has no samples on this worker.

        dataset_iters: list[Iterator[T_sample] | None] = []
        weights_list: list[float] = []
        for idx, (dataset, weight) in enumerate(self.dataset_weights):
            assert weight > 0, "All blending weights must be > 0"

            if dataset.worker_has_samples():
                dataset_iters.append(iter(dataset))
                weights_list.append(weight)
            else:
                dataset_iters.append(None)
                weights_list.append(0)
                self.exhausted[idx] = True

        weights = torch.tensor(weights_list, dtype=torch.float32)
        if weights.sum() == 0:
            raise RuntimeError(
                "There is a worker with no samples in any of the blended datasets. "
                "This can happen if you have a lot of workers and your dataset is too small. "
                "Currently this case is not supported."
            )

        # Some may already be exhausted on this worker when restoring a state.
        for idx, exhausted in enumerate(self.exhausted):
            if exhausted:
                weights[idx] = 0
                dataset_iters[idx] = None

        if self.sample_size_fn is not None:
            emitted = torch.tensor(self._emitted_sizes, dtype=torch.float64)

        while True:
            if self.sample_size_fn is None:
                ds_probs = weights
            else:
                ds_probs = self._compute_size_probs(weights, emitted)

            ds_idx = self._worker_rng.choice_idx(probs=ds_probs)

            dataset_iter = dataset_iters[ds_idx]
            if dataset_iter is None:
                if all(dataset_iter is None for dataset_iter in dataset_iters):
                    break
                continue
            try:
                sample = next(dataset_iter)
            except StopIteration:
                dataset_iters[ds_idx] = None
                weights[ds_idx] = 0
                self.exhausted[ds_idx] = True
                if all(dataset_iter is None for dataset_iter in dataset_iters):
                    break
            else:
                if self.sample_size_fn is not None:
                    size = float(self.sample_size_fn(sample))
                    assert size >= 0, "sample_size_fn must return a non-negative value"
                    self._emitted_sizes[ds_idx] += size
                    emitted[ds_idx] += size
                yield add_sample_restore_key(sample, ds_idx, src=self)

        self.exhausted = [False] * len(self.dataset_weights)

    def config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "type": type(self).__qualname__,
            "dataset_weights": [
                (dataset.config(), weight) for dataset, weight in self.dataset_weights
            ],
            "worker_config": self.worker_config.config(),
        }
        if self.sample_size_fn is not None:
            cfg["sample_size_fn"] = self._function_config(self.sample_size_fn)
            cfg["sample_size_alpha"] = self.sample_size_alpha
            cfg["sample_size_epsilon"] = self.sample_size_epsilon
        return cfg

    def __str__(self):
        return f"BlendDataset(dataset_weights={self.dataset_weights})"
