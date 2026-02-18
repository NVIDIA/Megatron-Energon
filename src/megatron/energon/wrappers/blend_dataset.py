# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Iterator, List, Sequence, Tuple, TypeVar

import torch

from megatron.energon.flavors.base_dataset import SavableDataset, add_sample_restore_key
from megatron.energon.rng import WorkerRng
from megatron.energon.weights import WeightLike, eval_weight, weight_from_config, weight_to_config
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class BlendDataset(BaseWrapperDataset[T_sample, T_sample]):
    """
    Blend multiple iterable datasets according to their (possibly scheduled) weights.

    Samples are drawn from the inner datasets with probability proportional to their weights.
    Weights may be constants or schedules; schedules are evaluated per batch index (typically
    `WorkerConfig.active_worker_batch_index`) to support deterministic schedules.

    Notes:
    - If inner datasets are finite, this iterator stops once all inner iterators are exhausted.
      In the standard training pipeline, inner datasets are typically wrapped in `RepeatDataset`,
      making blending effectively infinite.
    - Weights must be >= 0; 0 disables sampling from a dataset (until the weight becomes > 0 again).
    - If all active datasets have weight 0 for a given batch index on a worker, iteration raises
      `RuntimeError`.
    """

    datasets: List[SavableDataset[T_sample]]
    weights: Tuple[WeightLike, ...]
    dataset_weights: Sequence[Tuple[SavableDataset[T_sample], WeightLike]]
    exhausted: List[bool]
    _worker_rng: WorkerRng

    _savable_fields = ("exhausted", "_worker_rng")

    def __init__(
        self,
        *dataset_weights: Tuple[SavableDataset[T_sample], WeightLike],
        worker_config: WorkerConfig,
    ):
        """Construct a BlendDataset.

        Args:
            dataset_weights: Each argument should be a tuple of (dataset, weight) with a weight
                >= 0. The output samples are sampled from the input datasets with probabilities
                proportional to these weights. Weights may be dynamic via weight schedules.
            worker_config: Configuration for the workers.
        """
        # datasets = [dataset for dataset, _weight in dataset_weights]
        self.datasets, self.weights = zip(*dataset_weights)

        super().__init__(self.datasets, worker_config=worker_config)

        self.dataset_weights = dataset_weights
        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._worker_rng = WorkerRng(self.worker_config)
        self.exhausted = [False] * len(self.weights)

    def len_worker(self, worker_idx: int | None = None) -> int:
        # Give the number of samples in inner datasets, disregarding the weight
        return sum(dataset.len_worker(worker_idx) for dataset in self.datasets)

    def __iter__(self) -> Iterator[T_sample]:
        assert self.worker_has_samples(), "Cannot blend all empty datasets"

        # Create a list of datasets and their weights, but
        # set the weight to 0 if the dataset has no samples on this worker.

        dataset_iters: list[Iterator[T_sample] | None] = []
        weights: list[WeightLike] = []
        for idx, (dataset, weight) in enumerate(self.dataset_weights):
            if dataset.worker_has_samples():
                dataset_iters.append(iter(dataset))
                weights.append(weight_from_config(weight))
            else:
                dataset_iters.append(None)
                weights.append(0.0)
                self.exhausted[idx] = True

        # Some may already be exhausted on this worker when restoring a state.
        for idx, exhausted in enumerate(self.exhausted):
            if exhausted:
                weights[idx] = 0.0
                dataset_iters[idx] = None

        cached_batch_idx: int | None = None
        cached_exhausted: tuple[bool, ...] | None = None
        cached_probs: torch.Tensor | None = None

        while True:
            active_wc = WorkerConfig.active_worker_config
            batch_idx = 0 if active_wc is None else active_wc.active_worker_batch_index

            exhausted_key = tuple(self.exhausted)
            if (
                cached_probs is None
                or cached_batch_idx != batch_idx
                or cached_exhausted != exhausted_key
            ):
                # Evaluate (potentially scheduled) weights for this batch index.
                evaluated = [
                    0.0 if it is None else eval_weight(w, batch_idx)
                    for w, it in zip(weights, dataset_iters)
                ]
                probs = torch.tensor(evaluated, dtype=torch.float32)
                if probs.sum().item() <= 0:
                    raise RuntimeError(
                        "BlendDataset has no active datasets with non-zero weight on this worker "
                        f"(batch_idx={batch_idx}). This may happen if all schedules evaluate to 0 "
                        "or the worker has no samples."
                    )
                cached_batch_idx = batch_idx
                cached_exhausted = exhausted_key
                cached_probs = probs

            ds_idx = self._worker_rng.choice_idx(probs=cached_probs)

            if dataset_iters[ds_idx] is None:
                if all(dataset_iter is None for dataset_iter in dataset_iters):
                    break
                continue
            try:
                sample = next(dataset_iters[ds_idx])
            except StopIteration:
                dataset_iters[ds_idx] = None
                weights[ds_idx] = 0.0
                self.exhausted[ds_idx] = True
                if all(dataset_iter is None for dataset_iter in dataset_iters):
                    break
            else:
                yield add_sample_restore_key(sample, ds_idx, src=self)

        self.exhausted = [False] * len(self.dataset_weights)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset_weights": [
                (dataset.config(), weight_to_config(weight))
                for dataset, weight in self.dataset_weights
            ],
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"BlendDataset(dataset_weights={self.dataset_weights})"
