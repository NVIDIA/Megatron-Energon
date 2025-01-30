# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import MergedState, SavableDataset, State
from megatron.energon.flavors.webdataset.itar_reader import ITarReader
from megatron.energon.flavors.webdataset.structs import FilteredSample, reraise_exception
from megatron.energon.rng import WorkerRng, WorkerRngMergedState, WorkerRngState
from megatron.energon.worker import WorkerConfig


@dataclass(slots=True)
class ITarSampleLoaderState(State):
    #: Rng state
    rng: WorkerRngState
    # TODO


@dataclass(slots=True)
class ITarSampleLoaderMergedState(MergedState):
    #: Rng state
    rng: WorkerRngMergedState
    # TODO


class ITarSampleLoaderDataset(SavableDataset[Tuple[Optional[FilteredSample], ...]]):
    """Internal class for loading samples from an indexed webdataset."""

    itar_readers: List[ITarReader]

    # Sample keys to ignore
    exclude: Set[str]

    # If = 1, every sample is seen exactly once per epoch. If > 1, samples
    # (or rather shard slices) are shuffled within this number of epochs (i.e. randomly
    # selected without replacement). If None, the shards are effectively shuffle over
    # infinite epochs (i.e. shard slices are drawn with replacement).
    shuffle_over_epochs: Optional[int]
    # Number of parallel iterators to be opened simultaneously (and random sample between them)
    parallel_iter_count: int
    parallel_iters: List[Any]
    # Error handler
    handler: Callable[[Exception, Optional[str]], None]

    # Worker's random generator
    _worker_rng: WorkerRng

    def __init__(
        self,
        itar_readers: List[ITarReader],
        local_worker_sample_split_offsets: List[int],
        *,
        worker_config: WorkerConfig,
        exclude: Set[str],
        shuffle_over_epochs: Optional[int] = None,
        parallel_iter_count: int = 1,
        handler: Callable[[Exception, Optional[str]], None] = reraise_exception,
    ):
        """
        The webdataset loader. Iterates over the shard infos and yields the samples.

        Args:
            index: The index file of the webdataset.
            indexed_datasets: Factories for the indexed datasets, used to get meta info.
            worker_config: The worker configuration.
            exclude: A set of strings of the form "<shard name>" or "<shard name>/<sample index>" to
                exclude from iteration.
            shuffle_over_epochs: If None, disable shuffling.
                If = 1, every sample is seen exactly once per epoch.
                If > 1, samples (or rather shard slices) are shuffled within this number of epochs
                (i.e. randomly selected without replacement).
                If -1, the shards are effectively shuffle over infinite epochs (i.e. shard slices
                are drawn with replacement).
            parallel_iter_count: If > 1, samples are randomly drawn from parallel shard iterators.
                This will not impact performance, but increase randomness. If = 1, the shards are
                iterated in order.
            handler: Exception handler. Args: (exception, key).
        """
        super().__init__(worker_config=worker_config)
        self.itar_readers = itar_readers
        self.local_worker_sample_split_offsets = local_worker_sample_split_offsets
        self.exclude = exclude
        self.shuffle_over_epochs = shuffle_over_epochs
        self.parallel_iter_count = parallel_iter_count
        self.handler = handler
        self._worker_rng = WorkerRng(worker_config)

        assert shuffle_over_epochs is None or shuffle_over_epochs == -1 or shuffle_over_epochs >= 1
        assert self.parallel_iter_count >= 1

    def __len__(self) -> int:
        # TODO: This is only for the current rank, is this the right number?
        return len(self.local_worker_sample_split_offsets) - 1

    def worker_has_samples(self) -> bool:
        self.worker_config.assert_worker()
        worker_idx = self.worker_config.rank_worker_id()

        worker_range_start = self.local_worker_sample_split_offsets[worker_idx]
        worker_range_end = self.local_worker_sample_split_offsets[worker_idx + 1]
        worker_sample_cnt = worker_range_end - worker_range_start
        return worker_sample_cnt > 0

    def __iter__(self) -> Iterator[Tuple[Optional[FilteredSample], ...]]:
        self.worker_config.assert_worker()
        worker_idx = self.worker_config.rank_worker_id()

        # Slice itar datasets according to local worker ranges
        worker_range_start = self.local_worker_sample_split_offsets[worker_idx]
        worker_range_end = self.local_worker_sample_split_offsets[worker_idx + 1]
        worker_itar_readers = [
            itar_readers[worker_range_start:worker_range_end] for itar_readers in self.itar_readers
        ]

        if self.worker_config.should_log(level=1):
            self.worker_config.worker_log(
                {
                    "t": "ITarSampleLoaderDataset.__iter__",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "shard_range": [
                        # TODO
                    ],
                    "parallel_shard_iters": self.parallel_iters,
                    "shuffle_over_epochs": self.shuffle_over_epochs,
                }
            )

        if len(worker_itar_readers[0]) == 0:
            return

        # TODO: shuffling, parallel shard iters
        for sample_idx in range(len(worker_itar_readers[0])):
            yield tuple(itar_reader[sample_idx] for itar_reader in worker_itar_readers)

    def can_restore_sample(self) -> bool:
        return True

    def assert_can_restore(self) -> None:
        pass

    def restore_sample(
        self, key: Tuple[Union[str, int, tuple], ...]
    ) -> Tuple[Optional[FilteredSample], ...]:
        id, *shard_data = key
        assert id == "itar"
        assert isinstance(shard_data[0], str)

        # TODO
        return (None,)

    def save_state(self) -> ITarSampleLoaderState:
        self.worker_config.assert_worker()
        worker_idx = self.worker_config.rank_worker_id()

        # TODO
        return ITarSampleLoaderState(
            rng=self._worker_rng.save_state(),
        )

    def merge_states(self, states: List[ITarSampleLoaderState]) -> ITarSampleLoaderMergedState:
        assert all(s is None or isinstance(s, ITarSampleLoaderState) for s in states)
        # TODO
        return ITarSampleLoaderMergedState(
            rng=self._worker_rng.merge_states([s.rng for s in states if s is not None]),
        )

    def restore_state(self, state: Optional[ITarSampleLoaderMergedState]) -> None:
        # TODO
        pass

    def config(self) -> Dict[str, Any]:
        return {  # TODO
            "type": type(self).__qualname__,
            "samples": [],
            "worker_config": self.worker_config.config(),
            "exclude": list(self.exclude),
            "shuffle_over_epochs": self.shuffle_over_epochs,
            "parallel_shard_iters": self.parallel_iters,
        }

    def __str__(self):
        # TODO
        return f"ITarSampleLoaderDataset(shards=TODO, shuffle_over_epochs={self.shuffle_over_epochs}, parallel_shard_iters={self.parallel_iters})"
