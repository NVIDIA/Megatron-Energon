# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch

from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import MergedState, SavableDataset, State
from megatron.energon.flavors.webdataset.itar_dataset import ITarDataset
from megatron.energon.flavors.webdataset.structs import FilteredSample, reraise_exception
from megatron.energon.rng import WorkerRng, WorkerRngMergedState, WorkerRngState
from megatron.energon.worker import WorkerConfig


@dataclass(slots=True)
class SliceState:
    """Represents the iteration state of a single shard slice to the index."""

    #: The slice index of this slice state
    index: int
    #: The actual state: The global sample offset (`slice[index] <= offset < slice[index + 1]``)
    current: int


@dataclass(slots=True)
class RawSampleData:
    """Represents the iteration state of a single shard slice to the index."""

    #: Index of the sample. This is also the restore key
    __restore_key__: Tuple[str, int]
    #: The sample data
    data: Tuple[Optional[FilteredSample], ...]


@dataclass(slots=True)
class SampleLoaderState(State):
    """
    The savable state for the wds sample loader. Contains the active and pending shards.
    """

    #: Rng state
    rng: WorkerRngState
    #: The seed that was used to generate the current pending shards
    pending_slices_seed: WorkerRngState
    #: The number of shards that have already been opened / processed and thus been removed from the
    # pending shards. None if starting a new epoch.
    # Pending shards are the shards which have not yet been opened, but should be processed in the
    # current epoch.
    pending_slices_offset: Optional[int]
    #: The active shards are the currently opened shards. May contain `None`, if there are fewer
    # shards available (i.e. pending_shards empty) than parallel shard iterators requested.
    active_slices: Optional[List[Optional[SliceState]]]
    #: The total number of samples retrieved, it's just a monotonically increasing counter
    sample_count: int
    #: Number of epochs this dataset has been iterated over
    epoch_count: int
    #: Number of samples retrieved in current epoch
    epoch_sample_count: int


@dataclass(slots=True)
class SampleLoaderMergedState(MergedState):
    #: Rng state
    rng: WorkerRngMergedState
    #: The seed that was used to generate the current pending shards
    pending_slices_seed: WorkerRngMergedState
    #: The number of shards that have already been opened / processed and thus been removed from the
    # pending shards. None if starting a new epoch.
    # Pending shards are the shards which have not yet been opened, but should be processed in the
    # current epoch.
    pending_slices_offset: List[Optional[int]]
    #: The active shards are the currently opened shards. May contain `None`, if there are fewer
    # shards available (i.e. pending_shards empty) than parallel shard iterators requested.
    active_slices: List[Optional[List[Optional[SliceState]]]]
    #: The total number of samples retrieved, it's just a monotonically increasing counter
    sample_count: List[int]
    #: Number of epochs this dataset has been iterated over
    epoch_count: List[int]
    #: The number of samples retrieved in current epoch
    epoch_sample_count: List[int]


class WebdatasetSampleLoaderDataset(SavableDataset[RawSampleData]):
    """Internal class for loading samples from webdataset shards"""

    #: The readers for each joined dataset
    join_readers: Sequence[ITarDataset]
    #: The offsets of the shard slices to iterate over
    worker_slice_offsets: Sequence[Sequence[int]]
    # Sample keys to ignore
    exclude: Set[str]

    # If = 1, every sample is seen exactly once per epoch. If > 1, samples
    # (or rather shard slices) are shuffled within this number of epochs (i.e. randomly
    # selected without replacement). If None, the shards are effectively shuffle over
    # infinite epochs (i.e. shard slices are drawn with replacement).
    shuffle_over_epochs: Optional[int]
    # Number of parallel iterators to be opened simultaneously (and random sample between them)
    parallel_shard_iters: int

    # Error handler
    handler: Callable[[Exception, Optional[str]], None]

    # Worker's random generator
    _worker_rng: WorkerRng
    #: The seed to be used for regenerating the pending shards
    _pending_shards_seed: WorkerRngMergedState
    #: The number of shards that have already been opened / processed and thus been removed from the
    # pending shards.
    _pending_shards_offset: List[Optional[int]]
    #: Pending shards are the shards which have not yet been opened, but should be processed
    # in the current "epoch". If None, regenerate from the seed and offset.
    _pending_slice_indexes: List[Optional[List[int]]]
    #: The active shards are the currently opened shards. May contain `None`, if there are fewer
    # shards available (i.e. pending_shards empty) than parallel shard iterators requested.
    _active_slice_state: List[List[Optional[SliceState]]]
    #: The total number of samples retrieved, it's just a monotonically increasing counter
    _sample_count: List[int]
    #: Number of epochs this dataset has been iterated over
    _epoch_count: List[int]
    #: The number of samples retrieved in current epoch
    _epoch_sample_count: List[int]

    def __init__(
        self,
        join_readers: Sequence[ITarDataset],
        workers_sample_slice_offsets: Sequence[Sequence[int]],
        *,
        worker_config: WorkerConfig,
        exclude: Set[str],
        part_filter: Optional[Callable[[str], bool]] = None,
        shuffle_over_epochs: Optional[int] = None,
        parallel_shard_iters: int = 1,
        handler: Callable[[Exception, Optional[str]], None] = reraise_exception,
    ):
        """
        The webdataset loader. Iterates over the shard infos and yields the samples.

        Args:
            join_readers: A sequence of the joined readers (or just a single shard) to iterate over.
            worker_slice_offsets: The offsets of the shard slices to iterate over, for each worker.
            worker_config: The worker configuration.
            exclude: A set of strings of the form "<shard name>" or "<shard name>/<sample index>" to
                exclude from iteration.
            part_filter: If not None, use this function to filter out wds files.
            shuffle_over_epochs: If None, disable shuffling.
                If = 1, every sample is seen exactly once per epoch.
                If > 1, samples (or rather shard slices) are shuffled within this number of epochs
                (i.e. randomly selected without replacement).
                If -1, the shards are effectively shuffle over infinite epochs (i.e. shard slices
                are drawn with replacement).
            parallel_shard_iters: If > 1, samples are randomly drawn from parallel shard iterators.
                This will not impact performance, but increase randomness. If = 1, the shards are
                iterated in order.
            handler: Exception handler. Args: (exception, key).
        """
        super().__init__(worker_config=worker_config)
        self.join_readers = join_readers
        self.worker_slice_offsets = workers_sample_slice_offsets
        self.exclude = exclude
        self.part_filter = part_filter
        self.shuffle_over_epochs = shuffle_over_epochs
        self.parallel_shard_iters = parallel_shard_iters
        self.handler = handler
        self._worker_rng = WorkerRng(worker_config)
        n_workers = max(1, worker_config.num_workers)
        self._pending_slice_indexes = [None] * n_workers
        self._pending_shards_offset = [None] * n_workers
        self._pending_shards_seed = WorkerRngMergedState(rng=[None] * n_workers)
        self._active_slice_state = [[None] * parallel_shard_iters for _ in range(n_workers)]
        self._sample_count = [0] * n_workers
        self._epoch_count = [0] * n_workers
        self._epoch_sample_count = [0] * n_workers
        assert shuffle_over_epochs is None or shuffle_over_epochs == -1 or shuffle_over_epochs >= 1
        assert self.parallel_shard_iters >= 1

    def _get_sample(self, index: int) -> RawSampleData:
        return RawSampleData(
            __restore_key__=("Webdataset", index),
            data=tuple(reader[index] for reader in self.join_readers),
        )

    def _slices_once(self) -> List[int]:
        """Yields the indexes to slice offsets once. Possibly shuffles the list."""
        worker_idx = self.worker_config.rank_worker_id()
        slice_offsets = self.worker_slice_offsets[worker_idx]
        num_slices = len(slice_offsets) - 1
        shards_offset = self._pending_shards_offset[worker_idx]

        if self.shuffle_over_epochs is None:
            # No shuffling
            res_list = list(range(num_slices))
            if shards_offset is None:
                shards_offset = 0
        else:
            # Restore state or start new (and save)
            if shards_offset is None:
                # Start new state. First, save the state to restore the same order.
                self._pending_shards_seed.rng[worker_idx] = self._worker_rng.save_state().rng
                rng = self._worker_rng
                shards_offset = 0
            else:
                # Restore the state. Create a dedicated rng for this, as the main rng is in the
                # state for iterating from the next iterator.
                rng = WorkerRng(self.worker_config)
                rng.restore_state(self._pending_shards_seed)

            if self.shuffle_over_epochs == -1:
                # Shuffle with replacement (i.e. infinite epochs), effectively return as many shards
                # as are required for parallel shard iterators.
                # Next shards are drawn in the _shards_iter function.
                res_list = [rng.randbelow(num_slices) for _ in range(self.parallel_shard_iters)]
            elif self.shuffle_over_epochs >= 1:
                # Shuffle without replacement (potentially over multiple epochs)
                res_list = rng.shuffle(list(range(num_slices)) * self.shuffle_over_epochs)
            else:
                raise ValueError(f"Invalid shuffle_over_epochs: {self.shuffle_over_epochs}")
        # Reverse, such that pop returns the first element (in O(1) time)
        res_list.reverse()
        # Skip restored shard list already processed shards
        assert shards_offset is not None
        self._pending_shards_offset[worker_idx] = shards_offset
        if shards_offset > 0:
            # Those have already been popped in the current state
            del res_list[-shards_offset:]
        # Set the pending shards
        self._pending_slice_indexes[worker_idx] = res_list
        return res_list

    def _shards_iter(self) -> Generator[RawSampleData, None, None]:
        """Iterates the samples in a list of shards, possibly using multiple parallel iterators over
        the shards."""
        worker_idx = self.worker_config.rank_worker_id()
        # Cleanup other states
        for i in range(self.worker_config.num_workers):
            if i != worker_idx:
                self._active_slice_state[i] = [None] * self.parallel_shard_iters
                self._pending_slice_indexes[i] = None
        active_slice_probs = torch.zeros(self.parallel_shard_iters, dtype=torch.float32)
        active_slices = self._active_slice_state[worker_idx]
        pending_slice_indexes = self._pending_slice_indexes[worker_idx]
        slice_offsets = self.worker_slice_offsets[worker_idx]

        def slice_at(idx: int) -> SliceState:
            return SliceState(
                index=idx,
                current=slice_offsets[idx],
            )

        # Weight the shards by their size to get a more even distribution of samples
        if (
            any(s is not None for s in active_slices)
            or self._pending_shards_offset[worker_idx] is not None
        ):
            # Having an active state, or pending shards. This means we are resuming an epoch.
            if pending_slice_indexes is None:
                # Need to restore the pending shards
                pending_slice_indexes = self._slices_once()
            assert pending_slice_indexes is not None

            # Restore the state
            assert len(active_slices) == self.parallel_shard_iters
            for idx, shard_state in enumerate(active_slices):
                if shard_state is not None:
                    active_slice_probs[idx] = (
                        slice_offsets[shard_state.index + 1] - slice_offsets[shard_state.index]
                    )

            if self.worker_config.should_log(level=1):
                self.worker_config.worker_log(
                    {
                        "t": "WebdatasetSampleLoaderDataset._shards_iter.resume_epoch",
                        "r": self.worker_config.rank,
                        "w": self.worker_config.rank_worker_id(),
                        "pending_slice_indexes": pending_slice_indexes,
                        "active_shards": [
                            (
                                None
                                if state is None
                                else {
                                    "index": state.index,
                                    "current": state.current,
                                }
                            )
                            for state in active_slices
                        ],
                        "count": self._sample_count[worker_idx],
                        "epoch": self._epoch_count[worker_idx],
                        "epoch_count": self._epoch_sample_count[worker_idx],
                        "probs": active_slice_probs.tolist(),
                    }
                )

        else:
            # Start a new epoch
            assert pending_slice_indexes is None
            pending_slice_indexes = self._slices_once()

            if self.worker_config.should_log(level=1):
                self.worker_config.worker_log(
                    {
                        "t": "WebdatasetSampleLoaderDataset._shards_iter.next_epoch",
                        "r": self.worker_config.rank,
                        "w": self.worker_config.rank_worker_id(),
                        "pending_slice_indexes": pending_slice_indexes,
                        "count": self._sample_count[worker_idx],
                        "epoch": self._epoch_count[worker_idx],
                        "epoch_count": self._epoch_sample_count[worker_idx],
                        "probs": active_slice_probs.tolist(),
                        "shuffle_over_epochs": self.shuffle_over_epochs,
                    }
                )

            # List of shard iterators, always of length `parallel_shard_iters`. May contain `None`.
            active_slices.clear()
            # Fill up the shard iterators
            while len(pending_slice_indexes) > 0 and len(active_slices) < self.parallel_shard_iters:
                slice_index = pending_slice_indexes.pop()
                self._pending_shards_offset[worker_idx] += 1
                shard_state = slice_at(slice_index)
                active_slice_probs[len(active_slices)] = (
                    slice_offsets[shard_state.index + 1] - slice_offsets[shard_state.index]
                )
                active_slices.append(shard_state)
            # Fill up the shard iterators with None
            for _ in range(len(active_slices), self.parallel_shard_iters):
                active_slices.append(None)

        # print(
        #     f"Next shard iters generated for {self.worker_config.rank}:{self.worker_config.rank_worker_id()}: probs={shards_probs}"
        # )

        # Iterate over the shard iterators while there is an iterator left
        while torch.count_nonzero(active_slice_probs).item() > 0:
            if self.shuffle_over_epochs is None:
                # No shuffling, deterministic order, always the same
                assert self.parallel_shard_iters == 1
                shard_idx = 0
            else:
                # Take a random shard iterator
                shard_idx = self._worker_rng.choice_idx(active_slice_probs)
            shard_state = active_slices[shard_idx]
            assert shard_state is not None
            sample = self._get_sample(shard_state.current)
            shard_state.current += 1
            if shard_state.current >= slice_offsets[shard_state.index + 1]:
                # Iterator exhausted -> take next / remove from list
                if len(pending_slice_indexes) > 0 or self.shuffle_over_epochs == -1:
                    if len(pending_slice_indexes) > 0:
                        # Take the next shard (without replacement)
                        next_idx = pending_slice_indexes.pop()
                        assert self._pending_shards_offset[worker_idx] is not None
                        self._pending_shards_offset[worker_idx] += 1
                    else:
                        # Randomly select a new shard directly (with replacement)
                        next_idx = self._worker_rng.randbelow(len(slice_offsets))
                    shard_state = slice_at(next_idx)
                    active_slice_probs[shard_idx] = (
                        slice_offsets[shard_state.index + 1] - slice_offsets[shard_state.index]
                    )
                    active_slices[shard_idx] = shard_state
                    # print(
                    #     f"Shard iter for {self.worker_config.rank}:{self.worker_config.rank_worker_id()} exhausted, taking next shard {shard.name} [{shard.offset}, {shard.offset + shard.count}), {len(shards_order)} shards left, probs={shards_probs}"
                    # )
                else:
                    active_slice_probs[shard_idx] = 0
                    active_slices[shard_idx] = None
                    # print(
                    #     f"Shard iter for {self.worker_config.rank}:{self.worker_config.rank_worker_id()} exhausted, no next shards, probs={shards_probs}"
                    # )
                if self.worker_config.should_log(level=2):
                    self.worker_config.worker_log(
                        {
                            "t": "WebdatasetSampleLoaderDataset._shards_iter.exhausted",
                            "r": self.worker_config.rank,
                            "w": self.worker_config.rank_worker_id(),
                            "remaining": len(pending_slice_indexes),
                            "count": self._sample_count[worker_idx],
                            "epoch": self._epoch_count[worker_idx],
                            "epoch_count": self._epoch_sample_count[worker_idx],
                            "probs": active_slice_probs.tolist(),
                        }
                    )
            else:
                assert sample is not None
                self._sample_count[worker_idx] += 1
                self._epoch_sample_count[worker_idx] += 1
                if self.worker_config.should_log(level=1):
                    self.worker_config.worker_log(
                        {
                            "t": "WebdatasetSampleLoaderDataset._shards_iter.yield",
                            "r": self.worker_config.rank,
                            "w": self.worker_config.rank_worker_id(),
                            "index": sample.__restore_key__[1],
                            "key": sample.data[0]["__key__"],
                            "count": self._sample_count[worker_idx],
                            "epoch": self._epoch_count[worker_idx],
                            "epoch_count": self._epoch_sample_count[worker_idx],
                        }
                    )
                yield sample
        if self.worker_config.should_log(level=2):
            self.worker_config.worker_log(
                {
                    "t": "WebdatasetSampleLoaderDataset._shards_iter.all_exhausted",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "count": self._sample_count[worker_idx],
                    "epoch": self._epoch_count[worker_idx],
                    "epoch_count": self._epoch_sample_count[worker_idx],
                }
            )

        # Epoch has finished, reset states.
        self._epoch_count[worker_idx] += 1
        self._epoch_sample_count[worker_idx] = 0
        self._pending_slice_indexes[worker_idx] = None
        self._pending_shards_offset[worker_idx] = None
        # print(
        #     f"Shard iters exhausted for {self.worker_config.rank}:{self.worker_config.rank_worker_id()} after {cnt} samples"
        # )

    def __len__(self) -> int:
        return sum(
            slice_offsets[-1] - slice_offsets[0] for slice_offsets in self.worker_slice_offsets
        )

    def worker_has_samples(self) -> bool:
        self.worker_config.assert_worker()
        return len(self.worker_slice_offsets[self.worker_config.rank_worker_id()]) > 1

    def __iter__(self) -> Iterator[RawSampleData]:
        self.worker_config.assert_worker()

        slice_offsets = self.worker_slice_offsets[self.worker_config.rank_worker_id()]
        if self.worker_config.should_log(level=1):
            self.worker_config.worker_log(
                {
                    "t": "WebdatasetSampleLoaderDataset.__iter__",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "slice_offsets": slice_offsets,
                    "parallel_shard_iters": self.parallel_shard_iters,
                    "shuffle_over_epochs": self.shuffle_over_epochs,
                }
            )

        if len(slice_offsets) <= 1:
            return

        yield from self._shards_iter()

    def can_restore_sample(self) -> bool:
        return True

    def assert_can_restore(self) -> None:
        pass

    def restore_sample(self, key: Tuple[Union[str, int, tuple], ...]) -> RawSampleData:
        # Key is: ("Webdataset", index)
        # The key is joined in the dataset's typed joining (i.e. load_sample of JoinedWebdatasetFactory).
        id, index = key
        assert id == "Webdataset"
        assert isinstance(index, int)
        return self._get_sample(index)

    def save_state(self) -> SampleLoaderState:
        self.worker_config.assert_worker()
        worker_idx = self.worker_config.rank_worker_id()
        if self.worker_config.should_log(level=3):
            self.worker_config.worker_log(
                {
                    "t": "WebdatasetSampleLoaderDataset.save_state",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "count": self._sample_count[worker_idx],
                    "epoch": self._epoch_count[worker_idx],
                    "epoch_count": self._epoch_sample_count[worker_idx],
                    "pending_shards": self._pending_slice_indexes[worker_idx],
                    "active_shards": (
                        None
                        if self._active_slice_state[worker_idx] is None
                        else [
                            (
                                None
                                if slice_state is None
                                else {
                                    "index": slice_state.index,
                                    "current": slice_state.current,
                                }
                            )
                            for slice_state in self._active_slice_state[worker_idx]
                        ]
                    ),
                }
            )
        return SampleLoaderState(
            rng=self._worker_rng.save_state(),
            pending_slices_offset=self._pending_shards_offset[worker_idx],
            pending_slices_seed=WorkerRngState(rng=self._pending_shards_seed.rng[worker_idx]),
            active_slices=(
                None
                if self._active_slice_state[worker_idx] is None
                else [
                    None if slice_state is None else dataclasses.replace(slice_state)
                    for slice_state in self._active_slice_state[worker_idx]
                ]
            ),
            sample_count=self._sample_count[worker_idx],
            epoch_count=self._epoch_count[worker_idx],
            epoch_sample_count=self._epoch_sample_count[worker_idx],
        )

    def merge_states(self, states: List[SampleLoaderState]) -> SampleLoaderMergedState:
        assert all(s is None or isinstance(s, SampleLoaderState) for s in states)
        assert len(states) == max(self.worker_config.num_workers, 1)
        return SampleLoaderMergedState(
            rng=self._worker_rng.merge_states([None if s is None else s.rng for s in states]),
            pending_slices_offset=[s.pending_slices_offset for s in states],
            pending_slices_seed=self._worker_rng.merge_states(
                [None if s is None else s.pending_slices_seed for s in states]
            ),
            active_slices=[
                [None] * self.parallel_shard_iters if s is None else s.active_slices for s in states
            ],
            sample_count=[0 if s is None else s.sample_count for s in states],
            epoch_count=[0 if s is None else s.epoch_count for s in states],
            epoch_sample_count=[0 if s is None else s.epoch_sample_count for s in states],
        )

    def restore_state(self, state: Optional[SampleLoaderMergedState]) -> None:
        if self.worker_config.should_log(level=3):
            self.worker_config.worker_log(
                {
                    "t": "WebdatasetSampleLoaderDataset.restore_state",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "state": str(state),
                }
            )
        # print(f"Restore state {state}")
        n_workers = max(1, self.worker_config.num_workers)
        if state is None:
            # Restore initial state
            self._worker_rng.restore_state(None)
            self._pending_slice_indexes = [None] * n_workers
            self._pending_shards_offset = [None] * n_workers
            self._pending_shards_seed = WorkerRngMergedState(rng=[None] * n_workers)
            self._active_slice_state = [
                [None] * self.parallel_shard_iters for _ in range(n_workers)
            ]
            self._sample_count = [0] * n_workers
            self._epoch_count = [0] * n_workers
            self._epoch_sample_count = [0] * n_workers
        else:
            assert isinstance(state, SampleLoaderMergedState)
            self._worker_rng.restore_state(state.rng)
            # Restore state
            assert len(state.pending_slices_offset) == n_workers
            assert len(state.active_slices) == n_workers
            self._pending_slice_indexes = [None] * n_workers
            self._pending_shards_offset = list(state.pending_slices_offset)
            self._pending_shards_seed = state.pending_slices_seed
            self._active_slice_state = [
                (
                    [None] * self.parallel_shard_iters
                    if shard_active_slices is None
                    else [
                        None if slice_state is None else dataclasses.replace(slice_state)
                        for slice_state in shard_active_slices
                    ]
                )
                for shard_active_slices in state.active_slices
            ]
            self._sample_count = state.sample_count
            self._epoch_count = state.epoch_count
            self._epoch_sample_count = state.epoch_sample_count
            # Assert the state is valid
            for worker_slice_states, worker_slice_offsets in zip(
                self._active_slice_state, self.worker_slice_offsets
            ):
                if worker_slice_states is not None:
                    for slice_state in worker_slice_states:
                        if slice_state is not None:
                            assert (
                                worker_slice_offsets[slice_state.index]
                                <= slice_state.current
                                < worker_slice_offsets[slice_state.index + 1]
                            ), f"Invalid slice state: slice_offsets[{slice_state.index}] must hold {worker_slice_offsets[slice_state.index]} <= {slice_state.current} < {worker_slice_offsets[slice_state.index + 1]}"

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "total_count": len(self.join_readers),
            "readers": [
                {
                    # 'base_path': EPath.common_path(reader.tar_filepaths),
                    "tar_filenames": [str(p) for p in reader.tar_filenames],
                }
                for reader in self.join_readers
            ],
            "shard_slices": self.worker_slice_offsets,
            "worker_config": self.worker_config.config(),
            "exclude": list(self.exclude),
            "shuffle_over_epochs": self.shuffle_over_epochs,
            "parallel_shard_iters": self.parallel_shard_iters,
        }

    def __str__(self):
        return f"WebdatasetSampleLoaderDataset(join_readers={self.join_readers}, shuffle_over_epochs={self.shuffle_over_epochs}, parallel_shard_iters={self.parallel_shard_iters})"
