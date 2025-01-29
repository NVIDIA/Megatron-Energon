# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import dataclasses
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch

from megatron.energon.epathlib import EPath
from megatron.energon.errors import SYSTEM_EXCEPTIONS, FatalSampleError
from megatron.energon.flavors.base_dataset import MergedState, SavableDataset, State
from megatron.energon.flavors.webdataset.single_shard_reader import SingleShardReader
from megatron.energon.flavors.webdataset.structs import (
    FilteredSample,
    ShardInfo,
    ShardState,
    reraise_exception,
)
from megatron.energon.rng import WorkerRng, WorkerRngMergedState, WorkerRngState
from megatron.energon.worker import WorkerConfig


@dataclass
class SampleLoaderState(State):
    """
    The savable state for the wds sample loader. Contains the active and pending shards.
    """

    #: Rng state
    rng: WorkerRngState
    #: The seed that was used to generate the current pending shards
    pending_shards_seed: WorkerRngState
    #: The number of shards that have already been opened / processed and thus been removed from the
    # pending shards. None if starting a new epoch.
    # Pending shards are the shards which have not yet been opened, but should be processed in the
    # current epoch.
    pending_shards_offset: Optional[int]
    #: The active shards are the currently opened shards. May contain `None`, if there are fewer
    # shards available (i.e. pending_shards empty) than parallel shard iterators requested.
    active_shards: Optional[List[Optional[List[ShardState]]]]
    #: The total number of samples retrieved, it's just a monotonically increasing counter
    sample_count: int
    #: Number of epochs this dataset has been iterated over
    epoch_count: int
    #: Number of samples retrieved in current epoch
    epoch_sample_count: int


@dataclass
class SampleLoaderMergedState(MergedState):
    #: Rng state
    rng: WorkerRngMergedState
    #: The seed that was used to generate the current pending shards
    pending_shards_seed: WorkerRngMergedState
    #: The number of shards that have already been opened / processed and thus been removed from the
    # pending shards. None if starting a new epoch.
    # Pending shards are the shards which have not yet been opened, but should be processed in the
    # current epoch.
    pending_shards_offset: List[Optional[int]]
    #: The active shards are the currently opened shards. May contain `None`, if there are fewer
    # shards available (i.e. pending_shards empty) than parallel shard iterators requested.
    active_shards: List[Optional[List[Optional[List[ShardState]]]]]
    #: The total number of samples retrieved, it's just a monotonically increasing counter
    sample_count: List[int]
    #: Number of epochs this dataset has been iterated over
    epoch_count: List[int]
    #: The number of samples retrieved in current epoch
    epoch_sample_count: List[int]


class WebdatasetSampleLoaderDataset(SavableDataset[Tuple[Optional[FilteredSample], ...]]):
    """Internal class for loading samples from webdataset shards"""

    #: All shards for all workers `shards[worker_idx][shard_idx][merge_idx]`
    shards: List[List[Sequence[ShardInfo]]]
    #: All shards for all workers accessible by name and offset
    # `shards[worker_idx][(shard_name, offset)]`, created lazily
    _shards_by_key: Optional[List[Dict[Tuple[str, int], Sequence[ShardInfo]]]] = None
    #: Paths to shards for all workers by name `shards[worker_idx][shard_name]`, created lazily
    _shard_paths_by_name: Optional[Dict[str, List[EPath]]] = None
    # Sample keys to ignore
    exclude: Set[str]

    # If = 1, every sample is seen exactly once per epoch. If > 1, samples
    # (or rather shard slices) are shuffled within this number of epochs (i.e. randomly
    # selected without replacement). If None, the shards are effectively shuffle over
    # infinite epochs (i.e. shard slices are drawn with replacement).
    shuffle_over_epochs: Optional[int]
    # Number of parallel iterators to be opened simultaneously (and random sample between them)
    parallel_shard_iters: int
    # The method to join datasets. One of 'inner_match', 'inner', 'left'.
    # inner_match: Both data sources must exactly match, otherwise an exception is raised for a non-
    #     matching sample.
    # inner: The primary dataset is iterated and the other datasets are merged via key. If there is
    #     no match of the extra datasets, skip the sample.
    # left: The primary dataset is iterated and the other datasets are merged via key. If there is
    #     no match of the extra datasets, the sample is still yielded without the missing column.
    dataset_join_method: Literal["inner_match", "inner", "left"]
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
    _pending_shards: List[Optional[List[Sequence[ShardInfo]]]]
    #: The active shards are the currently opened shards. May contain `None`, if there are fewer
    # shards available (i.e. pending_shards empty) than parallel shard iterators requested.
    _active_shards_state: List[Optional[List[Optional[List[ShardState]]]]]
    #: The total number of samples retrieved, it's just a monotonically increasing counter
    _sample_count: List[int]
    #: Number of epochs this dataset has been iterated over
    _epoch_count: List[int]
    #: The number of samples retrieved in current epoch
    _epoch_sample_count: List[int]

    def __init__(
        self,
        rank_shards: List[List[Sequence[ShardInfo]]],
        *,
        worker_config: WorkerConfig,
        exclude: Set[str],
        part_filter: Optional[Callable[[str], bool]] = None,
        shuffle_over_epochs: Optional[int] = None,
        parallel_shard_iters: int = 1,
        dataset_join_method: Literal["inner_match"] = "inner_match",
        handler: Callable[[Exception, Optional[str]], None] = reraise_exception,
    ):
        """
        The webdataset loader. Iterates over the shard infos and yields the samples.

        Args:
            rank_shards: The shards to iterate over for each worker of the current rank for each
                merged dataset.
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
            dataset_join_method: The method to join datasets. One of 'inner_match'. Further modes
                may come in the future if needed.
            handler: Exception handler. Args: (exception, key).
        """
        super().__init__(worker_config=worker_config)
        self.shards = rank_shards
        self.exclude = exclude
        self.part_filter = part_filter
        self.shuffle_over_epochs = shuffle_over_epochs
        self.parallel_shard_iters = parallel_shard_iters
        self.dataset_join_method = dataset_join_method
        self.handler = handler
        self._worker_rng = WorkerRng(worker_config)
        self._pending_shards = [None] * len(self.shards)
        self._pending_shards_offset = [None] * len(self.shards)
        self._pending_shards_seed = WorkerRngMergedState(rng=[None] * len(self.shards))
        self._active_shards_state = [[None] * parallel_shard_iters for _ in range(len(self.shards))]
        self._sample_count = [0] * len(self.shards)
        self._epoch_count = [0] * len(self.shards)
        self._epoch_sample_count = [0] * len(self.shards)
        assert shuffle_over_epochs is None or shuffle_over_epochs == -1 or shuffle_over_epochs >= 1
        assert self.parallel_shard_iters >= 1

    @property
    def shards_by_key(self) -> List[Dict[Tuple[str, int], Sequence[ShardInfo]]]:
        if self._shards_by_key is None:
            self._shards_by_key = [
                {(subshards[0].name, subshards[0].offset): subshards for subshards in worker_shards}
                for worker_shards in self.shards
            ]
        return self._shards_by_key

    @property
    def shard_path_map(self) -> Dict[str, List[EPath]]:
        if self._shard_paths_by_name is None:
            self._shard_paths_by_name = {
                shard[0].name: [s.path for s in shard] for shards in self.shards for shard in shards
            }
        return self._shard_paths_by_name

    def _read_multicolumn_shard(
        self,
        shard_states: List[ShardState],
    ) -> Generator[Tuple[Optional[FilteredSample], ...], None, None]:
        """
        Reads samples, possibly from multiple column shards, resuming from a saved state if needed.
        The first shard is assumed to be the primary shard (giving the keys of the samples), while the
        other shards are the extra column shards, possibly overwriting columns.
        """
        ctx = contextlib.ExitStack()
        readers = [
            ctx.enter_context(
                SingleShardReader(
                    self.worker_config,
                    self.exclude,
                    shard_state,
                    self.part_filter,
                    self.handler,
                )
            )
            for shard_state in shard_states
        ]
        groups: List[Optional[FilteredSample]] = [None] * len(readers)
        with ctx:
            while True:
                try:
                    group_name, groups[0] = readers[0].read_next()
                except StopIteration:
                    # Expecting all column readers to raise StopIteration as well
                    for reader in readers[1:]:
                        try:
                            reader.read_next()
                        except StopIteration:
                            pass
                        else:
                            raise FatalSampleError.from_sample_key(
                                f"{reader.shard_state.shard.name}"
                            ) from ValueError(
                                "Extra column shard exhausted when primary shard is exhausted"
                            )
                    break
                except SYSTEM_EXCEPTIONS:
                    raise FatalSampleError.from_sample_key(f"{readers[0].shard_state.shard.path}")
                except Exception as e:
                    self.handler(e, readers[0].shard_state.shard.name)
                else:
                    for idx, reader in enumerate(readers[1:], 1):
                        try:
                            _, groups[idx] = reader.read_next(group_name, must_match=True)
                        except StopIteration:
                            if self.dataset_join_method == "inner_match":
                                raise FatalSampleError.from_sample_key(
                                    f"{reader.shard_state.shard.name}"
                                ) from ValueError(
                                    "Extra column shard exhausted when primary shard is exhausted"
                                )
                            else:
                                assert (
                                    False
                                ), f"join method {self.dataset_join_method} not implemented"
                            continue
                        except SYSTEM_EXCEPTIONS:
                            raise FatalSampleError.from_sample_key(
                                f"{reader.shard_state.shard.path}"
                            )
                        except Exception as e:
                            self.handler(e, reader.shard_state.shard.name)
                    yield tuple(groups)

    def _shards_once(self) -> List[Sequence[ShardInfo]]:
        """Possibly (re)shuffles the shards using the random generator."""
        worker_idx = self.worker_config.rank_worker_id()
        shards = self.shards[worker_idx]
        shards_offset = self._pending_shards_offset[worker_idx]

        if self.shuffle_over_epochs is None:
            # No shuffling
            res_list = list(shards)
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
                res_list = [
                    shards[rng.randbelow(len(shards))] for _ in range(self.parallel_shard_iters)
                ]
            elif self.shuffle_over_epochs >= 1:
                # Shuffle without replacement (potentially over multiple epochs)
                res_list = rng.shuffle(shards * self.shuffle_over_epochs)
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
        self._pending_shards[worker_idx] = res_list
        return res_list

    def _shards_iter(self) -> Generator[Tuple[Optional[FilteredSample], ...], None, None]:
        """Iterates the samples in a list of shards, possibly using multiple parallel iterators over
        the shards."""
        worker_idx = self.worker_config.rank_worker_id()
        # Cleanup other states
        for i in range(self.worker_config.num_workers):
            if i != worker_idx:
                self._active_shards_state[i] = [None] * self.parallel_shard_iters
                self._pending_shards[i] = None
        shards_probs = torch.empty(self.parallel_shard_iters, dtype=torch.float32)
        shard_iters: List[Optional[Generator[Tuple[Optional[FilteredSample], ...], None, None]]] = (
            []
        )
        active_shards: List[Optional[List[ShardState]]] = self._active_shards_state[worker_idx]

        # Weight the shards by their size to get a more even distribution of samples
        shards_probs[:] = 0
        shard_iters = []
        pending_shards = self._pending_shards[worker_idx]
        if (
            any(s is not None for s in active_shards)
            or self._pending_shards_offset[worker_idx] is not None
        ):
            # Having an active state, or pending shards. This means we are resuming an epoch.
            if pending_shards is None:
                # Need to restore the pending shards
                pending_shards = self._shards_once()
            assert pending_shards is not None

            # Restore the state
            assert active_shards is not None
            assert len(active_shards) == self.parallel_shard_iters
            for shard_states in active_shards:
                if shard_states is None:
                    shard_iters.append(None)
                else:
                    shards_probs[len(shard_iters)] = shard_states[0].shard.count
                    shard_iters.append(self._read_multicolumn_shard(shard_states))

            if self.worker_config.should_log(level=1):
                self.worker_config.worker_log(
                    {
                        "t": "WebdatasetSampleLoaderDataset._shards_iter.resume_epoch",
                        "r": self.worker_config.rank,
                        "w": self.worker_config.rank_worker_id(),
                        "shards": [
                            [
                                {
                                    "name": subshard.name,
                                    "path": str(subshard.path),
                                    "offset": subshard.offset,
                                    "count": subshard.count,
                                }
                                for subshard in subshards
                            ]
                            for subshards in pending_shards
                        ],
                        "active_shards": [
                            (
                                None
                                if subshard_states is None
                                else [
                                    {
                                        "shard": {
                                            "name": subshard_state.shard.name,
                                            "path": str(subshard_state.shard.path),
                                            "offset": subshard_state.shard.offset,
                                            "count": subshard_state.shard.count,
                                        },
                                        "offset": subshard_state.offset,
                                    }
                                    for subshard_state in subshard_states
                                ]
                            )
                            for subshard_states in active_shards
                        ],
                        "count": self._sample_count[worker_idx],
                        "epoch": self._epoch_count[worker_idx],
                        "epoch_count": self._epoch_sample_count[worker_idx],
                        "probs": shards_probs.tolist(),
                    }
                )

        else:
            # Start a new epoch
            assert pending_shards is None
            pending_shards = self._shards_once()

            if self.worker_config.should_log(level=1):
                self.worker_config.worker_log(
                    {
                        "t": "WebdatasetSampleLoaderDataset._shards_iter.next_epoch",
                        "r": self.worker_config.rank,
                        "w": self.worker_config.rank_worker_id(),
                        "shards": [
                            [
                                {
                                    "name": shard.name,
                                    "path": str(shard.path),
                                    "offset": shard.offset,
                                    "count": shard.count,
                                }
                                for shard in subshards
                            ]
                            for subshards in pending_shards
                        ],
                        "count": self._sample_count[worker_idx],
                        "epoch": self._epoch_count[worker_idx],
                        "epoch_count": self._epoch_sample_count[worker_idx],
                        "probs": shards_probs.tolist(),
                        "shuffle_over_epochs": self.shuffle_over_epochs,
                    }
                )

            # List of shard iterators, always of length `parallel_shard_iters`. May contain `None`.
            active_shards = []
            # Fill up the shard iterators
            while len(pending_shards) > 0 and len(shard_iters) < self.parallel_shard_iters:
                subshards = pending_shards.pop()
                assert self._pending_shards_offset[worker_idx] is not None
                self._pending_shards_offset[worker_idx] += 1
                shard_states = [
                    ShardState(shard=shard, byte_offset=0, offset=0) for shard in subshards
                ]
                shards_probs[len(shard_iters)] = subshards[0].count
                shard_iters.append(self._read_multicolumn_shard(shard_states))
                active_shards.append(shard_states)
            # Fill up the shard iterators with None
            for _ in range(len(shard_iters), self.parallel_shard_iters):
                shard_iters.append(None)
                active_shards.append(None)

            self._active_shards_state[worker_idx] = active_shards

        # print(
        #     f"Next shard iters generated for {self.worker_config.rank}:{self.worker_config.rank_worker_id()}: probs={shards_probs}"
        # )

        # Iterate over the shard iterators while there is an iterator left
        while torch.count_nonzero(shards_probs).item() > 0:
            if self.shuffle_over_epochs is None:
                # No shuffling, deterministic order, always the same
                assert self.parallel_shard_iters == 1
                shard_iter = shard_iters[0]
            else:
                # Take a random shard iterator
                shard_iter = self._worker_rng.choice(shard_iters, probs=shards_probs)
            assert shard_iter is not None
            try:
                sample: Tuple[Optional[FilteredSample], ...] = next(shard_iter)
            except StopIteration:
                # Iterator exhausted -> take next / remove from list
                rm_idx = shard_iters.index(shard_iter)
                if len(pending_shards) > 0 or self.shuffle_over_epochs == -1:
                    if len(pending_shards) > 0:
                        # Take the next shard (without replacement)
                        subshards = pending_shards.pop()
                        assert self._pending_shards_offset[worker_idx] is not None
                        self._pending_shards_offset[worker_idx] += 1
                    else:
                        # Randomly select a new shard directly (with replacement)
                        shards = self.shards[worker_idx]
                        subshards = shards[self._worker_rng.randbelow(len(shards))]
                    shard_states = [
                        ShardState(shard=shard, byte_offset=0, offset=0) for shard in subshards
                    ]
                    shard_iters[rm_idx] = self._read_multicolumn_shard(shard_states)
                    shards_probs[rm_idx] = subshards[0].count
                    active_shards[rm_idx] = shard_states
                    # print(
                    #     f"Shard iter for {self.worker_config.rank}:{self.worker_config.rank_worker_id()} exhausted, taking next shard {shard.name} [{shard.offset}, {shard.offset + shard.count}), {len(shards_order)} shards left, probs={shards_probs}"
                    # )
                else:
                    shard_iters[rm_idx] = None
                    shards_probs[rm_idx] = 0
                    active_shards[rm_idx] = None
                    # print(
                    #     f"Shard iter for {self.worker_config.rank}:{self.worker_config.rank_worker_id()} exhausted, no next shards, probs={shards_probs}"
                    # )
                if self.worker_config.should_log(level=2):
                    self.worker_config.worker_log(
                        {
                            "t": "WebdatasetSampleLoaderDataset._shards_iter.exhausted",
                            "r": self.worker_config.rank,
                            "w": self.worker_config.rank_worker_id(),
                            "remaining": len(pending_shards),
                            "count": self._sample_count[worker_idx],
                            "epoch": self._epoch_count[worker_idx],
                            "epoch_count": self._epoch_sample_count[worker_idx],
                            "probs": shards_probs.tolist(),
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
                            "key": sample[0]["__key__"] if sample[0] is not None else None,
                            "shard": sample[0]["__shard__"] if sample[0] is not None else None,
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
        self._pending_shards[worker_idx] = None
        self._pending_shards_offset[worker_idx] = None
        # print(
        #     f"Shard iters exhausted for {self.worker_config.rank}:{self.worker_config.rank_worker_id()} after {cnt} samples"
        # )

    def __len__(self) -> int:
        return sum(
            subshards[0].count for worker_shards in self.shards for subshards in worker_shards
        )

    def worker_has_samples(self) -> bool:
        self.worker_config.assert_worker()
        worker_shards = self.shards[self.worker_config.rank_worker_id()]
        return any(subshard[0].count > 0 for subshard in worker_shards)

    def __iter__(self) -> Iterator[Tuple[Optional[FilteredSample], ...]]:
        self.worker_config.assert_worker()

        worker_shards = self.shards[self.worker_config.rank_worker_id()]
        if self.worker_config.should_log(level=1):
            self.worker_config.worker_log(
                {
                    "t": "WebdatasetSampleLoaderDataset.__iter__",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "shard_range": [
                        [
                            f"{shard.name}[{shard.offset}, {shard.offset+shard.count})"
                            for shard in subshards
                        ]
                        for subshards in worker_shards
                    ],
                    "parallel_shard_iters": self.parallel_shard_iters,
                    "shuffle_over_epochs": self.shuffle_over_epochs,
                }
            )

        if len(worker_shards) == 0:
            return

        yield from self._shards_iter()

    def can_restore_sample(self) -> bool:
        return True

    def assert_can_restore(self) -> None:
        pass

    def restore_sample(
        self, key: Tuple[Union[str, int, tuple], ...]
    ) -> Tuple[Optional[FilteredSample], ...]:
        # Key is: ("Webdataset", shard_name, shard_offset, shard_name2, shard_offset2, ...)
        # The key is joined in the dataset's typed joining (i.e. load_sample of MergedWebdataset).
        id, *shard_data = key
        assert id == "Webdataset"
        assert isinstance(shard_data[0], str)
        shard_paths = self.shard_path_map[shard_data[0]]
        assert len(shard_paths) * 2 == len(shard_data), "Key does not match joined datasets"

        sample_shard_infos = [
            ShardState(
                shard=ShardInfo(
                    name=name,
                    path=shard_path,
                    offset=max(offset, 0),
                    count=1 if offset >= 0 else 0,
                    byte_offset=None,
                    byte_size=None,
                ),
                offset=0,
                byte_offset=0,
            )
            for shard_path, name, offset in zip(shard_paths, shard_data[::2], shard_data[1::2])
        ]

        gen = self._read_multicolumn_shard(sample_shard_infos)
        sample = next(gen)
        gen.close()
        return sample

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
                    "pending_shards": (
                        [
                            [
                                {
                                    "name": subshard.name,
                                    "path": str(subshard.path),
                                    "offset": subshard.offset,
                                    "count": subshard.count,
                                }
                                for subshard in subshards
                            ]
                            for subshards in self._pending_shards[worker_idx]
                        ]
                        if self._pending_shards[worker_idx] is not None
                        else None
                    ),
                    "active_shards": (
                        None
                        if self._active_shards_state[worker_idx] is None
                        else [
                            (
                                None
                                if subshard_states is None
                                else [
                                    {
                                        "shard": {
                                            "name": shard_state.shard.name,
                                            "path": str(shard_state.shard.path),
                                            "offset": shard_state.shard.offset,
                                            "count": shard_state.shard.count,
                                        },
                                        "offset": shard_state.offset,
                                    }
                                    for shard_state in subshard_states
                                ]
                            )
                            for subshard_states in self._active_shards_state[worker_idx]
                        ]
                    ),
                }
            )
        return SampleLoaderState(
            rng=self._worker_rng.save_state(),
            pending_shards_offset=self._pending_shards_offset[worker_idx],
            pending_shards_seed=WorkerRngState(rng=self._pending_shards_seed.rng[worker_idx]),
            active_shards=(
                None
                if self._active_shards_state[worker_idx] is None
                else [
                    (
                        None
                        if active_subshards is None
                        else [dataclasses.replace(subshard) for subshard in active_subshards]
                    )
                    for active_subshards in self._active_shards_state[worker_idx]
                ]
            ),
            sample_count=self._sample_count[worker_idx],
            epoch_count=self._epoch_count[worker_idx],
            epoch_sample_count=self._epoch_sample_count[worker_idx],
        )

    def merge_states(self, states: List[SampleLoaderState]) -> SampleLoaderMergedState:
        assert all(s is None or isinstance(s, SampleLoaderState) for s in states)
        assert len(states) == len(self.shards)
        return SampleLoaderMergedState(
            rng=self._worker_rng.merge_states([None if s is None else s.rng for s in states]),
            pending_shards_offset=[s.pending_shards_offset for s in states],
            pending_shards_seed=self._worker_rng.merge_states(
                [None if s is None else s.pending_shards_seed for s in states]
            ),
            # pending_shards=[[] if s is None else s.pending_shards for s in states],
            active_shards=[
                [None] * self.parallel_shard_iters if s is None else s.active_shards for s in states
            ],
            sample_count=[0 if s is None else s.sample_count for s in states],
            epoch_count=[0 if s is None else s.epoch_count for s in states],
            epoch_sample_count=[0 if s is None else s.epoch_sample_count for s in states],
        )

    def _restore_find_shard(
        self,
        subshards_data: Sequence[ShardInfo],
        shards_by_key: Dict[Tuple[str, int], Sequence[ShardInfo]],
    ) -> Sequence[ShardInfo]:
        subshards = shards_by_key[(subshards_data[0].name, subshards_data[0].offset)]
        for subshard_data, subshard in zip(subshards_data, subshards):
            if subshard != subshard_data:
                raise ValueError(
                    f"Shard {subshard_data!r} not found in {self.shards!r}, states differ, not recoverable"
                )
            # Copy over the byte size and offset. Saves some loading time,
            # especially for restoring lots of random samples :)
            if subshard_data.byte_offset is not None and subshard.byte_offset is None:
                subshard.byte_offset = subshard_data.byte_offset
            if subshard_data.byte_size is not None and subshard.byte_size is None:
                subshard.byte_size = subshard_data.byte_size
        return subshards

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
        if state is None:
            # Restore initial state
            self._worker_rng.restore_state(None)
            self._pending_shards = [None] * len(self.shards)
            self._pending_shards_offset = [None] * len(self.shards)
            self._pending_shards_seed = WorkerRngMergedState(rng=[None] * len(self.shards))
            self._active_shards_state = [
                [None] * self.parallel_shard_iters for _ in range(len(self.shards))
            ]
            self._sample_count = [0] * len(self.shards)
            self._epoch_count = [0] * len(self.shards)
            self._epoch_sample_count = [0] * len(self.shards)
        else:
            assert isinstance(state, SampleLoaderMergedState)
            self._worker_rng.restore_state(state.rng)
            # Restore state
            assert len(state.pending_shards_offset) == len(self.shards)
            assert len(state.active_shards) == len(self.shards)
            self._pending_shards = [None] * len(self.shards)
            self._pending_shards_offset = list(state.pending_shards_offset)
            self._pending_shards_seed = state.pending_shards_seed
            self._active_shards_state = [
                (
                    [
                        (
                            None
                            if subshard_states is None
                            else [
                                ShardState(
                                    shard=subshard,
                                    offset=subshard_state.offset,
                                    byte_offset=subshard_state.byte_offset,
                                )
                                for subshard, subshard_state in zip(
                                    self._restore_find_shard(
                                        [s.shard for s in subshard_states], shards_by_key
                                    ),
                                    subshard_states,
                                )
                            ]
                        )
                        for subshard_states in active_shard_states
                    ]
                    if active_shard_states is not None
                    else None
                )
                for active_shard_states, shards_by_key in zip(
                    state.active_shards, self.shards_by_key
                )
            ]
            self._sample_count = state.sample_count
            self._epoch_count = state.epoch_count
            self._epoch_sample_count = state.epoch_sample_count

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "shards": [
                [
                    [
                        {
                            "name": subshard.name,
                            "path": str(subshard.path),
                            "offset": subshard.offset,
                            "count": subshard.count,
                            "byte_offset": subshard.byte_offset,
                            "byte_size": subshard.byte_size,
                        }
                        for subshard in subshards
                    ]
                    for subshards in shards
                ]
                for shards in self.shards
            ],
            "worker_config": self.worker_config.config(),
            "exclude": list(self.exclude),
            "shuffle_over_epochs": self.shuffle_over_epochs,
            "parallel_shard_iters": self.parallel_shard_iters,
        }

    def __str__(self):
        return f"WebdatasetSampleLoaderDataset(shards={self.shards}, shuffle_over_epochs={self.shuffle_over_epochs}, parallel_shard_iters={self.parallel_shard_iters})"
