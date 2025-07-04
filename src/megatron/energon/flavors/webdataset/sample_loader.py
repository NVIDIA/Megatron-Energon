# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generator, Iterator, List, Optional, Sequence, Tuple, Union

import torch

from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import FlexState, SavableDataset
from megatron.energon.flavors.webdataset.itar_reader import ITarReader
from megatron.energon.flavors.webdataset.structs import FilteredSample
from megatron.energon.rng import WorkerRng
from megatron.energon.worker import WorkerConfig


@edataclass
class RawSampleData:
    """Represents the iteration state of a single slice slice to the index."""

    #: Index of the sample. This is also the restore key
    __restore_key__: Tuple[str, int]
    #: The sample data
    data: Tuple[Optional[FilteredSample], ...]


@edataclass
class SliceState:
    """Represents the iteration state of a single slice slice to the index."""

    #: The slice index of this slice state
    index: int
    #: The actual state: The global sample offset (`slice[index] <= offset < slice[index + 1]``)
    current: int


class WebdatasetSampleLoaderDataset(SavableDataset[RawSampleData]):
    """Internal class for loading samples from webdataset slices"""

    #: The readers for each joined dataset
    join_readers: Sequence[ITarReader]

    #: The offsets of the slice slices to iterate over for the current worker
    slice_offsets: Optional[Sequence[int]]

    # If = 1, every sample is seen exactly once per epoch. If > 1, samples
    # (or rather slice slices) are shuffled within this number of epochs (i.e. randomly
    # selected without replacement). If None, the slices are effectively shuffle over
    # infinite epochs (i.e. slice slices are drawn with replacement).
    shuffle_over_epochs: Optional[int]
    # Number of parallel iterators to be opened simultaneously (and random sample between them)
    parallel_slice_iters: int

    # Worker's random generator
    _worker_rng: WorkerRng

    #: The RNG state to be used for regenerating the pending slices
    _pending_slices_rng_state: Optional[FlexState]
    #: The number of slices that have already been opened / processed and thus been removed from the
    # pending slices.
    _pending_slices_offset: Optional[int]
    #: Pending slices are the slices which have not yet been opened, but should be processed
    # in the current "epoch". If None, regenerate from the seed and offset.
    _pending_slice_indexes: Optional[List[int]]
    #: The active slices are the currently opened slices. May contain `None`, if there are fewer
    # slices available (i.e. pending_slices empty) than parallel slice iterators requested.
    _active_slice_state: List[Optional[SliceState]]
    #: The total number of samples retrieved, it's just a monotonically increasing counter
    _sample_count: int
    #: Number of epochs this dataset has been iterated over
    _epoch_count: int
    #: The number of samples retrieved in current epoch
    _epoch_sample_count: int

    _savable_fields = (
        "_worker_rng",
        "_pending_slices_offset",
        "_pending_slice_indexes",
        "_active_slice_state",
        "_sample_count",
        "_epoch_count",
        "_epoch_sample_count",
    )

    def __init__(
        self,
        join_readers: Sequence[ITarReader],
        workers_sample_slice_offsets: Sequence[Sequence[int]],
        *,
        worker_config: WorkerConfig,
        shuffle_over_epochs: Optional[int] = None,
        parallel_slice_iters: int = 1,
    ):
        """
        The webdataset loader. Iterates over the slice infos and yields the samples.

        Args:
            join_readers: A sequence of the joined readers (or just a single reader) to iterate over.
            worker_slice_offsets: The offsets of the slice slices to iterate over, for each worker.
            worker_config: The worker configuration.
            shuffle_over_epochs: If None, disable shuffling.
                If = 1, every sample is seen exactly once per epoch.
                If > 1, samples (or rather slice slices) are shuffled within this number of epochs
                (i.e. randomly selected without replacement).
                If -1, the slices are effectively shuffle over infinite epochs (i.e. slice slices
                are drawn with replacement).
            parallel_slice_iters: If > 1, samples are randomly drawn from parallel slice iterators.
                This will not impact performance, but increase randomness. If = 1, the slices are
                iterated in order.
        """
        super().__init__(worker_config=worker_config)

        self.join_readers = join_readers
        self.shuffle_over_epochs = shuffle_over_epochs
        self.parallel_slice_iters = parallel_slice_iters

        # Store the slices for all workers
        # The slices for the current worker, will have to be extracted from this list later
        self.workers_slice_offsets = workers_sample_slice_offsets
        self.slice_offsets = None

        self.reset_state_own()

        assert shuffle_over_epochs is None or shuffle_over_epochs == -1 or shuffle_over_epochs >= 1
        assert self.parallel_slice_iters >= 1

    def reset_state_own(self) -> None:
        self._worker_rng = WorkerRng(self.worker_config)
        self._pending_slice_indexes = None
        self._pending_slices_offset = None
        self._pending_slices_rng_state = None
        self._active_slice_state = [None] * self.parallel_slice_iters
        self._sample_count = 0
        self._epoch_count = 0
        self._epoch_sample_count = 0

    def ensure_slice_offsets(self) -> None:
        self.worker_config.assert_worker()

        if self.slice_offsets is None:
            self.slice_offsets = self.workers_slice_offsets[self.worker_config.rank_worker_id()]

    def _get_sample(self, index: int) -> RawSampleData:
        return RawSampleData(
            __restore_key__=("Webdataset", index),
            data=tuple(reader[index] for reader in self.join_readers),
        )

    def _slices_once(self) -> List[int]:
        """Yields the indexes to slice offsets once. Possibly shuffles the list."""
        assert self.slice_offsets is not None

        num_slices = len(self.slice_offsets) - 1
        slices_offset = self._pending_slices_offset

        if self.shuffle_over_epochs is None:
            # No shuffling
            res_list = list(range(num_slices))
            if slices_offset is None:
                slices_offset = 0
        else:
            # Restore state or start new (and save)
            if slices_offset is None:
                # Start new state. First, save the state to restore the same order.
                self._pending_slices_rng_state = self._worker_rng.save_state()
                rng = self._worker_rng
                slices_offset = 0
            else:
                # Restore the state. Create a dedicated rng for this, as the main rng is in the
                # state for iterating from the next iterator.
                assert self._pending_slices_rng_state is not None
                rng = WorkerRng(self.worker_config)
                rng.restore_state(self._pending_slices_rng_state)

            if self.shuffle_over_epochs == -1:
                # Shuffle with replacement (i.e. infinite epochs), effectively return as many slices
                # as are required for parallel slice iterators.
                # Next slices are drawn in the _slices_iter.
                res_list = [rng.randbelow(num_slices) for _ in range(self.parallel_slice_iters)]
            elif self.shuffle_over_epochs >= 1:
                # Shuffle without replacement (potentially over multiple epochs)
                res_list = rng.shuffle(list(range(num_slices)) * self.shuffle_over_epochs)
            else:
                raise ValueError(f"Invalid shuffle_over_epochs: {self.shuffle_over_epochs}")
        # Reverse, such that pop returns the first element (in O(1) time)
        res_list.reverse()
        # Skip restored slice list already processed slices
        assert slices_offset is not None
        self._pending_slices_offset = slices_offset
        if slices_offset > 0:
            # Those have already been popped in the current state
            del res_list[-slices_offset:]
        # Set the pending slices
        self._pending_slice_indexes = res_list
        return res_list

    def _slices_iter(self) -> Generator[RawSampleData, None, None]:
        """Iterates the samples in a list of slices, possibly using multiple parallel iterators over
        the slices."""

        assert self.slice_offsets is not None

        active_slice_probs = torch.zeros(self.parallel_slice_iters, dtype=torch.float32)
        active_slices = self._active_slice_state
        pending_slice_indexes = self._pending_slice_indexes

        def slice_at(idx: int) -> SliceState:
            assert self.slice_offsets is not None
            return SliceState(
                index=idx,
                current=self.slice_offsets[idx],
            )

        # Weight the slices by their size to get a more even distribution of samples
        if any(s is not None for s in active_slices) or self._pending_slices_offset is not None:
            # Having an active state, or pending slices. This means we are resuming an epoch.
            if pending_slice_indexes is None:
                # Need to restore the pending slices
                pending_slice_indexes = self._slices_once()
            assert pending_slice_indexes is not None

            # Restore the state
            assert len(active_slices) == self.parallel_slice_iters
            for idx, slice_state in enumerate(active_slices):
                if slice_state is not None:
                    active_slice_probs[idx] = (
                        self.slice_offsets[slice_state.index + 1]
                        - self.slice_offsets[slice_state.index]
                    )

            if self.worker_config.should_log(level=1):
                self.worker_config.worker_log(
                    {
                        "t": "WebdatasetSampleLoaderDataset._slices_iter.resume_epoch",
                        "r": self.worker_config.rank,
                        "w": self.worker_config.rank_worker_id(),
                        "pending_slice_indexes": pending_slice_indexes,
                        "active_slices": [
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
                        "count": self._sample_count,
                        "epoch": self._epoch_count,
                        "epoch_count": self._epoch_sample_count,
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
                        "t": "WebdatasetSampleLoaderDataset._slices_iter.next_epoch",
                        "r": self.worker_config.rank,
                        "w": self.worker_config.rank_worker_id(),
                        "pending_slice_indexes": pending_slice_indexes,
                        "count": self._sample_count,
                        "epoch": self._epoch_count,
                        "epoch_count": self._epoch_sample_count,
                        "probs": active_slice_probs.tolist(),
                        "shuffle_over_epochs": self.shuffle_over_epochs,
                    }
                )

            assert self._pending_slices_offset is not None

            # List of slice iterators, always of length `parallel_slice_iters`. May contain `None`.
            active_slices.clear()
            # Fill up the slice iterators
            while len(pending_slice_indexes) > 0 and len(active_slices) < self.parallel_slice_iters:
                slice_index = pending_slice_indexes.pop()
                self._pending_slices_offset += 1
                slice_state = slice_at(slice_index)
                active_slice_probs[len(active_slices)] = (
                    self.slice_offsets[slice_state.index + 1]
                    - self.slice_offsets[slice_state.index]
                )
                active_slices.append(slice_state)
            # Fill up the slice iterators with None
            for _ in range(len(active_slices), self.parallel_slice_iters):
                active_slices.append(None)

        # print(
        #     f"Next slice iters generated for {self.worker_config.rank}:{self.worker_config.rank_worker_id()}: probs={active_slice_probs}"
        # )
        # for slice_state in active_slices:
        #     if slice_state is None:
        #         print("  - None")
        #     else:
        #         print(
        #             f"  - [{slice_offsets[slice_state.index]}, {slice_offsets[slice_state.index + 1]}] at {slice_state.current}"
        #         )

        # Iterate over the slice iterators while there is an iterator left
        while torch.count_nonzero(active_slice_probs).item() > 0:
            if self.shuffle_over_epochs is None:
                # No shuffling, deterministic order, always the same
                assert self.parallel_slice_iters == 1
                slice_idx = 0
            else:
                # Take a random slice iterator
                slice_idx = self._worker_rng.choice_idx(active_slice_probs)
            slice_state = active_slices[slice_idx]
            assert slice_state is not None
            sample = self._get_sample(slice_state.current)
            # print(f"Read sample at {slice_state.current} -> {'None' if sample is None or sample.data[0] is None else sample.data[0]['__key__']}")
            slice_state.current += 1
            self._sample_count += 1
            self._epoch_sample_count += 1
            if slice_state.current >= self.slice_offsets[slice_state.index + 1]:
                # Iterator exhausted -> take next / remove from list
                if len(pending_slice_indexes) > 0 or self.shuffle_over_epochs == -1:
                    if len(pending_slice_indexes) > 0:
                        # Take the next slice (without replacement)
                        next_idx = pending_slice_indexes.pop()
                        assert self._pending_slices_offset is not None
                        self._pending_slices_offset += 1
                    else:
                        # Randomly select a new slice directly (with replacement)
                        num_slices = len(self.slice_offsets) - 1
                        next_idx = self._worker_rng.randbelow(num_slices)
                    next_slice_state = slice_at(next_idx)
                    active_slice_probs[slice_idx] = (
                        self.slice_offsets[next_slice_state.index + 1]
                        - self.slice_offsets[next_slice_state.index]
                    )
                    active_slices[slice_idx] = next_slice_state
                    # print(
                    #     f"Slice iter for {self.worker_config.rank}:{self.worker_config.rank_worker_id()} "
                    #     f"[{slice_offsets[slice_state.index]}, {slice_offsets[slice_state.index + 1]}] exhausted at {slice_state.current}, "
                    #     f"taking next slice {next_slice_state} [{slice_offsets[next_slice_state.index]}, {slice_offsets[next_slice_state.index + 1]}], "
                    #     f"{len(pending_slice_indexes)} slices left, probs={active_slice_probs.tolist()}"
                    # )
                else:
                    active_slice_probs[slice_idx] = 0
                    active_slices[slice_idx] = None
                    # print(
                    #     f"Slice iter for {self.worker_config.rank}:{self.worker_config.rank_worker_id()} "
                    #     f"[{slice_offsets[slice_state.index]}, {slice_offsets[slice_state.index + 1]}] exhausted at {slice_state.current}, "
                    #     f"no next slice, probs={active_slice_probs.tolist()}"
                    # )
                if self.worker_config.should_log(level=2):
                    self.worker_config.worker_log(
                        {
                            "t": "WebdatasetSampleLoaderDataset._slices_iter.exhausted",
                            "r": self.worker_config.rank,
                            "w": self.worker_config.rank_worker_id(),
                            "remaining": len(pending_slice_indexes),
                            "count": self._sample_count,
                            "epoch": self._epoch_count,
                            "epoch_count": self._epoch_sample_count,
                            "probs": active_slice_probs.tolist(),
                        }
                    )
            if sample.data[0] is not None:
                # Otherwise the sample was skipped.
                if self.worker_config.should_log(level=1):
                    self.worker_config.worker_log(
                        {
                            "t": "WebdatasetSampleLoaderDataset._slices_iter.yield",
                            "r": self.worker_config.rank,
                            "w": self.worker_config.rank_worker_id(),
                            "index": sample.__restore_key__[1],
                            "key": sample.data[0]["__key__"],
                            "shard": sample.data[0]["__shard__"],
                            "count": self._sample_count,
                            "epoch": self._epoch_count,
                            "epoch_count": self._epoch_sample_count,
                        }
                    )
                # Now, yield the sample
                yield sample
                del sample
        if self.worker_config.should_log(level=2):
            self.worker_config.worker_log(
                {
                    "t": "WebdatasetSampleLoaderDataset._slices_iter.all_exhausted",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "count": self._sample_count,
                    "epoch": self._epoch_count,
                    "epoch_count": self._epoch_sample_count,
                }
            )

        # Epoch has finished, reset states.
        self._epoch_count += 1
        self._epoch_sample_count = 0
        self._pending_slice_indexes = None
        self._pending_slices_offset = None
        # print(
        #     f"slice iters exhausted for {self.worker_config.rank}:{self.worker_config.rank_worker_id()} after {cnt} samples"
        # )

    def len_worker(self, worker_idx: int | None = None) -> int:
        if worker_idx is None:
            self.worker_config.assert_worker()
            worker_idx = self.worker_config.rank_worker_id()
        worker_slice_offsets = self.workers_slice_offsets[worker_idx]
        return worker_slice_offsets[-1] - worker_slice_offsets[0]

    def worker_has_samples(self) -> bool:
        self.worker_config.assert_worker()
        self.ensure_slice_offsets()
        assert self.slice_offsets is not None
        return len(self.slice_offsets) > 1

    def __iter__(self) -> Iterator[RawSampleData]:
        self.worker_config.assert_worker()

        self.ensure_slice_offsets()
        assert self.slice_offsets is not None

        if self.worker_config.should_log(level=1):
            self.worker_config.worker_log(
                {
                    "t": "WebdatasetSampleLoaderDataset.__iter__",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "slice_offsets": self.slice_offsets,
                    "parallel_slice_iters": self.parallel_slice_iters,
                    "shuffle_over_epochs": self.shuffle_over_epochs,
                }
            )

        if len(self.slice_offsets) <= 1:
            return

        yield from self._slices_iter()

    def can_restore_sample(self) -> bool:
        return True

    def assert_can_restore(self) -> None:
        pass

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> RawSampleData:
        # Key is: ("Webdataset", index)
        # The key is joined in the dataset's typed joining (i.e. load_sample of JoinedWebdatasetFactory).
        id, index = restore_key
        assert id == "Webdataset"
        assert isinstance(index, int)
        return self._get_sample(index)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "joins": len(self.join_readers),
            "len": len(self),
            "slice_offsets": [
                list(worker_slice_offsets) for worker_slice_offsets in self.workers_slice_offsets
            ],
            "worker_config": self.worker_config.config(),
            "shuffle_over_epochs": self.shuffle_over_epochs,
            "parallel_slice_iters": self.parallel_slice_iters,
        }

    def __str__(self):
        return f"WebdatasetSampleLoaderDataset(join_readers={self.join_readers}, shuffle_over_epochs={self.shuffle_over_epochs}, parallel_slice_iters={self.parallel_slice_iters})"
