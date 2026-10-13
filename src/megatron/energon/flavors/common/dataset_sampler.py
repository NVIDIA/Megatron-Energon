# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import ExitStack
from typing import Any, Dict, Generator, Iterator, List, Optional, Sequence, Tuple, Union

import torch

from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.flavors.common.reader import IndexedSampleReader
from megatron.energon.flavors.common.sample_record import SampleRecord
from megatron.energon.rng import FeistelPermutation, WorkerRng
from megatron.energon.worker import WorkerConfig


@edataclass
class RawSampleData:
    """Represents the iteration state of a single slice slice to the index."""

    #: Index of the sample. This is also the restore key
    __restore_key__: Tuple[str, int]
    #: The sample data
    data: Tuple[Optional[SampleRecord], ...]


@edataclass
class SliceState:
    """Represents the iteration state of a single slice slice to the index."""

    #: The slice index of this slice state
    index: int
    #: The actual state: The global sample offset (`slice[index] <= offset < slice[index + 1]``)
    current: int


class DatasetSampler(SavableDataset[RawSampleData]):
    """Samples indexed dataset slices across workers and epochs."""

    #: The readers for each joined dataset
    join_readers: Sequence[IndexedSampleReader]

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

    #: Seed of the slice permutation of the current epoch. None if the slices are not permuted.
    _pending_slices_seed: Optional[int]
    #: Number of slices of the current epoch that have already been opened. None if no epoch is in
    # progress.
    _pending_slices_offset: Optional[int]
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
        "_pending_slices_seed",
        "_pending_slices_offset",
        "_active_slice_state",
        "_sample_count",
        "_epoch_count",
        "_epoch_sample_count",
    )

    def __init__(
        self,
        join_readers: Sequence[IndexedSampleReader],
        workers_sample_slice_offsets: Sequence[Sequence[int]],
        *,
        worker_config: WorkerConfig,
        shuffle_over_epochs: Optional[int] = None,
        parallel_slice_iters: int = 1,
        restore_key_kind: str = "Webdataset",
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
        self.restore_key_kind = restore_key_kind

        # Store the slices for all workers
        # The slices for the current worker, will have to be extracted from this list later
        self.workers_slice_offsets = workers_sample_slice_offsets
        self.slice_offsets = None

        self.reset_state_own()

        assert shuffle_over_epochs is None or shuffle_over_epochs == -1 or shuffle_over_epochs >= 1
        assert self.parallel_slice_iters >= 1

    def close(self) -> None:
        with ExitStack() as stack:
            for reader in self.join_readers:
                stack.callback(reader.close)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors may run during interpreter shutdown.
            pass

    def reset_state_own(self) -> None:
        self._worker_rng = WorkerRng(self.worker_config)
        self._pending_slices_seed = None
        self._pending_slices_offset = None
        self._active_slice_state = [None] * self.parallel_slice_iters
        self._sample_count = 0
        self._epoch_count = 0
        self._epoch_sample_count = 0

    def ensure_slice_offsets(self) -> None:
        self.worker_config.assert_worker()

        if self.slice_offsets is None:
            self.slice_offsets = self.workers_slice_offsets[self.worker_config.rank_worker_id()]

    @staticmethod
    def _slice_offsets_config(slice_offsets: Sequence[int]) -> list[int] | dict[str, int | str]:
        if isinstance(slice_offsets, range):
            return {
                "type": "range",
                "start": slice_offsets.start,
                "stop": slice_offsets.stop,
                "step": slice_offsets.step,
            }
        return list(slice_offsets)

    def _get_sample(self, index: int) -> RawSampleData:
        return RawSampleData(
            __restore_key__=(self.restore_key_kind, index),
            data=tuple(reader[index] for reader in self.join_readers),
        )

    def _pending_slices(self) -> Sequence[int]:
        """Returns the slice order of the current epoch. Entries are taken modulo the number of
        slices, as the order may span multiple epochs."""
        assert self.slice_offsets is not None

        num_slices = len(self.slice_offsets) - 1
        if self.shuffle_over_epochs is None:
            # No shuffling
            return range(num_slices)
        if self.shuffle_over_epochs == -1:
            # Shuffle with replacement (i.e. infinite epochs), slices are drawn in
            # `_pop_pending_slice`.
            return ()
        # Shuffle without replacement (potentially over multiple epochs)
        assert self._pending_slices_seed is not None
        return FeistelPermutation(num_slices * self.shuffle_over_epochs, self._pending_slices_seed)

    def _pop_pending_slice(self, pending_slices: Sequence[int]) -> Optional[int]:
        """Returns the next slice index of the current epoch, or None if there is none left."""
        assert self.slice_offsets is not None
        assert self._pending_slices_offset is not None

        num_slices = len(self.slice_offsets) - 1
        if self.shuffle_over_epochs == -1:
            # Randomly select a new slice directly (with replacement)
            return self._worker_rng.randbelow(num_slices)
        if self._pending_slices_offset >= len(pending_slices):
            return None
        # Take the next slice (without replacement)
        slice_index = pending_slices[self._pending_slices_offset] % num_slices
        self._pending_slices_offset += 1
        return slice_index

    def _slices_iter(self) -> Generator[RawSampleData, None, None]:
        """Iterates the samples in a list of slices, possibly using multiple parallel iterators over
        the slices."""

        assert self.slice_offsets is not None

        active_slice_probs = torch.zeros(self.parallel_slice_iters, dtype=torch.float32)
        active_slices = self._active_slice_state

        def slice_at(idx: int) -> SliceState:
            assert self.slice_offsets is not None
            return SliceState(
                index=idx,
                current=self.slice_offsets[idx],
            )

        # Weight the slices by their size to get a more even distribution of samples
        if any(s is not None for s in active_slices) or self._pending_slices_offset is not None:
            # Having an active state, or pending slices. This means we are resuming an epoch.
            assert self._pending_slices_offset is not None
            pending_slices = self._pending_slices()

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
                        "t": "DatasetSampler._slices_iter.resume_epoch",
                        "r": self.worker_config.rank,
                        "w": self.worker_config.rank_worker_id(),
                        "pending_slices_remaining": len(pending_slices)
                        - self._pending_slices_offset,
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
            if self.shuffle_over_epochs is not None and self.shuffle_over_epochs >= 1:
                self._pending_slices_seed = self._worker_rng.randbelow(torch.iinfo(torch.int64).max)
            self._pending_slices_offset = 0
            pending_slices = self._pending_slices()

            if self.worker_config.should_log(level=1):
                self.worker_config.worker_log(
                    {
                        "t": "DatasetSampler._slices_iter.next_epoch",
                        "r": self.worker_config.rank,
                        "w": self.worker_config.rank_worker_id(),
                        "pending_slices_remaining": len(pending_slices)
                        - self._pending_slices_offset,
                        "count": self._sample_count,
                        "epoch": self._epoch_count,
                        "epoch_count": self._epoch_sample_count,
                        "probs": active_slice_probs.tolist(),
                        "shuffle_over_epochs": self.shuffle_over_epochs,
                    }
                )

            # List of slice iterators, always of length `parallel_slice_iters`. May contain `None`.
            active_slices.clear()
            # Fill up the slice iterators
            while len(active_slices) < self.parallel_slice_iters:
                slice_index = self._pop_pending_slice(pending_slices)
                if slice_index is None:
                    break
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
                next_idx = self._pop_pending_slice(pending_slices)
                if next_idx is not None:
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
                    #     f"{len(pending_slices) - self._pending_slices_offset} slices left, probs={active_slice_probs.tolist()}"
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
                            "t": "DatasetSampler._slices_iter.exhausted",
                            "r": self.worker_config.rank,
                            "w": self.worker_config.rank_worker_id(),
                            "remaining": len(pending_slices) - self._pending_slices_offset,
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
                            "t": "DatasetSampler._slices_iter.yield",
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
                    "t": "DatasetSampler._slices_iter.all_exhausted",
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
        self._pending_slices_seed = None
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
                    "t": "DatasetSampler.__iter__",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "slice_offsets": self._slice_offsets_config(self.slice_offsets),
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
        # Key is: (self.restore_key_kind, index)
        # The key is joined in the dataset's typed joining (i.e. load_sample of JoinedWebdatasetFactory).
        id, index = restore_key
        assert id == self.restore_key_kind
        assert isinstance(index, int)
        return self._get_sample(index)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "joins": len(self.join_readers),
            "len": len(self),
            "slice_offsets": [
                self._slice_offsets_config(worker_slice_offsets)
                for worker_slice_offsets in self.workers_slice_offsets
            ],
            "worker_config": self.worker_config.config(),
            "shuffle_over_epochs": self.shuffle_over_epochs,
            "parallel_slice_iters": self.parallel_slice_iters,
        }

    def __str__(self):
        return f"DatasetSampler(join_readers={self.join_readers}, shuffle_over_epochs={self.shuffle_over_epochs}, parallel_slice_iters={self.parallel_slice_iters})"
