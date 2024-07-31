# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from megatron.energon.errors import SYSTEM_EXCEPTIONS, FatalSampleError
from megatron.energon.flavors.base_dataset import SavableDataset, set_sample_restore_key
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers._log_exception import log_exception
from megatron.energon.wrappers.base import (
    BaseSingleWrapperDataset,
    BaseSingleWrapperMergedState,
    BaseSingleWrapperState,
    SampleIndex,
    get_sample_restore_key,
)
from megatron.energon.wrappers.buffer import (
    SampleBufferMergedState,
    SampleBufferState,
    SavableSampleBuffer,
)
from megatron.energon.wrappers.skip import SkipSample

T_batch = TypeVar("T_batch", covariant=True)
T_batch_sample = TypeVar("T_batch_sample", covariant=True)


@dataclass
class SliceBatchState(BaseSingleWrapperState):
    reading_buffer: SampleBufferState
    slice_buffer: SampleBufferState
    slice_lengths: List[int]
    slicer_index: int
    batcher_index: int


@dataclass
class SliceBatchMergedState(BaseSingleWrapperMergedState):
    reading_buffer: SampleBufferMergedState
    slice_buffer: SampleBufferMergedState
    slice_lengths: List[List[int]]
    slicer_index: List[int]
    batcher_index: List[int]


class SliceBatchDataset(
    BaseSingleWrapperDataset[T_batch_sample, T_batch], Generic[T_batch_sample, T_batch]
):
    """This dataset wrapper transforms samples of a dataset into chunks/packs of samples, which are
    then combined into a batch."""

    buffer_size: int
    slicer: Callable[[List[T_batch_sample]], List[List[T_batch]]]
    batcher: Callable[[List[T_batch_sample]], T_batch]
    batcher_stateless: bool
    error_handler: Callable[[Exception, List[T_batch_sample]], None]
    worker_config: WorkerConfig

    #: The buffer for collecting samples before slicing.
    _reading_buffer: SavableSampleBuffer
    #: Contains the sliced samples, indexed by `_slice_lengths`.
    _slice_buffer: SavableSampleBuffer
    #: Length of the slices in the slice buffer.
    _slice_lengths: List[List[int]]
    #: Sample index for the slicer function
    _slicer_sample_index: SampleIndex
    #: Sample index for the batcher function
    _batcher_sample_index: SampleIndex

    def __init__(
        self,
        dataset: SavableDataset[T_batch_sample],
        buffer_size: int,
        slicer: Callable[[List[T_batch_sample]], List[List[T_batch_sample]]],
        batcher: Callable[[List[T_batch_sample]], T_batch],
        *,
        batcher_stateless: bool = False,
        error_handler: Callable[[Exception, List[T_batch_sample]], None] = log_exception,
        worker_config: WorkerConfig,
    ):
        """Construct a BufferedBatchDataset. The internal buffer is filled during the construction
        of the next batch.

        Args:
            dataset: The input dataset to wrap
            buffer_size: The desired size of the buffer for slicing. Last buffer may be smaller.
            slicer: Function which slices the buffer into smaller chunks. May raise
                :exc:`megatron.energon.SkipSample` to skip a slice.
            batcher: Function which combines separate samples into a single object. May raise
                :exc:`megatron.energon.SkipSample` to skip a batch.
            batcher_stateless: If True, the batcher is stateless, thus samples can be stored/
                restored.
            error_handler: Function which handles exceptions raised by the batcher. The default
                implementation logs the exception.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset)
        self.buffer_size = buffer_size
        self.slicer = slicer
        self.batcher = batcher
        self.batcher_stateless = batcher_stateless
        self.error_handler = error_handler
        self.worker_config = worker_config
        self._reading_buffer = SavableSampleBuffer(dataset, worker_config)
        self._slice_buffer = SavableSampleBuffer(dataset, worker_config)
        self._slice_lengths = [[] for _ in range(max(worker_config.num_workers, 1))]
        self._slicer_sample_index = SampleIndex(worker_config, src=self)
        self._batcher_sample_index = SampleIndex(worker_config, src=self)

    def __len__(self):
        n_samples = len(self.dataset)
        num_workers = max(self.worker_config.num_workers, 1)
        n_samples_per_worker_floor = n_samples // num_workers
        remaining_n_sample_workers = n_samples % num_workers
        n_batches_per_worker_floor = n_samples_per_worker_floor // self.buffer_size
        if n_samples_per_worker_floor % self.buffer_size != 0 and not self.drop_last:
            n_batches_per_worker_floor += 1
        # Correct number of batches for the workers which yield 1 more sample (to balance)
        n_batches_per_worker_ceil = (n_samples_per_worker_floor + 1) // self.buffer_size
        if n_batches_per_worker_ceil % self.buffer_size != 0 and not self.drop_last:
            n_batches_per_worker_ceil += 1

        return (
            n_batches_per_worker_floor * (num_workers - remaining_n_sample_workers)
            + n_batches_per_worker_ceil * remaining_n_sample_workers
        )

    def __iter__(self) -> Iterator[T_batch]:
        worker_idx = self.worker_config.rank_worker_id()
        slice_lengths = self._slice_lengths[worker_idx]
        # The source dataset
        src_iter = iter(self.dataset)

        self._slice_buffer.worker_start()
        self._reading_buffer.worker_start()

        def _fill_buffer():
            """Fill the collecting buffer with samples from the source dataset."""
            while len(self._reading_buffer) + len(self._slice_buffer) < self.buffer_size:
                try:
                    sample = next(src_iter)
                except StopIteration:
                    return False
                self._reading_buffer.append(sample)
            return True

        def next_slicer():
            """Create a new slicer from the collecting buffer."""
            assert len(self._slice_buffer) == 0
            if len(self._reading_buffer) > 0:
                # Take all samples that have been written in the beginning of the buffer and slice
                # them
                samples = list(self._reading_buffer)
                # Clear buffer and slices
                self._reading_buffer.clear()
                slice_lengths.clear()
                # Now slice
                try:
                    with self._slicer_sample_index.ctx():
                        slices = self.slicer(samples)
                except SkipSample:
                    slices = []
                except SYSTEM_EXCEPTIONS:
                    raise FatalSampleError.from_sample(samples)
                except Exception as e:
                    self.error_handler(e, samples)
                # Set the slice offsets and assign slices back to the buffer
                for slice in slices:
                    self._slice_buffer.extend(slice)
                    slice_lengths.append(len(slice))

        def next_batch():
            """Yield the next batch(es) from the slicer."""
            slice = list(self._slice_buffer[: slice_lengths[0]])
            del self._slice_buffer[: slice_lengths[0]]
            del slice_lengths[0]
            try:
                batch_restore_keys = tuple(get_sample_restore_key(sample) for sample in slice)
                with self._batcher_sample_index.ctx() as batch_idx:
                    batch_sample = self.batcher(slice)
                if isinstance(batch_sample, Generator):
                    assert inspect.isgeneratorfunction(self.batcher)
                    for batch_sub_idx, (batch_idx, inner_batch_sample) in enumerate(
                        self._batcher_sample_index.iter_ctx(batch_sample, batch_idx)
                    ):
                        yield set_sample_restore_key(
                            inner_batch_sample,
                            batch_idx,
                            batch_sub_idx,
                            *batch_restore_keys,
                            src=self,
                        )
                else:
                    yield set_sample_restore_key(
                        batch_sample,
                        batch_idx,
                        *batch_restore_keys,
                        src=self,
                    )
            except SkipSample:
                pass
            except SYSTEM_EXCEPTIONS:
                raise FatalSampleError.from_sample(slice)
            except Exception as e:
                self.error_handler(e, slice)

        # Main loop:
        # 1. Fill the buffer, i.e. take a few samples from the source dataset and put them into the
        #   collecting buffer.
        # 2. If there is no active slicer (i.e. creating slices from the buffer), create a new one.
        # 3. Yield the next batch from the next slice by calling the batcher.
        # Break out of the main loop when the source is exhausted
        while True:
            # Fill a portion of the buffer
            if not _fill_buffer():
                if len(self._reading_buffer) > 0:
                    next_slicer()
                break
            # Create a new slicer if necessary
            if len(slice_lengths) == 0:
                assert len(self._slice_buffer) == 0
                assert len(self._reading_buffer) == self.buffer_size
                next_slicer()
            yield from next_batch()
        # Yield the remaining slices, flushing the collecting buffer
        while len(slice_lengths) > 0:
            yield from next_batch()

    def save_state(self) -> SliceBatchState:
        return SliceBatchState.extend(
            super().save_state(),
            reading_buffer=self._reading_buffer.save_state(),
            slice_buffer=self._slice_buffer.save_state(),
            slice_lengths=list(self._slice_lengths[self.worker_config.rank_worker_id()]),
            batcher_index=self._batcher_sample_index.save_state(),
            slicer_index=self._slicer_sample_index.save_state(),
        )

    def merge_states(self, states: List[SliceBatchState]) -> SliceBatchMergedState:
        assert all(s is None or isinstance(s, SliceBatchState) for s in states)
        return SliceBatchMergedState.extend(
            super().merge_states(states),
            reading_buffer=self._reading_buffer.merge_states(
                [None if s is None else s.reading_buffer for s in states]
            ),
            slice_buffer=self._slice_buffer.merge_states(
                [None if s is None else s.slice_buffer for s in states]
            ),
            slice_lengths=[[0, 0] if s is None else s.slice_lengths for s in states],
            batcher_index=self._batcher_sample_index.merge_states(
                [0 if state is None else state.batcher_index for state in states]
            ),
            slicer_index=self._slicer_sample_index.merge_states(
                [0 if state is None else state.slicer_index for state in states]
            ),
        )

    def restore_state(self, state: Optional[SliceBatchMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._reading_buffer.restore_state(None)
            self._slice_buffer.restore_state(None)
            self._slice_lengths = [[] for _ in range(max(self.worker_config.num_workers, 1))]
            self._slicer_sample_index.restore_state(None)
            self._batcher_sample_index.restore_state(None)
        else:
            assert isinstance(state, SliceBatchMergedState)
            self._reading_buffer.restore_state(state.reading_buffer)
            self._slice_buffer.restore_state(state.slice_buffer)
            self._slice_lengths = state.slice_lengths
            self._slicer_sample_index.restore_state(state.slicer_index)
            self._batcher_sample_index.restore_state(state.batcher_index)

    def can_restore_sample(self) -> bool:
        # Cannot really verify if the returned elements contain a __restore_key__.
        # If the user wants to use this, well...
        return self.batcher_stateless and self.dataset.can_restore_sample()

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_batch:
        # We need to store multiple indices to restore a batch.
        assert (
            self.batcher_stateless
        ), f"Batcher {self.batcher} must be stateless to restore samples."
        if inspect.isgeneratorfunction(self.batcher):
            id, batch_idx, batch_sub_idx, *batch_restore_keys = index
            assert id == type(self).__name__
        else:
            id, batch_idx, *batch_restore_keys = index
            assert id == type(self).__name__
        batch = [self.dataset.restore_sample(inner_idx) for inner_idx in batch_restore_keys]
        with self._batcher_sample_index.ctx(batch_idx):
            batch_sample = self.batcher(batch)
        if isinstance(batch_sample, Generator):
            assert inspect.isgeneratorfunction(self.batcher)
            for cur_batch_sub_idx, (batch_idx, inner_batch_sample) in enumerate(
                self._batcher_sample_index.iter_ctx(batch_sample, batch_idx)
            ):
                if cur_batch_sub_idx == batch_sub_idx:
                    return set_sample_restore_key(
                        inner_batch_sample,
                        batch_idx,
                        batch_sub_idx,
                        *batch_restore_keys,
                        src=self,
                    )
            assert False, f"Batch sub-index {batch_sub_idx} not found in batch"
        else:
            return set_sample_restore_key(batch_sample, batch_idx, *batch_restore_keys, src=self)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "buffer_size": self.buffer_size,
            "slicer": self._function_config(self.slicer),
            "batcher": self._function_config(self.batcher),
            "batcher_stateless": self.batcher_stateless,
            "error_handler": self._function_config(self.error_handler),
            "worker_config": self.worker_config.config(),
            "dataset": self.dataset.config(),
        }

    def __str__(self):
        return f"BatchDataset(buffer_size={self.buffer_size}, slicer={self.slicer}, batcher={self.batcher}, dataset={self.dataset})"
