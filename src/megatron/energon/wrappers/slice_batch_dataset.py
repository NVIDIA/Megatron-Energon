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
from megatron.energon.flavors.base_dataset import SavableDataset, add_sample_restore_key
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers._log_exception import log_exception
from megatron.energon.wrappers.base import (
    BaseSingleWrapperDataset,
    BaseSingleWrapperMergedState,
    BaseSingleWrapperState,
    get_sample_restore_key,
    wrap_worker_sample_index,
    wrap_worker_sample_index_ctx,
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
    buffer: SampleBufferState
    sample_index: int


@dataclass
class SliceBatchMergedState(BaseSingleWrapperMergedState):
    buffer: SampleBufferMergedState
    sample_indexes: List[int]


class SliceBatchDataset(
    BaseSingleWrapperDataset[T_batch_sample, T_batch], Generic[T_batch_sample, T_batch]
):
    """This dataset wrapper transforms samples of a dataset into chunks/packs of samples, which are
    then combined into a batch."""

    buffer_size: int
    _active_buffer: SavableSampleBuffer[T_batch_sample]
    batcher: Callable[[List[T_batch_sample]], T_batch]
    slicer: Callable[[List[T_batch_sample]], List[List[T_batch]]]
    drop_last: bool
    error_handler: Callable[[Exception, List[T_batch_sample]], None]
    worker_config: WorkerConfig
    _sample_index: List[int]
    #: The buffer for collecting samples before slicing
    _collecting_buffer: SavableSampleBuffer

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
        self._collecting_buffer = SavableSampleBuffer(dataset, worker_config)

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
        worker_index = self.worker_config.rank_worker_id()

        # The source dataset
        src_iter = iter(wrap_worker_sample_index(self.dataset, self._sample_index, worker_index))

        def _fill_buffer(ensure_full: bool = False):
            """Fill the collecting buffer with samples from the source dataset."""
            if ensure_full:
                step_samples = self.buffer_size - len(self._collecting_buffer)
            else:
                step_samples = avg_samples_per_slice
            while len(self._collecting_buffer) < self.buffer_size and step_samples > 0:
                try:
                    sample_idx, sample = next(src_iter)
                except StopIteration:
                    return False
                self._collecting_buffer.append(sample_idx, sample)
                step_samples -= 1
            return True

        # The active slicer
        slicer = None

        def next_slicer():
            """Create a new slicer from the collecting buffer."""
            nonlocal slicer
            assert slicer is None
            if len(self._collecting_buffer) > 0:
                samples = list(self._collecting_buffer)
                try:
                    slicer = iter(self.slicer(samples))
                except SkipSample:
                    slicer = iter([])
                except SYSTEM_EXCEPTIONS:
                    raise FatalSampleError.from_sample(samples)
                except Exception as e:
                    self.error_handler(e, samples)
                self._collecting_buffer.clear()

        # For updating the running average
        cur_num_slices = 0
        # The running average for optimal buffer filling
        avg_samples_per_slice = self.buffer_size

        def next_batch():
            """Yield the next batch(es) from the slicer."""
            nonlocal cur_num_slices, avg_samples_per_slice, slicer
            assert slicer is not None
            try:
                slice = next(slicer)
            except (StopIteration, SkipSample):
                # Update the running mean for better loading speed
                avg_samples_per_slice = (
                    0.9 * avg_samples_per_slice + 0.1 * self.buffer_size / cur_num_slices
                )
                cur_num_slices = 0
                slicer = None
                return
            except SYSTEM_EXCEPTIONS:
                raise FatalSampleError.from_sample(slice)
            except Exception as e:
                self.error_handler(e, slice)
            cur_num_slices += 1
            try:
                batch_restore_key = tuple(get_sample_restore_key(sample) or () for sample in slice)
                batch_sample = self.batcher(slice)
                if isinstance(batch_sample, Generator):
                    assert inspect.isgeneratorfunction(self.batcher)
                    for batch_sub_idx, inner_batch_sample in enumerate(batch_sample):
                        yield add_sample_restore_key(
                            inner_batch_sample,
                            batch_restore_key,
                            batch_sub_idx,
                            src=self,
                        )
                else:
                    yield add_sample_restore_key(batch_sample, batch_restore_key, src=self)
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
                break
            # Create a new slicer if necessary
            if slicer is None:
                # Ensure that the buffer is filled completely
                if not _fill_buffer(ensure_full=True):
                    break
                next_slicer()
            yield from next_batch()
        # Yield the remaining slices, flushing the collecting buffer
        while True:
            if slicer is None:
                if len(self._collecting_buffer) == 0:
                    break
                next_slicer()
            yield from next_batch()

    def save_state(self) -> SliceBatchState:
        return SliceBatchState.extend(
            super().save_state(),
            buffer=self._collecting_buffer.save_state(),
            sample_index=self._sample_index[self.worker_config.rank_worker_id()],
        )

    def merge_states(self, states: List[SliceBatchState]) -> SliceBatchMergedState:
        assert all(s is None or isinstance(s, SliceBatchState) for s in states)
        return SliceBatchMergedState.extend(
            super().merge_states(states),
            buffer=self._collecting_buffer.merge_states(
                [None if s is None else s.buffer for s in states]
            ),
            sample_indexes=[0 if state is None else state.sample_index for state in states],
        )

    def restore_state(self, state: Optional[SliceBatchMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._collecting_buffer.restore_state(None)
            self._sample_index = [0] * max(self.worker_config.num_workers, 1)
        else:
            assert isinstance(state, SliceBatchMergedState)
            self._collecting_buffer.restore_state(state.buffer)
            self._sample_index = state.sample_indexes

    def can_restore_sample(self) -> bool:
        # Cannot really verify if the returned elements contain a __restore_key__.
        # If the user wants to use this, well...
        return self.batcher_stateless and self.dataset.can_restore_sample()

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_batch:
        # We need to store multiple indices to restore a batch.
        assert self.batcher_stateless
        if inspect.isgeneratorfunction(self.batcher):
            id, sample_idx, batch_sub_idx = index[:3]
            assert id == type(self).__name__
            index = index[3:]
        else:
            id, sample_idx = index[:2]
            assert id == type(self).__name__
            index = index[2:]
        batch = []
        for sample_offset, inner_idx in enumerate(index):
            with wrap_worker_sample_index_ctx(sample_idx + sample_offset):
                batch.append(self.dataset.restore_sample(inner_idx))
        batch_sample = self.batcher(batch)
        if isinstance(batch_sample, Generator):
            assert inspect.isgeneratorfunction(self.batcher)
            for cur_batch_sub_idx, item in enumerate(batch_sample):
                if cur_batch_sub_idx == batch_sub_idx:
                    return add_sample_restore_key(
                        item,
                        sample_idx,
                        batch_sub_idx,
                        src=self,
                    )
            assert False, f"Batch sub-index {batch_sub_idx} not found in batch"
        else:
            return add_sample_restore_key(batch_sample, sample_idx, src=self)

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
