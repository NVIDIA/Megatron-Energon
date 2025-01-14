# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Hashable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from megatron.energon.errors import SYSTEM_EXCEPTIONS, FatalSampleError
from megatron.energon.flavors.base_dataset import (
    MergedState,
    SavableDataset,
    State,
    set_sample_restore_key,
)
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers._log_exception import log_exception
from megatron.energon.wrappers.base import (
    BaseSingleWrapperDataset,
    BaseSingleWrapperMergedState,
    BaseSingleWrapperState,
    SampleIndex,
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
class BucketState(State):
    key: Hashable
    batch_size: int
    samples: SampleBufferState


@dataclass
class BucketMergedState(MergedState):
    key: Hashable
    batch_size: int
    samples: SampleBufferState


@dataclass
class GroupBatchState(BaseSingleWrapperState):
    bucket_sample_index: int
    batch_sample_index: int
    buckets: List[BucketState]


@dataclass
class GroupBatchMergedState(BaseSingleWrapperMergedState):
    bucket_sample_index: List[int]
    batch_sample_index: List[int]
    buckets: List[List[BucketMergedState]]


@dataclass
class Bucket(Generic[T_batch_sample]):
    batch_size: int

    samples: SavableSampleBuffer[T_batch_sample]


class GroupBatchDataset(
    BaseSingleWrapperDataset[T_batch_sample, T_batch], Generic[T_batch_sample, T_batch]
):
    """This dataset wrapper transforms a dataset of samples into a dataset of batches, grouped by some criterion.
    The length is not correct, as this function can not predict the number of batches as there is no fixed batch size,
    instead it returns the inner dataset size.
    An example use case is: Image-Text samples, which are to be grouped by the image size into three
    size categories (e.g. 128x128, 256x256, 512x512) for efficient augmentation and batching.
    """

    dataset: SavableDataset[T_batch_sample]
    sample_group_key: Callable[[T_batch_sample], Tuple[Hashable, Optional[int]]]
    batcher: Callable[[List[T_batch_sample]], T_batch]
    drop_last: bool
    error_handler: Callable[[Exception, List[T_batch_sample]], None]
    _group_key_sample_index: SampleIndex
    _batch_sample_index: SampleIndex
    _buckets: List[Dict[Hashable, Bucket[T_batch_sample]]]

    def __init__(
        self,
        dataset: SavableDataset[T_batch_sample],
        fixed_batch_size: Optional[int],
        sample_group_key: Callable[[T_batch_sample], Tuple[Hashable, Optional[int]]],
        batcher: Callable[[List[T_batch_sample]], T_batch],
        *,
        batcher_stateless: bool = False,
        batcher_config: Optional[Union[Dict[str, Any], Callable[[], Dict[str, Any]]]] = None,
        drop_last: bool = False,
        error_handler: Callable[[Exception, List[T_batch_sample]], None] = log_exception,
        worker_config: WorkerConfig,
    ):
        """Construct a GroupBatchDataset.

        Args:
            dataset: The input dataset to wrap
            fixed_batch_size: Fixed batch size to use for all buckets. If None, the batch size is determined by the sample_group_key function.
            sample_group_key: Function which determines the bucket of a sample.
            batcher: Function which combines separate samples into a single object. May raise
                :exc:`megatron.energon.SkipSample` to skip a sample.
            drop_last: If True, the last batch is dropped if it is smaller than the batch size.
            error_handler: Handler for errors. Defaults to logging and ignoring the exception.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset, worker_config=worker_config)
        self.fixed_batch_size = fixed_batch_size
        self.sample_group_key = sample_group_key
        self.batcher = batcher
        self.batcher_stateless = batcher_stateless
        self.batcher_config = batcher_config
        self.drop_last = drop_last
        self.error_handler = error_handler
        self._group_key_sample_index = SampleIndex(worker_config, src=self)
        self._batch_sample_index = SampleIndex(worker_config, src=self)
        self._buckets = [{} for _ in range(max(self.worker_config.num_workers, 1))]
        assert not inspect.isgeneratorfunction(
            batcher
        ), f"Batcher {batcher} must not be a generator function for grouped batching."

    def __len__(self):
        # Return an upper bound. This is for sure not correct.
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_batch]:
        worker_idx = self.worker_config.rank_worker_id()
        buckets = self._buckets[worker_idx]
        # Cleanup other states
        for i in range(self.worker_config.num_workers):
            if i != worker_idx:
                self._buckets[i].clear()

        if buckets is None:
            buckets = self._buckets[worker_idx] = dict()

        # Load saved state if available
        for bucket in buckets.values():
            bucket.samples.worker_start()

        def flush(bucket: Bucket[T_batch_sample]) -> Generator[T_batch, None, None]:
            batch_items, sample_restore_keys = bucket.samples.flush()
            try:
                with self._batch_sample_index.ctx() as sample_idx:
                    batch_sample = self.batcher(batch_items)
                    assert not isinstance(
                        batch_sample, Generator
                    ), f"Batcher {self.batcher} returned a generator, which is not supported for grouped batching yet."
                set_sample_restore_key(batch_sample, sample_idx, *sample_restore_keys, src=self)
                yield batch_sample
            except SkipSample:
                pass
            except SYSTEM_EXCEPTIONS:
                raise FatalSampleError.from_sample(batch_items)
            except Exception as e:
                self.error_handler(e, batch_items)

        # Add samples to the buckets
        for sample in self.dataset:
            try:
                with self._group_key_sample_index.ctx():
                    bucket_key, batch_size = self.sample_group_key(sample)
                    assert (batch_size is None) != (self.fixed_batch_size is None), (
                        f"A sample in group for key {bucket_key} returned batch size {batch_size}, but fixed "
                        f"batch size is set to {self.fixed_batch_size}. One of the two should be None."
                    )
                    if self.fixed_batch_size is not None:
                        batch_size = self.fixed_batch_size
            except SkipSample:
                continue
            except SYSTEM_EXCEPTIONS:
                raise FatalSampleError.from_sample(sample)
            except Exception as e:
                self.error_handler(e, [sample])
                continue
            bucket = buckets.get(bucket_key)
            if bucket is None:
                buckets[bucket_key] = bucket = Bucket(
                    batch_size=batch_size,
                    samples=SavableSampleBuffer(self.dataset, worker_config=self.worker_config),
                )
            else:
                assert (
                    bucket.batch_size == batch_size
                ), f"Got different batch size for group {bucket_key}: {bucket.batch_size} != {batch_size}."
            bucket.samples.append(sample)
            if len(bucket.samples) >= bucket.batch_size:
                yield from flush(bucket)
        # Flush out last samples
        if not self.drop_last:
            for bucket in buckets.values():
                if len(bucket.samples) > 0:
                    yield from flush(bucket)
        # Clear the buckets
        self._buckets[worker_idx].clear()

    def save_state(self) -> GroupBatchState:
        rank = self.worker_config.rank_worker_id()
        return GroupBatchState.extend(
            super().save_state(),
            bucket_sample_index=self._group_key_sample_index.save_state(),
            batch_sample_index=self._batch_sample_index.save_state(),
            buckets=(
                [
                    BucketState(
                        key=key,
                        batch_size=bucket.batch_size,
                        samples=bucket.samples.save_state(),
                    )
                    for key, bucket in self._buckets[rank].items()
                ]
                if self._buckets[rank] is not None
                else []
            ),
        )

    def merge_states(self, states: List[Optional[GroupBatchState]]) -> GroupBatchMergedState:
        assert all(s is None or isinstance(s, GroupBatchState) for s in states)
        return GroupBatchMergedState.extend(
            super().merge_states(states),
            bucket_sample_index=self._group_key_sample_index.merge_states(
                [0 if s is None else s.bucket_sample_index for s in states]
            ),
            batch_sample_index=self._batch_sample_index.merge_states(
                [0 if s is None else s.batch_sample_index for s in states]
            ),
            buckets=[(state.buckets if state else []) for state in states],
        )

    def restore_state(self, state: Optional[GroupBatchMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._buckets = [{} for _ in range(max(self.worker_config.num_workers, 1))]
        else:
            assert isinstance(state, GroupBatchMergedState)
            self._group_key_sample_index.restore_state(state.bucket_sample_index)
            self._batch_sample_index.restore_state(state.batch_sample_index)
            for worker_idx, (buckets_state, buckets) in enumerate(
                zip(state.buckets, self._buckets)
            ):
                buckets.clear()
                if buckets_state is None:
                    continue
                else:
                    for bucket_state in buckets_state:
                        buckets[bucket_state.key] = Bucket(
                            batch_size=bucket_state.batch_size,
                            samples=SavableSampleBuffer(
                                self.dataset, worker_config=self.worker_config
                            ),
                        )
                        buckets[bucket_state.key].samples.restore_state(
                            SampleBufferMergedState(
                                buffer=[
                                    (
                                        []
                                        if gen_worker_idx == worker_idx
                                        else bucket_state.samples.buffer
                                    )
                                    for gen_worker_idx in range(
                                        max(self.worker_config.num_workers, 1)
                                    )
                                ]
                            )
                        )

    def can_restore_sample(self) -> bool:
        return self.batcher_stateless and self.dataset.can_restore_sample()

    def assert_can_restore(self) -> None:
        assert (
            self.batcher_stateless
        ), f"Batcher {self.batcher} must be stateless to restore samples"
        self.dataset.assert_can_restore()

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_batch:
        self.assert_can_restore()
        id, sample_idx, *sample_restore_keys = index
        assert id == type(self).__name__
        batch = [self.dataset.restore_sample(inner_idx) for inner_idx in sample_restore_keys]
        with self._batch_sample_index.ctx(sample_idx):
            batch_sample = self.batcher(batch)
        set_sample_restore_key(batch_sample, sample_idx, *sample_restore_keys, src=self)
        return batch_sample

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "bucket": self._function_config(self.sample_group_key),
            "batcher": self._function_config(self.batcher),
            **(
                {
                    "batcher_config": (
                        self.batcher_config()
                        if callable(self.batcher_config)
                        else self.batcher_config
                    )
                }
                if self.batcher_config
                else {}
            ),
            "batcher_stateless": self.batcher_stateless,
            "drop_last": self.drop_last,
            "error_handler": self._function_config(self.error_handler),
            "worker_config": self.worker_config.config(),
            "dataset": self.dataset.config(),
        }

    def __str__(self):
        return f"GroupBatchDataset(bucket={self.sample_group_key}, batcher={self.batcher}, drop_last={self.drop_last}, dataset={self.dataset})"
