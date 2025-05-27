# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
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
from megatron.energon.source_info import SourceInfo
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers._log_exception import log_exception
from megatron.energon.wrappers.base import BaseWrapperDataset, SampleIndex, get_sample_restore_key
from megatron.energon.wrappers.skip import SkipSample

T_batch = TypeVar("T_batch", covariant=True)
T_batch_sample = TypeVar("T_batch_sample", covariant=True)


class BatchDataset(BaseWrapperDataset[T_batch_sample, T_batch], Generic[T_batch_sample, T_batch]):
    """This dataset wrapper transforms a dataset of samples into a dataset of batches."""

    batch_size: int
    batcher: Callable[[List[T_batch_sample]], T_batch]
    drop_last: bool
    error_handler: Callable[[Exception, list[T_batch_sample], list[SourceInfo]], None]
    _sample_index: SampleIndex
    _generator_sample_keys: Optional[Any]
    _generator_offset: Optional[int]

    _savable_fields = ("_sample_index", "_generator_sample_keys", "_generator_offset")

    def __init__(
        self,
        dataset: SavableDataset[T_batch_sample],
        batch_size: int,
        batcher: Callable[[List[T_batch_sample]], T_batch],
        *,
        batcher_stateless: bool = False,
        batcher_config: Optional[Union[Dict[str, Any], Callable[[], Dict[str, Any]]]] = None,
        drop_last: bool = False,
        error_handler: Callable[
            [Exception, List[T_batch_sample], List[SourceInfo]], None
        ] = log_exception,
        failure_tolerance: Optional[int] = 100,
        worker_config: WorkerConfig,
    ):
        """Construct a BatchDataset.

        Args:
            dataset: The input dataset to wrap
            batch_size: The desired batch size. The last batch may be smaller.
            batcher: Function which combines separate samples into a single object. May raise
                :exc:`megatron.energon.SkipSample` to skip a sample.
            batcher_stateless: If True, the batcher is stateless, thus samples can be stored/
                restored.
            batcher_config: Configuration for the batcher function. If callable, it should return the
                configuration. Defaults to None.
            drop_last: If True, the last batch is dropped if it is smaller than the batch size.
            error_handler: Function which handles exceptions raised by the batcher. The default
                implementation logs the exception.
            failure_tolerance: The number of consecutive failures after which the dataset is considered broken.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset, worker_config=worker_config)
        self.batch_size = batch_size
        self.batcher = batcher
        self.batcher_stateless = batcher_stateless
        self.batcher_config = batcher_config
        self.drop_last = drop_last
        self.error_handler = error_handler
        self.failure_tolerance = failure_tolerance

        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._sample_index = SampleIndex(self.worker_config, src=self)
        self._generator_sample_keys = None
        self._generator_offset = None

    def __len__(self):
        n_samples = len(self.dataset)
        num_workers = max(self.worker_config.num_workers, 1)
        n_samples_per_worker_floor = n_samples // num_workers
        remaining_n_sample_workers = n_samples % num_workers
        n_batches_per_worker_floor = n_samples_per_worker_floor // self.batch_size
        if n_samples_per_worker_floor % self.batch_size != 0 and not self.drop_last:
            n_batches_per_worker_floor += 1
        # Correct number of batches for the workers which yield 1 more sample (to balance)
        n_batches_per_worker_ceil = (n_samples_per_worker_floor + 1) // self.batch_size
        if n_batches_per_worker_ceil % self.batch_size != 0 and not self.drop_last:
            n_batches_per_worker_ceil += 1

        return (
            n_batches_per_worker_floor * (num_workers - remaining_n_sample_workers)
            + n_batches_per_worker_ceil * remaining_n_sample_workers
        )

    def __iter__(self) -> Iterator[T_batch]:
        batch: List[T_batch_sample] = []
        sample_restore_keys = []

        last_batch_failures = 0

        if self._generator_sample_keys is not None:
            sample_restore_keys = self._generator_sample_keys
            assert self._generator_offset is not None
            batch = [self.dataset.restore_sample(inner_idx) for inner_idx in sample_restore_keys]
            with self._sample_index.ctx(self._sample_index.current_idx) as sample_idx:
                batch_sample = self.batcher(batch)
            assert isinstance(batch_sample, Generator)
            assert inspect.isgeneratorfunction(self.batcher), (
                f"Generator in {self.batcher} but not marked as such."
            )
            target_offset = self._generator_offset
            self._generator_offset = 0
            for batch_sub_idx, (sample_idx, inner_batch_sample) in enumerate(
                self._sample_index.iter_ctx(batch_sample, sample_idx)
            ):
                # Skip other samples
                if batch_sub_idx >= target_offset:
                    self._generator_offset = batch_sub_idx + 1
                    yield set_sample_restore_key(
                        inner_batch_sample,
                        sample_idx,
                        batch_sub_idx,
                        *sample_restore_keys,
                        src=self,
                    )
            self._generator_sample_keys = None
            self._generator_offset = None
            batch.clear()
            sample_restore_keys = []

        def flush():
            nonlocal last_batch_failures

            try:
                with self._sample_index.ctx() as sample_idx:
                    batch_sample = self.batcher(batch)
                if isinstance(batch_sample, Generator):
                    assert inspect.isgeneratorfunction(self.batcher), (
                        f"Generator in {self.batcher} but not marked as such."
                    )
                    self._generator_sample_keys = sample_restore_keys
                    self._generator_offset = 0
                    for batch_sub_idx, (sample_idx, inner_batch_sample) in enumerate(
                        self._sample_index.iter_ctx(batch_sample, sample_idx)
                    ):
                        last_batch_failures = 0
                        self._generator_offset = batch_sub_idx + 1
                        yield set_sample_restore_key(
                            inner_batch_sample,
                            sample_idx,
                            batch_sub_idx,
                            *sample_restore_keys,
                            src=self,
                        )
                    self._generator_sample_keys = None
                    self._generator_offset = None
                else:
                    last_batch_failures = 0
                    set_sample_restore_key(batch_sample, sample_idx, *sample_restore_keys, src=self)
                    yield batch_sample
            except GeneratorExit:
                raise
            except SkipSample:
                pass
            except SYSTEM_EXCEPTIONS:
                raise FatalSampleError.from_sample(batch)
            except Exception as e:
                self.error_handler(e, batch)
                last_batch_failures += 1
                if (
                    self.failure_tolerance is not None
                    and last_batch_failures >= self.failure_tolerance
                ):
                    raise FatalSampleError.from_sample(
                        batch,
                        f"BatchDataset {self.batcher} failed {last_batch_failures} times in a row. Likely your code or dataset are broken.",
                    )
            finally:
                sample_restore_keys.clear()

        for sample in self.dataset:
            batch.append(sample)
            sample_restore_keys.append(get_sample_restore_key(sample))
            if len(batch) == self.batch_size:
                yield from flush()
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield from flush()

    def can_restore_sample(self) -> bool:
        # Cannot really verify if the returned elements contain a __restore_key__.
        # If the user wants to use this, well...
        return super().can_restore_sample() and self.batcher_stateless

    def assert_can_restore(self) -> None:
        assert self.batcher_stateless, (
            f"Batcher {self.batcher} must be stateless to restore samples"
        )
        super().assert_can_restore()

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T_batch:
        # We need to store multiple indices to restore a batch.
        self.assert_can_restore()
        if inspect.isgeneratorfunction(self.batcher):
            id, sample_idx, batch_sub_idx, *samples_restore_keys = restore_key
            assert id == type(self).__name__
        else:
            id, sample_idx, *samples_restore_keys = restore_key
            assert id == type(self).__name__
        batch = [self.dataset.restore_sample(inner_idx) for inner_idx in samples_restore_keys]
        with self._sample_index.ctx(sample_idx):
            batch_sample = self.batcher(batch)
        if isinstance(batch_sample, Generator):
            assert inspect.isgeneratorfunction(self.batcher), (
                f"Generator in {self.batcher} but not marked as such."
            )
            for cur_batch_sub_idx, (sample_idx, inner_batch_sample) in enumerate(
                self._sample_index.iter_ctx(batch_sample, sample_idx)
            ):
                if cur_batch_sub_idx == batch_sub_idx:
                    return set_sample_restore_key(
                        inner_batch_sample,
                        sample_idx,
                        batch_sub_idx,
                        *samples_restore_keys,
                        src=self,
                    )
            assert False, f"Batch sub-index {batch_sub_idx} not found in batch"
        else:
            return set_sample_restore_key(
                batch_sample,
                sample_idx,
                *samples_restore_keys,
                src=self,
            )

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "batch_size": self.batch_size,
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
        return f"BatchDataset(batch_size={self.batch_size}, drop_last={self.drop_last}, batcher={self.batcher}, dataset={self.dataset})"
