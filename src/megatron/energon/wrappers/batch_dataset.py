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

from megatron.energon.errors import ErrorContext, handle_restore_errors
from megatron.energon.flavors.base_dataset import SavableDataset, set_sample_restore_key
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset, SampleIndex, get_sample_restore_key

T_batch = TypeVar("T_batch", covariant=True)
T_batch_sample = TypeVar("T_batch_sample", covariant=True)


class BatchDataset(BaseWrapperDataset[T_batch_sample, T_batch], Generic[T_batch_sample, T_batch]):
    """This dataset wrapper transforms a dataset of samples into a dataset of batches."""

    batch_size: int
    batcher: Callable[[List[T_batch_sample]], T_batch]
    drop_last: bool
    _sample_index: SampleIndex
    _generator_sample_keys: Optional[Any]
    _generator_offset: Optional[int]
    _batch_failure_handler: ErrorContext

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
        failure_tolerance: int = 100,
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
            failure_tolerance: The number of consecutive failures after which the dataset is considered broken. Set to 0 to disable.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset, worker_config=worker_config)
        self.batch_size = batch_size
        self.batcher = batcher
        self.batcher_stateless = batcher_stateless
        self.batcher_config = batcher_config
        self.drop_last = drop_last
        self.failure_tolerance = failure_tolerance
        self._batch_failure_handler = ErrorContext(
            name=f"BatchDataset.{self.batcher}",
            handler=worker_config.global_error_handler,
            tolerance=failure_tolerance,
        )

        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._sample_index = SampleIndex(self.worker_config, src=self)
        self._generator_sample_keys = None
        self._generator_offset = None

    def len_worker(self, worker_idx: int | None = None) -> int:
        n_samples = self.dataset.len_worker(worker_idx)
        n_batches = n_samples // self.batch_size
        if n_samples % self.batch_size != 0 and not self.drop_last:
            n_batches += 1
        return n_batches

    def __iter__(self) -> Iterator[T_batch]:
        batch: List[T_batch_sample] = []
        sample_restore_keys = []

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

        def flush() -> Generator[T_batch, None, None]:
            with self._batch_failure_handler.handle_errors(batch):
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
                        self._generator_offset = batch_sub_idx + 1
                        self._batch_failure_handler.reset()
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
                    self._batch_failure_handler.reset()
                    set_sample_restore_key(batch_sample, sample_idx, *sample_restore_keys, src=self)
                    yield batch_sample
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

        with handle_restore_errors(self.worker_config.restore_error_handler, batch):
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
            "worker_config": self.worker_config.config(),
            "dataset": self.dataset.config(),
        }

    def __str__(self):
        return f"BatchDataset(batch_size={self.batch_size}, drop_last={self.drop_last}, batcher={self.batcher}, dataset={self.dataset})"
