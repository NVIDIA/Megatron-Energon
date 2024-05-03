# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

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
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig, global_worker_config
from megatron.energon.wrappers._log_exception import log_exception
from megatron.energon.wrappers.base import BaseSingleWrapperDataset
from megatron.energon.wrappers.skip import SkipSample

T_batch = TypeVar("T_batch", covariant=True)
T_batch_sample = TypeVar("T_batch_sample", covariant=True)


class BatchDataset(
    BaseSingleWrapperDataset[T_batch_sample, T_batch], Generic[T_batch_sample, T_batch]
):
    """This dataset wrapper transforms a dataset of samples into a dataset of batches."""

    batch_size: int
    batcher: Callable[[List[T_batch_sample]], T_batch]
    drop_last: bool
    error_handler: Callable[[Exception, List[T_batch_sample]], None]
    worker_config: WorkerConfig

    def __init__(
        self,
        dataset: SavableDataset[T_batch_sample],
        batch_size: int,
        batcher: Callable[[List[T_batch_sample]], T_batch],
        drop_last: bool = False,
        error_handler: Callable[[Exception, List[T_batch_sample]], None] = log_exception,
        worker_config: Optional[WorkerConfig] = None,
    ):
        """Construct a BatchDataset.

        Args:
            dataset: The input dataset to wrap
            batch_size: The desired batch size. The last batch may be smaller.
            batcher: Function which combines separate samples into a single object. May raise
                :exc:`megatron.energon.SkipSample` to skip a sample.
            drop_last: If True, the last batch is dropped if it is smaller than the batch size.
            error_handler: Function which handles exceptions raised by the batcher. The default
                implementation logs the exception.
            worker_config: Configuration for the workers. Defaults to `global_worker_config`.
        """
        super().__init__(dataset)
        self.batch_size = batch_size
        self.batcher = batcher
        self.drop_last = drop_last
        self.error_handler = error_handler
        self.worker_config = worker_config or global_worker_config

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
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.batch_size:
                try:
                    batch_sample = self.batcher(batch)
                    if isinstance(batch_sample, Generator):
                        yield from batch_sample
                    else:
                        yield batch_sample
                except SkipSample:
                    pass
                except SYSTEM_EXCEPTIONS:
                    raise FatalSampleError.from_sample(batch)
                except Exception as e:
                    self.error_handler(e, batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            try:
                yield self.batcher(batch)
            except SkipSample:
                pass
            except SYSTEM_EXCEPTIONS:
                raise FatalSampleError.from_sample(batch)
            except Exception as e:
                self.error_handler(e, batch)

    def can_restore_sample(self) -> bool:
        return False

    def restore_sample(self, index: Tuple[Union[str, int], ...]) -> T_batch:
        # TODO: We'd need to store multiple indices to restore a batch
        # Also, returned elements don't support __restore_key__. Would need extension.
        raise NotImplementedError("BatchDataset does not support random access.")

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "batch_size": self.batch_size,
            "batcher": self._function_config(self.batcher),
            "drop_last": self.drop_last,
            "error_handler": self._function_config(self.error_handler),
            "worker_config": self.worker_config.config(),
            "dataset": self.dataset.config(),
        }

    def __str__(self):
        return f"BatchDataset(batch_size={self.batch_size}, drop_last={self.drop_last}, batcher={self.batcher}, dataset={self.dataset})"
