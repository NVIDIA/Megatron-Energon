# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from torch.utils.data import IterableDataset

from megatron.energon.errors import SYSTEM_EXCEPTIONS, FatalSampleError
from megatron.energon.flavors.base_dataset import SavableDataset, set_sample_restore_key
from megatron.energon.source_info import SourceInfo
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers._log_exception import log_exception
from megatron.energon.wrappers.base import BaseWrapperDataset, SampleIndex, get_sample_restore_key

T_sample = TypeVar("T_sample")
T_sample_out = TypeVar("T_sample_out")


class IterMapDataset(BaseWrapperDataset[T_sample, T_sample_out], Generic[T_sample, T_sample_out]):
    """This dataset wrapper applies a custom function to transform the stream of samples and yield
    a new stream of samples.
    If used in a savable dataset context, it is critical, that `iter_map_fn` is either stateless,
    or that the state of the `iter_map_fn` is saved and restored externally.
    """

    iter_map_fn: Callable[[Iterator[T_sample]], Iterator[T_sample_out]]
    len_map_fn: Callable[[int], int]
    error_handler: Callable[[Exception, Optional[T_sample], list[SourceInfo]], None]
    stateless_iter_fn: bool
    iter_map_fn_config: Optional[Union[Dict[str, Any], Callable[[], Dict[str, Any]]]]
    _sample_index: SampleIndex

    _savable_fields = ("_sample_index",)

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        iter_map_fn: Callable[[Iterator[T_sample]], Iterator[T_sample_out]],
        *,
        len_map_fn: Callable[[int], int] = lambda x: x,
        error_handler: Callable[
            [Exception, Optional[T_sample], list[SourceInfo]], None
        ] = log_exception,
        stateless_iter_fn: bool = False,
        iter_map_fn_config: Optional[Union[Dict[str, Any], Callable[[], Dict[str, Any]]]] = None,
        worker_config: WorkerConfig,
    ):
        """Construct a IterMapDataset.
        For saving and restoring samples, the iter_map_fn must only yield 0 or 1 sample per
        iterated sample.

        Args:
            dataset: The input dataset to wrap
            iter_map_fn: The function to apply to the stream of samples. Returns a new stream of
                samples. If savability should be preserved, this function should be stateless.
            len_map_fn: The function to apply to the length of the dataset. Returns the new
                (approximate) length of the resulting stream of samples based on the original
                length.
            error_handler: Handler for errors. Defaults to logging and ignoring the exception.
            stateless_iter_fn: If true, assume the iter_map_fn is deterministic and stateless
                (it does not aggregate samples (thus key for random access can propagate to inner
                dataset), yielding zero or multiple samples per fetched sample is fine).
                Defaults to False.
            iter_map_fn_config: Configuration for the iter_map_fn function. If callable, it should return the
                configuration. Defaults to None.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset, worker_config=worker_config)
        self.iter_map_fn = iter_map_fn
        self.len_map_fn = len_map_fn
        self.error_handler = error_handler
        self.stateless_iter_fn = stateless_iter_fn
        self.iter_map_fn_config = iter_map_fn_config

        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._sample_index = SampleIndex(self.worker_config, src=self)

    def len_worker(self, worker_idx: int | None = None) -> int:
        return self.len_map_fn(self.dataset.len_worker(worker_idx))

    def __iter__(self) -> Iterator[T_sample_out]:
        last_sample_wrapper = _LastSampleWrapper(self.dataset)
        # The iter_map_fn is stateless. Thus we need to know which inner sample created the
        # outer sample, and the relative outer sample index, so we can restore it.

        # This is the sample index within the currently yielded sample
        iter_idx = 0
        sample_idx = 0
        sample_restore_keys = []

        def reset_idx_iter() -> Generator[T_sample, None, None]:
            # Resets the inner sample index
            nonlocal iter_idx, sample_restore_keys
            for entry in last_sample_wrapper:
                iter_idx = 0
                sample_restore_keys.append(get_sample_restore_key(entry))
                yield entry

        ds_iter = iter(reset_idx_iter())

        # While True will break when the inner dataset is exhausted, but may continue on exception
        while True:
            iter_idx = 0
            try:
                for sample_idx, sample in self._sample_index.iter_ctx(self.iter_map_fn(ds_iter)):
                    yield set_sample_restore_key(
                        sample,
                        sample_idx,
                        iter_idx,
                        *sample_restore_keys,
                        src=self,
                    )
                    sample_restore_keys.clear()
                    iter_idx += 1
            except SYSTEM_EXCEPTIONS:
                raise FatalSampleError.from_sample(last_sample_wrapper.last_sample)
            except Exception as e:
                self.error_handler(e, last_sample_wrapper.last_sample)
            else:
                break

    def can_restore_sample(self) -> bool:
        return super().can_restore_sample() and self.stateless_iter_fn

    def assert_can_restore(self) -> None:
        assert self.stateless_iter_fn, (
            "IterMapDataset can only restore samples if iter_map_fn is stateless."
        )
        super().assert_can_restore()

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        self.assert_can_restore()
        id, sample_idx, iter_idx, *sample_restore_keys = restore_key
        assert id == type(self).__name__
        assert isinstance(iter_idx, int)
        inner_iter = iter(
            self.iter_map_fn(
                (self.dataset.restore_sample(inner_index) for inner_index in sample_restore_keys)
            )
        )
        try:
            # Skip inner yielded samples to get the correct sample
            for skip_idx in range(iter_idx):
                with self._sample_index.ctx(sample_idx - iter_idx + skip_idx):
                    next(inner_iter)
            # This is the sample to restore
            with self._sample_index.ctx(sample_idx):
                sample = next(inner_iter)
            return set_sample_restore_key(
                sample,
                sample_idx,
                iter_idx,
                *sample_restore_keys,
                src=self,
            )
        except StopIteration:
            raise RuntimeError(
                "Generator did not yield enough samples, but is marked stateless/deterministic."
            )
        finally:
            # Properly close if it's a generator
            if hasattr(inner_iter, "close"):
                inner_iter.close()

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "iter_map_fn": self._function_config(self.iter_map_fn),
            **(
                {
                    "iter_map_fn_config": (
                        self.iter_map_fn_config()
                        if callable(self.iter_map_fn_config)
                        else self.iter_map_fn_config
                    )
                }
                if self.iter_map_fn_config
                else {}
            ),
            "len_map_fn": self._function_config(self.len_map_fn),
            "error_handler": self._function_config(self.error_handler),
        }

    def __str__(self):
        return f"IterMapDataset(iter_map_fn={self.iter_map_fn}, dataset={self.dataset})"


class _LastSampleWrapper:
    """
    Wraps the inner dataset and stores the last iterated sample.
    """

    last_sample: Optional[T_sample] = None
    dataset: IterableDataset[T_sample]

    def __init__(self, dataset: IterableDataset[T_sample]):
        self.dataset = dataset

    def __iter__(self) -> Iterator[T_sample]:
        for sample in self.dataset:
            self.last_sample = sample
            yield sample
