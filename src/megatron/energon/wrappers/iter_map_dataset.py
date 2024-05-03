# Copyright (c) 2024, NVIDIA CORPORATION.
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
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.wrappers._log_exception import log_exception
from megatron.energon.wrappers.base import BaseSingleWrapperDataset

T_sample = TypeVar("T_sample")
T_sample_out = TypeVar("T_sample_out")


class _ExcSampler:
    last_sample: Optional[T_sample] = None
    dataset: IterableDataset[T_sample]

    def __init__(self, dataset: IterableDataset[T_sample]):
        self.dataset = dataset

    def __iter__(self) -> Iterator[T_sample]:
        for sample in self.dataset:
            self.last_sample = sample
            yield sample


class IterMapDataset(
    BaseSingleWrapperDataset[T_sample, T_sample_out], Generic[T_sample, T_sample_out]
):
    """This dataset wrapper applies a custom function to transform the stream of samples and yield
    a new stream of samples.
    If used in a savable dataset context, it is critical, that `iter_map_fn` is either stateless,
    or that the state of the `iter_map_fn` is saved and restored externally.
    """

    iter_map_fn: Callable[[Iterator[T_sample]], Iterator[T_sample_out]]
    len_map_fn: Callable[[int], int]
    error_handler: Callable[[Exception, Optional[T_sample]], None]
    stateless_iter_fn: bool

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        iter_map_fn: Callable[[Iterator[T_sample]], Iterator[T_sample_out]],
        len_map_fn: Callable[[int], int] = lambda x: x,
        error_handler: Callable[[Exception, Optional[T_sample]], None] = log_exception,
        stateless_iter_fn: bool = False,
    ):
        """Construct a IterMapDataset.
        For saving and restoring samples, the iter_map_fn must only yield 0 or 1 sample per
        iterated sample.

        TODO: Implement saving/restoring for arbitrary number of yielded samples.
        Problem is, that this would require the inner dataset to be restored on the previous sample,
        such that this dataset can skip the already yielded samples.

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
        """
        super().__init__(dataset)
        self.iter_map_fn = iter_map_fn
        self.len_map_fn = len_map_fn
        self.error_handler = error_handler
        self.stateless_iter_fn = stateless_iter_fn

    def __len__(self):
        return self.len_map_fn(len(self.dataset))

    def __iter__(self) -> Iterator[T_sample_out]:
        exc_sampler = _ExcSampler(self.dataset)
        if self.can_restore_sample():
            # The iter_map_fn is stateless. Thus we need to know which inner sample created the
            # outer sample, and the relative outer sample index, so we can restore it.

            # This is the sample index within the currently yielded sample
            iter_idx = 0

            def reset_idx_iter() -> Generator[T_sample, None, None]:
                # Resets the inner sample index
                nonlocal iter_idx
                for entry in exc_sampler:
                    iter_idx = 0
                    yield entry

            ds_iter = iter(reset_idx_iter())
            # While True will break when the inner dataset is exhausted, but continue on exception
            while True:
                iter_idx = 0
                try:
                    for res_sample in self.iter_map_fn(ds_iter):
                        yield self._add_sample_restore_key(
                            res_sample, iter_idx, fail_otherwise=True
                        )
                        iter_idx += 1
                except SYSTEM_EXCEPTIONS:
                    raise FatalSampleError.from_sample(exc_sampler.last_sample)
                except Exception as e:
                    self.error_handler(e, exc_sampler.last_sample)
                else:
                    break
        else:
            ds_iter = iter(exc_sampler)
            # While True will break when the inner dataset is exhausted, but continue on exception
            while True:
                try:
                    yield from self.iter_map_fn(ds_iter)
                except SYSTEM_EXCEPTIONS:
                    raise FatalSampleError.from_sample(exc_sampler.last_sample)
                except Exception as e:
                    self.error_handler(e, exc_sampler.last_sample)
                else:
                    break

    def can_restore_sample(self) -> bool:
        return self.stateless_iter_fn and self.dataset.can_restore_sample()

    def restore_sample(self, index: Tuple[Union[str, int], ...]) -> T_sample:
        assert self.stateless_iter_fn
        iter_idx = index[0]
        assert isinstance(iter_idx, int)
        inner_sample = self.dataset.restore_sample(index[1:])
        # Skip inner yielded samples to get the correct sample
        for idx, sample in enumerate(self.iter_map_fn(iter([inner_sample]))):
            if idx == iter_idx:
                return self._add_sample_restore_key(sample, idx, fail_otherwise=True)
        raise RuntimeError(
            "Generator did not yield enough samples, but is marked stateless/deterministic."
        )

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "iter_map_fn": self._function_config(self.iter_map_fn),
            "len_map_fn": self._function_config(self.len_map_fn),
            "error_handler": self._function_config(self.error_handler),
        }

    def __str__(self):
        return f"IterMapDataset(iter_map_fn={self.iter_map_fn}, dataset={self.dataset})"
