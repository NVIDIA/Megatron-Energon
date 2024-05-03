# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from typing import Any, Callable, Dict, Generator, Generic, Iterator, Tuple, TypeVar, Union

from megatron.energon.errors import SYSTEM_EXCEPTIONS, FatalSampleError
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.wrappers._log_exception import log_exception
from megatron.energon.wrappers.base import BaseSingleWrapperDataset
from megatron.energon.wrappers.skip import SkipSample

T_sample = TypeVar("T_sample")
T_sample_out = TypeVar("T_sample_out")


class MapDataset(BaseSingleWrapperDataset[T_sample, T_sample_out], Generic[T_sample, T_sample_out]):
    """This dataset wrapper applies a custom function to transform each sample."""

    map_fn: Callable[[T_sample], Union[T_sample_out, Generator[T_sample_out, None, None]]]
    error_handler: Callable[[Exception, T_sample], None]
    stateless_map_fn: bool

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        map_fn: Callable[[T_sample], Union[T_sample_out, Generator[T_sample_out, None, None]]],
        error_handler: Callable[[Exception, T_sample], None] = log_exception,
        stateless_map_fn: bool = False,
    ):
        """Construct a MapDataset.

        If this should be savable, the map_fn must only return a sample, or a generator yielding
        0 or 1 sample per input sample. Otherwise this will be broken (see `IterMapDataset`).

        Args:
            dataset: The input dataset to wrap
            map_fn: The function to apply to each sample. May raise
                :exc:`megatron.energon.SkipSample` to skip a sample. Alternatively, may return a
                generator to yield multiple or no samples.
            error_handler: Handler for errors. Defaults to logging and ignoring the exception.
            stateless_map_fn: If true, the map_fn is deterministic and stateless
                (thus key for random access can propagate to inner dataset). Defaults to False.
        """
        super().__init__(dataset)
        self.map_fn = map_fn
        self.error_handler = error_handler
        self.stateless_map_fn = stateless_map_fn

    def __len__(self):
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_sample_out]:
        for sample in self.dataset:
            try:
                mapped_sample = self.map_fn(sample)
                if isinstance(mapped_sample, Generator):
                    # In case of a generator, additionally store the index of the yielded samples
                    # per input sample
                    if self.can_restore_sample():
                        # If this is supposed to be restorable, map_fn must be a generator function
                        assert inspect.isgeneratorfunction(self.map_fn)
                        for idx, res_sample in enumerate(mapped_sample):
                            yield self._add_sample_restore_key(res_sample, idx, fail_otherwise=True)
                    else:
                        yield from mapped_sample
                else:
                    yield mapped_sample
            except SkipSample:
                pass
            except SYSTEM_EXCEPTIONS as e:
                raise FatalSampleError.from_sample(sample)
            except Exception as e:
                self.error_handler(e, sample)

    def can_restore_sample(self) -> bool:
        return self.stateless_map_fn and self.dataset.can_restore_sample()

    def restore_sample(self, index: Tuple[Union[str, int], ...]) -> T_sample_out:
        if self.stateless_map_fn:
            if inspect.isgeneratorfunction(self.map_fn):
                local_index = index[0]
                assert isinstance(local_index, int)
                index = index[1:]
            mapped_sample = self.map_fn(self.dataset.restore_sample(index))
            if isinstance(mapped_sample, Generator):
                assert inspect.isgeneratorfunction(self.map_fn)
                for idx, res_sample in enumerate(mapped_sample):
                    if idx == local_index:
                        return self._add_sample_restore_key(res_sample, idx, fail_otherwise=True)
                raise RuntimeError(
                    "Generator did not yield enough samples, but is marked stateless/deterministic."
                )
            else:
                return mapped_sample
        else:
            # Raise default error
            return super().__getitem__(index)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "map_fn": self._function_config(self.map_fn),
        }

    def __str__(self):
        return f"MapDataset(map_fn={self.map_fn}, dataset={self.dataset})"
