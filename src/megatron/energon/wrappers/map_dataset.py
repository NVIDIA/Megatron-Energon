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
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from megatron.energon.errors import SYSTEM_EXCEPTIONS, FatalSampleError
from megatron.energon.flavors.base_dataset import SavableDataset, add_sample_restore_key
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers._log_exception import log_exception
from megatron.energon.wrappers.base import BaseWrapperDataset, SampleIndex, get_sample_restore_key
from megatron.energon.wrappers.skip import SkipSample

T_sample = TypeVar("T_sample")
T_sample_out = TypeVar("T_sample_out")


class MapDataset(BaseWrapperDataset[T_sample, T_sample_out], Generic[T_sample, T_sample_out]):
    """This dataset wrapper applies a custom function to transform each sample."""

    map_fn: Callable[[T_sample], Union[T_sample_out, Generator[T_sample_out, None, None]]]
    error_handler: Callable[[Exception, T_sample], None]
    stateless_map_fn: bool
    map_fn_config: Optional[Union[Dict[str, Any], Callable[[], Dict[str, Any]]]]
    _sample_index: SampleIndex
    _generator_sample_key: Optional[Any]
    _generator_offset: Optional[int]

    _savable_fields = (
        "_sample_index",
        "_generator_sample_key",
        "_generator_offset",
    )

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        map_fn: Callable[[T_sample], Union[T_sample_out, Generator[T_sample_out, None, None]]],
        *,
        error_handler: Callable[[Exception, T_sample], None] = log_exception,
        stateless_map_fn: bool = False,
        map_fn_config: Optional[Union[Dict[str, Any], Callable[[], Dict[str, Any]]]] = None,
        worker_config: WorkerConfig,
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
            map_fn_config: Configuration for the map_fn function. If callable, it should return the
                configuration. Defaults to None.
            worker_config: Worker configuration.
        """
        super().__init__(dataset, worker_config=worker_config)
        self.map_fn = map_fn
        self.error_handler = error_handler
        self.stateless_map_fn = stateless_map_fn
        self.map_fn_config = map_fn_config

        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._sample_index = SampleIndex(self.worker_config, src=self)
        self._generator_sample_key = None
        self._generator_offset = None

    def __len__(self):
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_sample_out]:
        if self._generator_sample_key is not None:
            assert self._generator_offset is not None
            sample = self.dataset.restore_sample(self._generator_sample_key)
            # Do not increment the sample index, use previous index
            with self._sample_index.ctx(self._sample_index.current_idx) as sample_idx:
                mapped_sample = self.map_fn(sample)
            assert isinstance(mapped_sample, Generator)
            assert inspect.isgeneratorfunction(self.map_fn), (
                f"Generator in {self.map_fn} but not marked as such."
            )
            target_offset = self._generator_offset
            self._generator_offset = 0
            for idx, (sample_idx, inner_sample) in enumerate(
                self._sample_index.iter_ctx(mapped_sample, sample_idx)
            ):
                # Skip other samples
                if idx >= target_offset:
                    self._generator_offset = idx + 1
                    yield add_sample_restore_key(
                        inner_sample,
                        sample_idx,
                        idx,
                        src=self,
                    )
            self._generator_sample_key = None
            self._generator_offset = None

        for sample in self.dataset:
            try:
                with self._sample_index.ctx() as sample_idx:
                    mapped_sample = self.map_fn(sample)
                if isinstance(mapped_sample, Generator):
                    assert inspect.isgeneratorfunction(self.map_fn), (
                        f"Generator in {self.map_fn} but not marked as such."
                    )
                    self._generator_sample_key = get_sample_restore_key(sample)
                    self._generator_offset = 0
                    # In case of a generator, additionally store the index of the yielded samples
                    # per input sample
                    for idx, (sample_idx, inner_sample) in enumerate(
                        self._sample_index.iter_ctx(mapped_sample, sample_idx)
                    ):
                        self._generator_offset = idx + 1
                        yield add_sample_restore_key(
                            inner_sample,
                            sample_idx,
                            idx,
                            src=self,
                        )
                    self._generator_sample_key = None
                    self._generator_offset = None
                else:
                    yield add_sample_restore_key(
                        mapped_sample,
                        sample_idx,
                        src=self,
                    )
            except SkipSample:
                pass
            except SYSTEM_EXCEPTIONS:
                raise FatalSampleError.from_sample(sample)
            except Exception as e:
                self.error_handler(e, sample)

    def can_restore_sample(self) -> bool:
        return super().can_restore_sample() and self.stateless_map_fn

    def assert_can_restore(self) -> None:
        assert self.stateless_map_fn, (
            f"MapDataset can only restore samples if map_fn {self.map_fn} is stateless."
        )
        super().assert_can_restore()

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample_out:
        self.assert_can_restore()
        if inspect.isgeneratorfunction(self.map_fn):
            id, sample_idx, local_idx = index[:3]
            assert id == type(self).__name__
            index = index[3:]
            assert isinstance(local_idx, int)
        else:
            id, sample_idx = index[:2]
            assert id == type(self).__name__
            index = index[2:]
        inner_sample = self.dataset.restore_sample(index)
        with self._sample_index.ctx(sample_idx):
            mapped_sample = self.map_fn(inner_sample)
        if isinstance(mapped_sample, Generator):
            assert inspect.isgeneratorfunction(self.map_fn), (
                f"Generator in {self.map_fn} but not marked as such."
            )
            for idx, (sample_idx, res_sample) in enumerate(
                self._sample_index.iter_ctx(mapped_sample, sample_idx)
            ):
                if idx == local_idx:
                    return add_sample_restore_key(res_sample, sample_idx, local_idx, src=self)
            assert False, (
                "Generator did not yield enough samples, but is marked stateless/deterministic."
            )
        else:
            return add_sample_restore_key(mapped_sample, sample_idx, src=self)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "map_fn": self._function_config(self.map_fn),
            **(
                {
                    "map_fn_config": (
                        self.map_fn_config() if callable(self.map_fn_config) else self.map_fn_config
                    )
                }
                if self.map_fn_config
                else {}
            ),
            "map_fn_stateless": self.stateless_map_fn,
        }

    def __str__(self):
        return f"MapDataset(map_fn={self.map_fn}, dataset={self.dataset})"
