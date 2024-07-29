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
    wrap_worker_sample_index,
    wrap_worker_sample_index_ctx,
)
from megatron.energon.wrappers.skip import SkipSample

T_sample = TypeVar("T_sample")
T_sample_out = TypeVar("T_sample_out")


@dataclass
class MapState(BaseSingleWrapperState):
    sample_index: int


@dataclass
class MapMergedState(BaseSingleWrapperMergedState):
    sample_indexes: List[int]


class MapDataset(BaseSingleWrapperDataset[T_sample, T_sample_out], Generic[T_sample, T_sample_out]):
    """This dataset wrapper applies a custom function to transform each sample."""

    map_fn: Callable[[T_sample], Union[T_sample_out, Generator[T_sample_out, None, None]]]
    error_handler: Callable[[Exception, T_sample], None]
    stateless_map_fn: bool
    _sample_index: List[int]

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        map_fn: Callable[[T_sample], Union[T_sample_out, Generator[T_sample_out, None, None]]],
        *,
        error_handler: Callable[[Exception, T_sample], None] = log_exception,
        stateless_map_fn: bool = False,
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
        """
        super().__init__(dataset)
        self.map_fn = map_fn
        self.error_handler = error_handler
        self.stateless_map_fn = stateless_map_fn
        self.worker_config = worker_config
        self._sample_index = [0] * max(self.worker_config.num_workers, 1)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_sample_out]:
        worker_index = self.worker_config.rank_worker_id()
        for sample_idx, sample in wrap_worker_sample_index(
            self.dataset, self._sample_index, worker_index
        ):
            try:
                mapped_sample = self.map_fn(sample)
                if isinstance(mapped_sample, Generator):
                    assert inspect.isgeneratorfunction(self.map_fn)
                    # In case of a generator, additionally store the index of the yielded samples
                    # per input sample
                    for idx, inner_sample in enumerate(mapped_sample):
                        yield add_sample_restore_key(
                            inner_sample,
                            sample_idx,
                            idx,
                            src=self,
                        )
                else:
                    yield add_sample_restore_key(
                        mapped_sample,
                        sample_idx,
                        src=self,
                    )
            except SkipSample:
                pass
            except SYSTEM_EXCEPTIONS as e:
                raise FatalSampleError.from_sample(sample)
            except Exception as e:
                self.error_handler(e, sample)

    def save_state(self) -> MapState:
        return MapState.extend(
            super().save_state(),
            sample_index=self._sample_index[self.worker_config.rank_worker_id()],
        )

    def merge_states(self, states: List[MapState]) -> MapMergedState:
        assert all(s is None or isinstance(s, MapState) for s in states)
        return MapMergedState.extend(
            super().merge_states(states),
            sample_indexes=[0 if state is None else state.sample_index for state in states],
        )

    def restore_state(self, state: Optional[MapMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._sample_index = [0] * max(self.worker_config.num_workers, 1)
        else:
            assert isinstance(state, MapMergedState)
            self._sample_index = state.sample_indexes

    def can_restore_sample(self) -> bool:
        return self.stateless_map_fn and self.dataset.can_restore_sample()

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample_out:
        assert self.stateless_map_fn, "MapDataset is not stateless, but restore_sample was called."
        if inspect.isgeneratorfunction(self.map_fn):
            id, sample_idx, local_idx = index[:3]
            assert id == type(self).__name__
            index = index[3:]
            assert isinstance(local_idx, int)
        else:
            id, sample_idx = index[:2]
            assert id == type(self).__name__
            index = index[2:]
        with wrap_worker_sample_index_ctx(sample_idx):
            inner_sample = self.dataset.restore_sample(index)
        mapped_sample = self.map_fn(inner_sample)
        if isinstance(mapped_sample, Generator):
            assert inspect.isgeneratorfunction(self.map_fn)
            for idx, res_sample in enumerate(mapped_sample):
                if idx == local_idx:
                    return add_sample_restore_key(res_sample, sample_idx, local_idx, src=self)
            assert (
                False
            ), "Generator did not yield enough samples, but is marked stateless/deterministic."
        else:
            return add_sample_restore_key(mapped_sample, sample_idx, src=self)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "map_fn": self._function_config(self.map_fn),
        }

    def __str__(self):
        return f"MapDataset(map_fn={self.map_fn}, dataset={self.dataset})"
