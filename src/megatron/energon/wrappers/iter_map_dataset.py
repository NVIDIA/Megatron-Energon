# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

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

from torch.utils.data import IterableDataset

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

T_sample = TypeVar("T_sample")
T_sample_out = TypeVar("T_sample_out")


@dataclass
class IterMapState(BaseSingleWrapperState):
    sample_index: int


@dataclass
class IterMapMergedState(BaseSingleWrapperMergedState):
    sample_indexes: List[int]


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
    _sample_index: List[int]

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        iter_map_fn: Callable[[Iterator[T_sample]], Iterator[T_sample_out]],
        *,
        len_map_fn: Callable[[int], int] = lambda x: x,
        error_handler: Callable[[Exception, Optional[T_sample]], None] = log_exception,
        stateless_iter_fn: bool = False,
        worker_config: WorkerConfig,
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
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset)
        self.iter_map_fn = iter_map_fn
        self.len_map_fn = len_map_fn
        self.error_handler = error_handler
        self.stateless_iter_fn = stateless_iter_fn
        self.worker_config = worker_config
        self._sample_index = [0] * max(self.worker_config.num_workers, 1)

    def __len__(self):
        return self.len_map_fn(len(self.dataset))

    def __iter__(self) -> Iterator[T_sample_out]:
        worker_index = self.worker_config.rank_worker_id()
        last_sample_wrapper = _LastSampleWrapper(self.dataset)
        dataset_wrapper = wrap_worker_sample_index(
            last_sample_wrapper,
            self._sample_index,
            worker_index,
        )
        # The iter_map_fn is stateless. Thus we need to know which inner sample created the
        # outer sample, and the relative outer sample index, so we can restore it.

        # This is the sample index within the currently yielded sample
        iter_idx = 0
        batch_idx = 0

        def reset_idx_iter() -> Generator[T_sample, None, None]:
            # Resets the inner sample index
            nonlocal iter_idx, batch_idx
            for batch_idx, entry in dataset_wrapper:
                iter_idx = 0
                yield entry

        ds_iter = iter(reset_idx_iter())
        # While True will break when the inner dataset is exhausted, but continue on exception
        while True:
            iter_idx = 0
            try:
                for res_sample in self.iter_map_fn(ds_iter):
                    yield add_sample_restore_key(
                        res_sample,
                        iter_idx,
                        batch_idx,
                        src=self,
                    )
                    iter_idx += 1
            except SYSTEM_EXCEPTIONS:
                raise FatalSampleError.from_sample(last_sample_wrapper.last_sample)
            except Exception as e:
                self.error_handler(e, last_sample_wrapper.last_sample)
            else:
                break

    def save_state(self) -> IterMapState:
        return IterMapState.extend(
            super().save_state(),
            sample_index=self._sample_index[self.worker_config.rank_worker_id()],
        )

    def merge_states(self, states: List[IterMapState]) -> IterMapMergedState:
        assert all(s is None or isinstance(s, IterMapState) for s in states)
        return IterMapMergedState.extend(
            super().merge_states(states),
            sample_indexes=[0 if state is None else state.sample_index for state in states],
        )

    def restore_state(self, state: Optional[IterMapMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._sample_index = [0] * max(self.worker_config.num_workers, 1)
        else:
            assert isinstance(state, IterMapMergedState)
            self._sample_index = state.sample_indexes

    def can_restore_sample(self) -> bool:
        return self.stateless_iter_fn and self.dataset.can_restore_sample()

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        assert self.stateless_iter_fn
        id, iter_idx, sample_idx = index[:3]
        assert id == type(self).__name__
        index = index[3:]
        assert isinstance(iter_idx, int)
        with wrap_worker_sample_index_ctx(sample_idx):
            inner_sample = self.dataset.restore_sample(index)
        # Skip inner yielded samples to get the correct sample
        for idx, sample in enumerate(self.iter_map_fn(iter([inner_sample]))):
            if idx == iter_idx:
                return add_sample_restore_key(
                    sample, idx, sample_idx, fail_otherwise=True, src=self
                )
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
