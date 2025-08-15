# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.savable import Savable
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import get_sample_restore_key

T_sample = TypeVar("T_sample")


@edataclass
class SavableSampleBufferState:
    """State of a SavableSampleBuffer."""

    restore_keys: List[Tuple[Union[str, int, tuple], ...]]


class SavableSampleBuffer(Savable, Generic[T_sample]):
    """A buffer of samples, savable. State is shared, create a state-local instance."""

    _buffer: List[T_sample]
    _restore_keys: List[Tuple[Union[str, int, tuple], ...]]

    _restore_pending: bool = False

    def __init__(self, dataset: SavableDataset[T_sample], *, worker_config: WorkerConfig):
        self.dataset = dataset
        self.worker_config = worker_config
        self._buffer = []
        self._restore_keys = []

    def worker_start(self) -> None:
        if self._restore_pending:
            assert len(self._buffer) == 0
            self._restore_pending = False
            for restore_key in self._restore_keys:
                self._buffer.append(self.dataset.restore_sample(restore_key))
        assert len(self._buffer) == len(self._restore_keys)

    def append(self, sample: T_sample) -> T_sample:
        self._buffer.append(sample)
        self._restore_keys.append(get_sample_restore_key(sample))
        return sample

    def extend(self, samples: List[T_sample], restore_keys: Optional[Sequence[Any]] = None) -> None:
        self._buffer.extend(samples)
        if restore_keys is None:
            self._restore_keys.extend(get_sample_restore_key(sample) for sample in samples)
        else:
            self._restore_keys.extend(restore_keys)

    def append_iter(self) -> Generator[T_sample, None, None]:
        for sample in self.dataset:
            yield self.append(sample)

    def pop(self, index: int) -> T_sample:
        self._restore_keys.pop(index)
        return self._buffer.pop(index)

    def flush(self) -> Tuple[List[T_sample], Tuple[Any, ...]]:
        buffer = list(self._buffer)
        restore_key = tuple(self._restore_keys)
        self._buffer.clear()
        self._restore_keys.clear()
        return buffer, restore_key

    @property
    def buffer(self) -> List[T_sample]:
        return self._buffer

    def __iter__(self) -> Iterator[T_sample]:
        return iter(self._buffer)

    def __getitem__(self, index: Union[int, slice]) -> Union[T_sample, List[T_sample]]:
        return self._buffer[index]

    def __setitem__(self, index: Union[int, slice], value: T_sample) -> None:
        self._buffer[index] = value
        if isinstance(index, slice):
            self._restore_keys[index] = (get_sample_restore_key(v) for v in value)
        else:
            self._restore_keys[index] = get_sample_restore_key(value)

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self._buffer[index]
        del self._restore_keys[index]

    def len_worker(self, worker_idx: int | None = None) -> int:
        self.worker_config.assert_worker()
        assert worker_idx is None or worker_idx == self.worker_config.rank_worker_id(), (
            "SavableSampleBuffer.len_worker only available for the current worker"
        )
        return len(self._restore_keys)

    def len_rank(self) -> int:
        raise NotImplementedError("len_rank is not available for SavableSampleBuffer")

    def save_state(self) -> SavableSampleBufferState:
        # Don't call super().save_state() because we don't want to save the wrapped datasets
        # Just save the own state
        return SavableSampleBufferState(restore_keys=self._restore_keys.copy())

    def restore_state(self, state: SavableSampleBufferState) -> None:
        # Don't call super().restore_state() because we don't want to restore the wrapped datasets
        # Just restore the own state
        self._restore_keys = state.restore_keys.copy()
        self._restore_pending = True

    def restore_key(self) -> Tuple[Union[str, int], ...]:
        return tuple(self._restore_keys)

    def restore_samples(
        self, index: Tuple[Union[str, int, tuple], ...]
    ) -> Tuple[Tuple[Union[str, int, tuple], ...], List[T_sample]]:
        buffer = []
        restore_keys = []
        for sub_index in index:
            sample = self.dataset.restore_sample(sub_index)
            restore_keys.append(get_sample_restore_key(sample))
            buffer.append(sample)
        return tuple(restore_keys), buffer

    def clear(self) -> None:
        self._buffer.clear()
        self._restore_keys.clear()

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "worker_config": self.worker_config.config(),
        }

    def debug_print(self, indent: str = ""):
        print(
            f"{indent}SavableSampleBuffer(size={len(self._restore_keys)}, res_pend={self._restore_pending}):\n",
            end="",
        )
        for i, (sample, restore_key) in enumerate(zip(self._buffer, self._restore_keys)):
            print(f"{indent}Sample {i} [{restore_key!r}]: {sample.__key__}\n", end="")

    def __str__(self):
        return f"SavableSampleBuffer(size={len(self._buffer)})"
