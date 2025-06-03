# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, Generic, Iterator, List, Literal, Optional, TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.flavors.trace import TraceIter, default_get_keys, trace_iter
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class LogSampleDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """This dataset logs every yielded sample to the debug logs."""

    get_keys_fn: Callable[[T_sample], Optional[List[str]]]
    mode: Literal["train", "val"]
    _step: int

    _savable_fields = ("_step",)

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        mode: Literal["train", "val"],
        worker_config: WorkerConfig,
        get_keys_fn: Callable[[T_sample], Optional[List[str]]] = default_get_keys,
    ):
        """Construct the log sample dataset, which logs every yielded sample to the debug logs.

        Args:
            dataset: The input dataset to wrap
        """
        super().__init__(dataset, worker_config=worker_config)
        self.get_keys_fn = get_keys_fn
        self.mode = mode

        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._step = 0

    def __len__(self):
        return len(self.dataset)

    @trace_iter(
        next_args={
            "idx": lambda self: self._step,
        },
    )
    def __iter__(self, trace_iter: TraceIter) -> Iterator[T_sample]:
        for sample in self.dataset:
            self._step += 1
            trace_iter.sample(sample)
            yield sample

    def config(self) -> Dict[str, Any]:
        # Transparent logger, it won't change the samples
        return self.dataset.config()

    def __str__(self):
        return f"LogSampleDataset(mode={self.mode}, get_keys_fn={self.get_keys_fn}, dataset={self.dataset})"
