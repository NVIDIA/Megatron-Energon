# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, Generic, Iterator, Literal, TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.sample_utils import default_get_batch_keys
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class LogSampleDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """This dataset logs every yielded sample to the debug logs."""

    get_keys_fn: Callable[[T_sample], list[str] | None]
    mode: Literal["train", "val"]
    _step: int

    _savable_fields = ("_step",)

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        mode: Literal["train", "val"],
        worker_config: WorkerConfig,
        get_keys_fn: Callable[[T_sample], list[str] | None] = default_get_batch_keys,
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

    def len_worker(self, worker_idx: int | None = None) -> int:
        return self.dataset.len_worker(worker_idx)

    def _log(self, sample: T_sample) -> None:
        if self.worker_config.should_log(level=1):
            log_entry = {
                "t": "yield_batch",
                "r": self.worker_config.rank,
                "w": self.worker_config.global_worker_id(),
                "m": self.mode,
                "idx": self._step,
            }
            keys = self.get_keys_fn(sample)
            if keys is not None:
                log_entry["keys"] = keys

            self.worker_config.worker_log(log_entry)

    def __iter__(self) -> Iterator[T_sample]:
        for sample in self.dataset:
            self._log(sample)
            self._step += 1
            yield sample

    def config(self) -> Dict[str, Any]:
        # Transparent logger, it won't change the samples
        return self.dataset.config()

    def __str__(self):
        return f"LogSampleDataset(mode={self.mode}, get_keys_fn={self.get_keys_fn}, dataset={self.dataset})"
