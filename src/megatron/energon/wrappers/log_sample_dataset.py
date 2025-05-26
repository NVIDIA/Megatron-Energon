# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, Generic, Iterator, List, Literal, Optional, TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


def _flatten_str_list(keys: Any) -> Iterator[Optional[str]]:
    """Flatten a list of keys into a list of strings."""
    if isinstance(keys, str):
        yield keys
    elif isinstance(keys, (list, tuple)):
        for key in keys:
            yield from _flatten_str_list(key)
    else:
        yield None


def _flatten_str_list_or_none(keys: Any) -> Optional[List[str]]:
    """Flatten a list of keys into a list of strings. If this cannot be fetched, return None."""
    keys = list(_flatten_str_list(keys))
    if any(k is None for k in keys):
        return None
    return keys


def default_get_keys(batch: Any) -> Optional[List[str]]:
    """Default get_keys, which has some heuristics to find the sample keys."""
    if isinstance(batch, list):
        all_keys = []
        for b in batch:
            k = default_get_keys(b)
            if k is None:
                return None
            all_keys.extend(k)
        return all_keys
    if hasattr(batch, "__key__"):
        return _flatten_str_list_or_none(batch.__key__)
    elif hasattr(batch, "__keys__"):
        return _flatten_str_list_or_none(batch.__keys__)
    elif isinstance(batch, dict):
        if "__key__" in batch:
            return _flatten_str_list_or_none(batch["__key__"])
        elif "__keys__" in batch:
            return _flatten_str_list_or_none(batch["__keys__"])
        elif "keys" in batch:
            return _flatten_str_list_or_none(batch["keys"])
    return None


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

    def _log(self, sample: T_sample) -> dict:
        log_entry = {
            "idx": self._step,
        }
        keys = self.get_keys_fn(sample)
        if keys is not None:
            log_entry["keys"] = keys

        return log_entry

    def __iter__(self) -> Iterator[T_sample]:
        trace_span = self.worker_config.worker_trace_span()
        with trace_span.span(
            "LogSampleDataset.__iter__",
            args={
                "mode": self.mode,
            },
            level=1,
        ):
            for sample in trace_span.iterable(
                self.dataset, name="LogSampleDataset.__iter__.next", level=1
            ):
                with trace_span.span(
                    "LogSampleDataset.__iter__.yield", args=self._log(sample), level=1
                ):
                    self._step += 1
                    yield sample

    def config(self) -> Dict[str, Any]:
        # Transparent logger, it won't change the samples
        return self.dataset.config()

    def __str__(self):
        return f"LogSampleDataset(mode={self.mode}, get_keys_fn={self.get_keys_fn}, dataset={self.dataset})"
