# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    Tuple,
    TypeVar,
    Union,
)

import megatron.energon
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class TaskEncoderStateDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """This dataset wrapper applies a custom function to transform each sample."""

    # Will save it's own state
    _task_encoder: "megatron.energon.TaskEncoder"
    _task_encoder_was_reset: bool = False

    _savable_fields = ("_task_encoder",)

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        task_encoder: "megatron.energon.TaskEncoder",
        *,
        worker_config: WorkerConfig,
    ):
        """Construct a wrapper for saving/restoring the state of the task encoder.
        The dataset is transparently delegated.

        Args:
            dataset: The input dataset to wrap
            task_encoder: The task encoder to wrap.
            worker_config: Worker configuration.
        """
        super().__init__(dataset, worker_config=worker_config)
        self._task_encoder = task_encoder

    def reset_state_own(self) -> None:
        self._task_encoder_was_reset = False

    def __iter__(self) -> Iterator[T_sample]:
        if not self._task_encoder_was_reset:
            self._task_encoder_was_reset = True
            self._task_encoder.reset_state()
        yield from self.dataset

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        inner_sample = self.dataset.restore_sample(restore_key)
        inner_sample = self._task_encoder.restore_sample(inner_sample)
        return inner_sample

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
