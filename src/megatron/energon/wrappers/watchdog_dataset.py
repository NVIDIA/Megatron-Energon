# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import Any, Dict, Generic, Iterator, Optional, TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.watchdog import Watchdog
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class WatchdogDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """This dataset wraps another dataset and watches the time it takes to yield samples."""

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        worker_config: WorkerConfig,
        timeout_seconds: Optional[float] = 60,
        initial_timeout_seconds: Optional[float] = None,
        fail_on_timeout: bool = False,
    ):
        """Construct the watchdog dataset, which wraps another dataset and watches
        the time it takes to yield samples from the wrapped dataset.

        Args:
            dataset: The input dataset to wrap
            worker_config: The worker configuration
            timeout_seconds: The timeout in seconds. If None, the watchdog is disabled.
            initial_timeout_seconds: The initial timeout in seconds. If None, the timeout is the same as timeout_seconds.
            fail_on_timeout: If True, stops the whole process upon timeout, after printing a stack trace.
        """
        super().__init__(dataset, worker_config=worker_config)
        self.timeout_seconds = timeout_seconds
        self.initial_timeout_seconds = initial_timeout_seconds
        self.fail_on_timeout = fail_on_timeout

    def reset_state_own(self) -> None:
        pass

    def len_worker(self, worker_idx: int | None = None) -> int:
        return self.dataset.len_worker(worker_idx)

    def _watchdog_trigger(self) -> None:
        if self.fail_on_timeout:
            # Raising an exception here will kill the whole process
            raise TimeoutError(
                f"Watchdog triggered. Sample processing took longer than {self.timeout_seconds} seconds."
            )
        else:
            warnings.warn(
                f"Watchdog triggered. Sample processing took longer than {self.timeout_seconds} seconds.",
                RuntimeWarning,
            )

    def __iter__(self) -> Iterator[T_sample]:
        if self.timeout_seconds is None:
            yield from self.dataset
        else:
            watchdog = Watchdog(
                timeout=self.timeout_seconds,
                initial_timeout=self.initial_timeout_seconds,
                callback=self._watchdog_trigger,
                enabled=False,
            )
            yield from watchdog.watch_iter(self.dataset)

    def config(self) -> Dict[str, Any]:
        # Watchdog is transparent, it won't change the samples
        return self.dataset.config()

    def __str__(self):
        return f"WatchdogDataset(dataset={self.dataset})"
