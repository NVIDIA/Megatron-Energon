# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import threading
from typing import Generic, TypeVar, cast

import torch

from megatron.energon.dataloader.asynchronous import ThreadAsynchronous
from megatron.energon.dataloader.future import CallableFuture, Future
from megatron.energon.flavors.base_dataset import PinMemoryMixin

TSample = TypeVar("TSample")
T = TypeVar("T")


class PinMemory(Generic[TSample]):
    """Base class for pinning memory of samples.

    This class is used to pin memory of samples in the primary process.
    """

    def __init__(self, device: str | torch.device):
        self._device = device

    def _pin_memory(self, sample: TSample) -> TSample:
        return PinMemoryMixin.sample_pin_memory(sample, self._device)

    def __call__(self, sample: Future[TSample]) -> Future[TSample]:
        """Pin the memory of a sample. The default implementation runs in the main thread."""
        return CallableFuture.chain(sample, lambda fut: self._pin_memory(fut.get()))

    def shutdown(self, in_del: bool = False) -> None:
        """
        Shutdown any running threads.

        Args:
            in_del: Whether the shutdown is called from the garbage collector.
        """
        pass


class NoPinMemory(PinMemory[TSample]):
    """No-op implementation of :class:`PinMemory`.

    Does not pin the memory of samples.
    """

    def __init__(self):
        super().__init__(device="cpu")

    def __call__(self, sample: Future[TSample]) -> Future[TSample]:
        return sample


class PinMemoryThread(PinMemory[TSample], ThreadAsynchronous, Generic[TSample]):
    """Threaded implementation of :class:`PinMemory`.

    Pins the memory of samples in a separate thread in the background.

    Creates the thread on first use.
    """

    _SHUTDOWN = cast(Future[TSample], object())

    _thread: threading.Thread | None = None

    def __init__(
        self,
        device: str | torch.device,
    ):
        super().__init__(device)
        self._asynchronous_init(name="pin-memory")

    def _worker_run(self, *args, **kwargs) -> None:
        try:
            torch.multiprocessing._set_thread_name("pt_data_pin")
        except AttributeError:
            pass
        super()._worker_run(*args, **kwargs)

    def _wrk_pin_memory(self, sample: Future[TSample]) -> TSample:
        sample_data = sample.get()
        return self._pin_memory(sample_data)

    def __call__(self, sample: Future[TSample]) -> Future[TSample]:
        """
        Pin the memory of a sample.
        Submits the sample future to the thread to fetch it and pins the memory in the thread,
        then returns a future for fetching the pinned sample.
        """
        if not self.running():
            self.start()
        return self._worker_call(self._wrk_pin_memory, sample)
