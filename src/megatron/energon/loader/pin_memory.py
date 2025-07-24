# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import queue
import threading
from typing import Generic, Protocol, TypeVar, cast, override

import torch

from megatron.energon.flavors.base_dataset import PinMemoryMixin
from megatron.energon.loader.future import CallableFuture, Future

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

    def shutdown(self) -> None:
        """Shutdown any running threads."""
        pass


class NoPinMemory(PinMemory[TSample]):
    """No-op implementation of :class:`PinMemory`.

    Does not pin the memory of samples.
    """

    def __init__(self):
        super().__init__(device="cpu")

    def __call__(self, sample: Future[TSample]) -> Future[TSample]:
        return sample


class QueueProtocol(Protocol[T]):
    def get(self, /) -> T: ...

    def put(self, item: T, /) -> None: ...

    def qsize(self, /) -> int: ...

    def task_done(self, /) -> None: ...

    def join(self, /) -> None: ...


class PinMemoryThread(PinMemory[TSample], Generic[TSample]):
    """Threaded implementation of :class:`PinMemory`.

    Pins the memory of samples in a separate thread in the background.

    Creates the thread on first use and shuts it down on shutdown. May be reused after shutdown.
    """

    _SHUTDOWN = cast(Future[TSample], object())

    _thread: threading.Thread | None = None

    _item_queue: QueueProtocol[Future[TSample]]
    _result_queue: QueueProtocol[tuple[TSample, None] | tuple[None, Exception]]

    def __init__(
        self,
        device: str | torch.device,
    ):
        super().__init__(device)
        self._item_queue = queue.Queue()
        self._result_queue = queue.Queue()

    def _run(self) -> None:
        """The pin memory thread. It will fetch the sample from the item future queue and pin the memory."""
        while True:
            try:
                sample = self._item_queue.get()
                if sample is self._SHUTDOWN:
                    break
                sample = self._pin_memory(sample.get())
            except Exception as e:
                self._result_queue.put((None, e))
            else:
                self._result_queue.put((sample, None))
            self._item_queue.task_done()

    def _get_next_result(self) -> TSample:
        result, exception = self._result_queue.get()
        if exception is not None:
            raise exception
        return cast(TSample, result)

    def __call__(self, sample: Future[TSample]) -> Future[TSample]:
        """
        Pin the memory of a sample.
        Submits the sample future to the thread to fetch it and pins the memory in the thread,
        then returns a future for fetching the pinned sample.
        """
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="PinMemoryThread")
            self._thread.start()
        self._item_queue.put(sample)
        return CallableFuture(self._get_next_result)

    @override
    def shutdown(self) -> None:
        if self._thread is not None:
            self._item_queue.put(self._SHUTDOWN)
            self._item_queue.join()
            self._thread.join()
            self._thread = None
