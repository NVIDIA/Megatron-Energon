import contextlib
import heapq
import queue
import threading
import weakref
from collections import abc
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

import torch.utils.data
from torch.types import Number
from torch.utils.data import _utils, get_worker_info
from torch.utils.data.dataloader import (
    _BaseDataLoaderIter,
    _DatasetKind,
    _SingleProcessDataLoaderIter,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


@dataclass(frozen=True, order=True)
class _WorkerData:
    """Wrapper for data returned by workers to include a priority"""

    iter_id: int
    index: int
    data: Any = field(compare=False)
    is_sentinel: bool = field(default=False, compare=False)
    skip: bool = field(default=False, compare=False)

    @property
    def exception(self) -> BaseException | None:
        return self.data if self.is_sentinel else None

    def __post_init__(self):
        # Exception must appear first to be reported before the end of the iteration
        if isinstance(self.data, BaseException):
            object.__setattr__(self, "index", -1)
        # Set the indices to inf for sentinels to make sure they appear after regular batches of the same iteration
        elif self.is_sentinel:
            object.__setattr__(self, "index", float("inf"))


class _ThreadedDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader: "DataLoader"):
        super().__init__(loader)

        self._prefetch_factor = loader.prefetch_factor

        assert self._num_workers > 0
        assert self._prefetch_factor is not None and self._prefetch_factor >= 1

        self._worker_init_fn = loader.worker_init_fn
        self._workers_manager = _WorkersManager(
            self,
            self._next_index,
            self._base_seed,
            self._pin_memory,
            self._pin_memory_device,
        )
        self._workers_manager.submit_workers()

        self._finalizer = weakref.finalize(self, self._workers_manager.stop)

        # Workers might return batches out of order. In this case, it is stored locally.
        self._current_index = 0
        self._local_data = list[_WorkerData]()  # heap

    @override
    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)
        self._current_index = 0
        self._local_data.clear()
        self._workers_manager.reset()

    @override
    def _next_data(self):
        while True:
            # has the next batch already been received?
            if self._local_data and self._local_data[0].index == self._current_index:
                self._current_index += 1
                data = heapq.heappop(self._local_data)
                if data.skip:
                    continue
                return data.data

            try:
                # can raise exceptions propagated by a worker
                data = self._workers_manager.next_data()
            except queue.Empty:  # timeout exceeded
                raise RuntimeError(
                    f"DataLoader timed out after {self._timeout} seconds"
                )

            # Ignore any outdated data
            if self._workers_manager.is_outdated_data(data):
                continue

            if not data.is_sentinel:
                if data.index == self._current_index:
                    self._current_index += 1
                    if data.skip:
                        continue
                    return data.data

                assert data.index > self._current_index
                heapq.heappush(self._local_data, data)
                continue

            # One worker is done, continue with others
            if self._workers_manager.has_active_workers() > 0:
                continue

            # All workers are done, terminate
            if not self._persistent_workers:
                self._finalizer()

            raise StopIteration()

    @property
    def _workers(self):
        return self._workers_manager.workers


class _WorkersManager:
    """
    Delocating workers management from the iterator prevents cyclic references
    and allows the iterator to be destroyed properly.

    The workers manager only keeps a weak reference to the dataloader and iterator,
    which allows for garbage collection. Threads are automatically shutdown upon
    garbage collection of either the dataloader or the iterator.
    """

    def __init__(
        self,
        base: _ThreadedDataLoaderIter,
        index_fetcher: abc.Callable[[], list[int]],
        seed: Number,
        pin_memory: bool,
        pin_memory_device: str | None,
    ):
        self._base = weakref.proxy(base)
        if hasattr(index_fetcher, "__self__"):  # is it a bound method?
            self._index_fetcher = weakref.WeakMethod(index_fetcher)
        else:
            self._index_fetcher = weakref.ref(index_fetcher)

        self._base_seed = seed
        self._pin_memory = pin_memory
        self._pin_memory_device = pin_memory_device

        self._queue = queue.PriorityQueue[_WorkerData]()
        self._num_workers = base._num_workers
        self._running = False
        self._timeout = base._timeout if base._timeout > 0 else None

        self._num_active_workers = 0
        self._active_lock = threading.Lock()

        # The indices are provided by an iterator so we lock it. This lock is also used to increment the batch number
        self._index_lock = threading.Lock()
        self._batch_index = 0

        self._prefetch_count = base._num_workers * base._prefetch_factor  # type: ignore
        self._prefetch_sema = threading.BoundedSemaphore(self._prefetch_count)

        # Each iteration is assigned an id used to identify sentinels.
        # When reusing workers, this allows the caller to discard sentinels that are leftovers from previous iterations.
        self._iter_id = 0

        # When reusing threads, the semaphore allows them to restart
        if base._persistent_workers:
            # This barrier needs all workers + the main thread to be unlocked
            self._reuse_barrier = threading.Barrier(self._num_workers + 1)
        else:
            self._reuse_barrier = None

        self.workers = []

    def submit_workers(self):
        self._running = True

        if _utils.worker._worker_info is None:
            _utils.worker._worker_info = _ThreadWorkerInfo()

        self._num_active_workers = 0
        for worker_id in range(self._num_workers):
            if self._running:
                args = (worker_id, self._iter_id)
                thread = threading.Thread(target=self._worker, args=args, daemon=True)

                # Mokey-patch threads to allow tests to use them as processes
                thread.exitcode = None  # pyright: ignore [reportAttributeAccessIssue]
                thread.terminate = lambda: ...  # pyright: ignore

                thread.start()
                self.workers.append(thread)

    def has_active_workers(self) -> bool:
        with self._active_lock:
            return self._num_active_workers > 0

    def next_data(self):
        data = self._queue.get(timeout=self._timeout)

        if not data.is_sentinel and not self.is_outdated_data(data):
            self._prefetch_sema.release()
        elif exc := data.exception:
            # Exceptions can cause cyclic references, perform a copy to prevent that
            raise deepcopy(exc).with_traceback(exc.__traceback__)

        return data

    def reset(self):
        self._running = False  # make sure that no worker starts a new task
        self._iter_id += 1
        self._batch_index = 0
        self._clear_prefetch_sema()
        self._running = True
        if self._reuse_barrier is not None:
            self._reuse_barrier.wait()

    def stop(self):
        self._running = False
        self._iter_id += 1
        self._clear_prefetch_sema()
        if self._reuse_barrier is not None:
            self._reuse_barrier.abort()

    def is_outdated_data(self, data: _WorkerData) -> bool:
        return data.iter_id < self._iter_id

    def _clear_prefetch_sema(self):
        offset = self._prefetch_count - self._prefetch_sema._value
        if offset > 0:
            self._prefetch_sema.release(offset)

    def _build_dataset_fetcher(self):
        return _DatasetKind.create_fetcher(
            self._base._dataset_kind,
            self._base._dataset,
            self._base._auto_collation,
            self._base._collate_fn,
            self._base._drop_last,
        )

    def _worker(
        self,
        worker_id: int,
        iter_id: int,
        dataset_fetcher: _utils.fetch._BaseDatasetFetcher | None = None,
        new_iter: bool = True,
    ):
        if new_iter:
            with self._active_lock:
                self._num_active_workers += 1

        exception = None
        # Indicate if have acquired the prefetch semaphore and thus must release it on exception
        must_release = False
        try:
            get_worker_info().init(  # type: ignore
                id=worker_id,
                num_workers=self._num_workers,
                seed=self._base_seed + worker_id,
                dataset=self._base._dataset,
            )

            init_fn = self._base._worker_init_fn
            if init_fn is not None:
                init_fn(worker_id)

            if dataset_fetcher is None:
                dataset_fetcher = self._build_dataset_fetcher()

            while self._running:
                self._prefetch_sema.acquire()
                if iter_id < self._iter_id:
                    # `reset` or `stop` has been called -- cancel this iteration
                    raise StopIteration

                must_release = True

                # If any exception happens here, send dummy data with the right index
                # We cannot safely decrement the index here as other threads might have already used it
                try:
                    with self._index_lock:
                        old_batch_idx = self._batch_index
                        self._batch_index += 1
                        # It is important not to store the result of a call to
                        # index the fetcher in a variable as this coult create
                        # a strong reference to the bound method
                        index = self._index_fetcher()()  # type: ignore

                    data = dataset_fetcher.fetch(index)

                    if self._pin_memory:
                        data = _utils.pin_memory.pin_memory(
                            data, self._pin_memory_device
                        )

                    self._queue.put(_WorkerData(self._iter_id, old_batch_idx, data))
                except:
                    self._queue.put(_WorkerData(self._iter_id, old_batch_idx, None, skip=True))  # type: ignore
                    raise
                finally:
                    # The semaphore will be released by the consumer
                    must_release = False
        except (StopIteration, ReferenceError):
            ...
        except BaseException as exc:
            exception = exc
        finally:
            if must_release:
                self._prefetch_sema.release()

            if exception is None:
                with self._active_lock:
                    self._num_active_workers -= 1

            sentinel = _WorkerData(
                iter_id=iter_id,
                index=self._batch_index,
                data=exception,
                is_sentinel=True,
            )
            self._queue.put(sentinel)

            # Continue working even if we encountered an exception
            if exception is not None:
                return self._worker(worker_id, iter_id, dataset_fetcher, new_iter=False)

            try:
                if self._base._persistent_workers:
                    assert self._reuse_barrier is not None
                    self._reuse_barrier.wait()
                    if self._running:
                        self._worker(worker_id, iter_id + 1)
            except (ReferenceError, threading.BrokenBarrierError):
                ...

        self.workers[worker_id].exitcode = 0


class _ThreadWorkerInfo(_utils.worker.WorkerInfo, threading.local):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.__init_called = False
        # Do not call WorkerInfo.__init__ yet

    def init(self, **kwargs):
        if self.__init_called:
            return

        self.__init_called = True
        _utils.worker.WorkerInfo.__init__(self, **kwargs)


def patch_data_loader():
    def get_iterator(self) -> _BaseDataLoaderIter:
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _ThreadedDataLoaderIter(self)

    def get_worker_info():
        info = _utils.worker._worker_info
        if not isinstance(info, _ThreadWorkerInfo | None):
            return info

        if info is None or not info._WorkerInfo__initialized:  # pyright: ignore
            return None

        return info

    torch.utils.data.DataLoader._get_iterator = get_iterator
    torch.utils.data.get_worker_info = get_worker_info
    _ThreadWorkerInfo.__name__ = "WorkerInfo"


@contextlib.contextmanager
def patched_loader():
    old_getter = torch.utils.data.DataLoader._get_iterator
    try:
        patch_data_loader()
        yield
    finally:
        torch.utils.data.DataLoader._get_iterator = old_getter
