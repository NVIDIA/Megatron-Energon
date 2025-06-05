# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import os
import pickle
import random
import string
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, TypeVar

from megatron.energon.cache.base import CachePool, FileStore, Lazy
from megatron.energon.cache.file_store import DecodeFileStore
from megatron.energon.edataclass import edataclass
from megatron.energon.fork_hook import ForkMixin
from megatron.energon.source_info import SourceInfo, add_source_info

T = TypeVar("T")


@edataclass
class FileCacheLazy(Lazy[T]):
    """
    Represents a reference to a background prefetch.
    """

    # The cache pool that we're using.
    pool: "FileStoreCachePool"
    # The entry in the cache pool that we're using.
    entry: "_PendingTask"

    # If get() was called, this will be the data (uncached).
    _data: Optional[tuple[T, SourceInfo]] = None

    def get(self, sample: Any = None) -> T:
        """
        Returns the data and adds the source info to the sample.
        If the background job hasn't started, we cancel it,
        do a direct read, and remove ourselves from the pool's references.
        Otherwise, we wait for the job to finish, read from cache, and remove ourselves.
        """
        if self._data is not None:
            return self._data
        self._data = self.pool._get_data(self.ds, self.fname, self.entry)
        assert self._data is not None
        add_source_info(sample, self._data[1])
        return self._data[0]

    def __hash__(self) -> int:
        """Allows usage in sets and dicts as key."""
        return hash((id(self.ds), self.fname))

    def __eq__(self, other: Any) -> bool:
        """Allows usage in sets and dicts as key. Compares the data source and the filename."""
        if not isinstance(other, Lazy):
            return False
        return self.ds is other.ds and self.fname == other.fname

    def __del__(self):
        if self._data is None:
            with self.pool._lock:
                # Data was never fetched, still decrement refcount to delete the cache entry
                self.pool._decrement_refcount_and_cleanup((self.ds.get_path(), self.fname))


@edataclass
class _PendingTask:
    """Dataclass for storing a pending background task"""

    # The dataset that we're caching.
    ds: FileStore
    # The file name that we're caching.
    fname: str
    # The future for the background task that sends the data to the cache.
    send_to_cache_future: Future
    # The number of references to the cache entry.
    refcount: int = 1
    # The size of the data to be cached.
    data_size: int = 0
    # Whether the data is required now, i.e. a reading thread is waiting for it.
    require_data_now: bool = False
    # The path to the cache file.
    cache_path: Optional[Path] = None
    # The source info for the data.
    source_info: Optional[SourceInfo] = None


class FileStoreCachePool(CachePool, ForkMixin):
    """
    Manages a thread pool to pre-fetch data onto an SSD cache.
    Each (ds, fname) has one Future (one read). Multiple requests
    share that same future. We track usage with a refcount.

    To avoid multi-process collisions, we generate a random subfolder
    for each instance.
    """

    cache_dir: Path
    max_cache_size: int
    max_cache_count: int
    current_cache_size: int
    current_cache_count: int
    method: Literal["raw", "pickle"]

    # Thread pool for out-caching tasks
    _worker_pool: Optional[ThreadPoolExecutor] = None
    # (ds.path, fname) -> PendingTask
    _pending_tasks: Dict[Tuple[str, str], _PendingTask]

    # Lock for all shared structures
    _lock: threading.Lock
    # Condition variable to signal when cache space is available
    _cache_space_available: threading.Condition
    # Whether the pool is shutting down
    _shutting_down: bool = False

    def __init__(
        self,
        *,
        parent_cache_dir: Optional[Path] = None,
        num_workers: int = 8,
        max_cache_size_gbytes: float = 1024,
        max_cache_count: int = 10_000_000,
        method: Literal["raw", "pickle"] = "raw",
    ):
        """
        Initialize the cache pool.

        Args:
            parent_cache_dir: The parent directory for the cache.
            num_workers: The number of worker threads to use for copying the data to the cache for lazy loading.
            max_cache_size_gbytes: The maximum size of the cache in gigabytes. If the cache exceeds this size,
                the prefetching will wait until the cache is below this size.
            max_cache_count: The maximum number of files in the cache. If the cache exceeds this number,
                the prefetching will wait until the cache is below this number.
            method: The method to use for caching. "raw" store the non-decoded raw data. "pickle": first decode the data
                and then store the pickled data.
        """
        super().__init__()
        # If no parent directory is given, create a temp directory
        if parent_cache_dir is None:
            parent_cache_dir = Path(tempfile.gettempdir())
        self.parent_cache_dir = parent_cache_dir
        self.num_workers = num_workers

        # Initialize the cache pool (process volatile fields)
        self.__after_fork__(initial=True)

        self.method = method

        # We'll store _pending_tasks in the form:
        #   (ds.path, fname) -> PendingTask
        self._pending_tasks = {}

        # Cache size management
        self.max_cache_size = int(max_cache_size_gbytes * (1024**3))
        self.max_cache_count = max_cache_count
        self.current_cache_size = 0
        self.current_cache_count = 0

        # A lock to protect all shared structures
        self._lock = threading.Lock()

        # Condition variable to signal when cache space is available
        self._cache_space_available = threading.Condition(self._lock)

    def get(self, ds: FileStore, fname: str, sample: Any = None) -> Any:
        """
        Synchronous read from the dataset (no cache usage).
        """
        return ds.get(fname, sample)

    def _get_data(self, ds: FileStore, fname: str, entry: _PendingTask) -> tuple[Any, SourceInfo]:
        """
        Get the data for a given file from the cache and purge cache if no references are left.

        * If the cache-out is complete, read from cache.
        * If the cache-out is currently prefetching the data to local storage, wait until it's done.
        * If the cache-out job is waiting for space, skip the cache and do a direct read.
        * If the cache-out job is queued for caching, cancel and do a direct read.
        * If the cache-out job failed, raise through and keep for other references.
        * If the cache-out job is cancelled, requeue if there are other references waiting for it.
        """
        result: tuple[Any, SourceInfo]
        with self._lock:
            try:
                # Attempt to cancel if the job hasn't started
                if entry.send_to_cache_future.cancel():
                    was_cached = False
                    try:
                        # Cancelled => job never ran. We'll do a direct read.
                        result = ds[fname]
                    finally:
                        # Decrement refcount
                        self._decrement_refcount_and_cleanup(key=(ds.get_path(), fname))
                else:
                    # Future is already running or done.
                    # Release the lock so the background job can proceed,
                    # then reacquire it after waiting. Otherwise we might block the worker.
                    entry.require_data_now = True
                    self._cache_space_available.notify_all()
                    self._lock.release()

                    # If the job failed, let's keep the exception for other references.
                    was_cached = True

                    try:
                        # Can raise exception if job failed
                        was_cached = entry.send_to_cache_future.result()

                        if was_cached:
                            # The job is complete; read from cache
                            result = self._read_from_cache(entry)
                        else:
                            # The job failed, so we'll do a direct decode
                            result = ds[fname]
                    finally:
                        self._lock.acquire()
                        entry.require_data_now = False

                        # Decrement refcount
                        self._decrement_refcount_and_cleanup(key=(ds.get_path(), fname))
            finally:
                if entry.refcount > 0 and not was_cached:
                    # TODO: Could write to cache here, data is already fetched.
                    # Write the result to the cache
                    # Requeue the job, there is another reference to the cache entry
                    entry.send_to_cache_future = self._worker_pool.submit(
                        self._cache_out_task, ds, fname, entry
                    )

            return result

    def _cache_out_task(self, ds: FileStore, fname: str, entry: _PendingTask) -> bool:
        with self._lock:
            if self._shutting_down:
                return False

        # Perform the data read
        if self.method == "raw":
            if isinstance(ds, DecodeFileStore):
                data, entry.source_info = ds.inner_reader[fname]
            else:
                data, entry.source_info = ds[fname]
        elif self.method == "pickle":
            data, entry.source_info = ds[fname]
            data = pickle.dumps(data)
        else:
            raise ValueError(f"Invalid method: {self.method}")

        # Wait until there's enough space in the cache
        with self._lock:
            entry.data_size = file_size = len(data)

            while (
                self.current_cache_count + 1 > self.max_cache_count
                or self.current_cache_size + entry.data_size > self.max_cache_size
            ):
                # Release the lock and wait for notification
                self._cache_space_available.wait()
                if entry.require_data_now or self._shutting_down:
                    # At least one reference requires the data now, stop waiting for space and exit immediately
                    return False

            # Reserve the space
            self.current_cache_size += file_size
            self.current_cache_count += 1

            if self._shutting_down or entry.refcount <= 0:
                # No more references to this background job, don't write to cache
                return False

        try:
            assert entry.cache_path is None, (
                f"cache_path should be None, but is {entry.cache_path!r}"
            )
            # Write to cache
            cache_path = self._make_cache_path(ds, fname)
            self._write_to_cache(cache_path, data)
        except:
            with self._lock:
                # Revert the space reservation
                self.current_cache_size -= file_size
                self.current_cache_count -= 1
                self._cache_space_available.notify_all()
            raise
        else:
            with self._lock:
                entry.cache_path = cache_path
            # print(
            #     f"FSCP r={torch.distributed.get_rank()}, pid={os.getpid()}: Wrote to cache {cache_path} (rc={entry.refcount}, size={file_size}, name={fname})\n",
            #     end="",
            # )

        # Data is cached now, return True
        return True

    def get_lazy(self, ds: FileStore, fname: str) -> FileCacheLazy:
        """
        Schedule a background pre-fetch. If multiple calls come in for the same (ds, fname),
        they'll share the same Future and increment reference counts.
        """
        key = (ds.get_path(), fname)
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("Cache pool is already shutting down")
            entry = self._pending_tasks.get(key)
            if entry:
                # Already have a background task for this (ds, fname)
                entry.refcount += 1
            else:
                # Create a new background task
                entry = _PendingTask(
                    ds=ds,
                    fname=fname,
                    send_to_cache_future=None,
                )
                self._pending_tasks[key] = entry

                entry.send_to_cache_future = self._worker_pool.submit(
                    self._cache_out_task, ds, fname, entry
                )

        return FileCacheLazy(ds=ds, fname=fname, pool=self, entry=entry)

    def close(self) -> None:
        """
        Shutdown the pool, wait for tasks, and clear our structures.
        """
        with self._lock:
            self._shutting_down = True
            for entry in self._pending_tasks.values():
                entry.send_to_cache_future.cancel()
            self._cache_space_available.notify_all()
        self._worker_pool.shutdown(wait=True)
        with self._lock:
            self._pending_tasks.clear()

    def _decrement_refcount_and_cleanup(self, key: Tuple[FileStore, str]) -> None:
        """
        Decrement the reference count in `_pending_tasks`.
        If it hits zero, remove the entry. Optionally remove the file if so.
        Assumes the caller holds `self._lock`.
        """
        entry = self._pending_tasks.get(key)
        if not entry:
            # Already cleaned up
            return

        entry.refcount -= 1
        if entry.refcount <= 0:
            # No more references to this background job
            del self._pending_tasks[key]

            self._remove_cached_file(entry)

            assert entry.refcount == 0, f"refcount should be 0: {entry.refcount}"

    def _make_cache_path(self, ds: FileStore, fname: str) -> Path:
        # This is safe, because the parent cache dir is unique per instance.
        ds_hash = hashlib.md5(ds.get_path().encode("utf-8")).hexdigest()
        fn_hash = hashlib.md5(fname.encode("utf-8")).hexdigest()
        # ds_hash = str(ds.get_path()).replace("/", "_")
        # fn_hash = fname.replace("/", "_")
        return self.cache_dir / f"{ds_hash}_{fn_hash}"

    def _write_to_cache(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def _read_from_cache(self, entry: _PendingTask) -> tuple[Any, SourceInfo]:
        assert entry.source_info is not None, "source_info should have been set"
        with open(entry.cache_path, "rb") as f:
            if self.method == "raw":
                raw = f.read()
                if isinstance(entry.ds, DecodeFileStore):
                    return entry.ds.decoder.decode(entry.fname, raw), entry.source_info
                else:
                    return raw, entry.source_info
            else:
                return pickle.load(f), entry.source_info

    def _remove_cached_file(self, entry: _PendingTask) -> None:
        """
        Removes a file from disk and updates size counters.
        Assumes the caller holds `self._lock`.
        """
        if entry.cache_path is None:
            return
        if not entry.cache_path.exists():
            return

        try:
            # print(
            #     f"FSCP r={torch.distributed.get_rank()}, pid={os.getpid()}: Removing cached file {entry.cache_path} (rc={entry.refcount})\n",
            #     end="",
            # )
            entry.cache_path.unlink()
        except OSError:
            pass
        entry.cache_path = None

        if entry.data_size > 0:
            self.current_cache_size -= entry.data_size
            self.current_cache_count -= 1
            # Notify waiting threads that space is now available
            self._cache_space_available.notify_all()

    def __before_fork__(self):
        # Ensure the worker pool is shutdown before the fork
        assert len(self._pending_tasks) == 0, "Pending tasks should be empty before fork"
        # print(
        #     f"FSCP r={torch.distributed.get_rank()}, pid={os.getpid()}: Before fork for oid={id(self)} random_suffix={self.cache_dir.name!r}\n",
        #     end="",
        # )
        self._worker_pool.shutdown(wait=True)
        self._worker_pool = None

    def __after_in_child_fork__(self):
        self.__after_fork__()

    def __after_in_parent_fork__(self):
        self.__after_fork__()

    def __after_fork__(self, initial: bool = False):
        random_suffix = "".join(
            random.Random(os.getpid() ^ random.randint(0, 2**32)).choices(
                string.ascii_lowercase + string.digits, k=16
            )
        )
        assert self._worker_pool is None
        self._worker_pool = ThreadPoolExecutor(
            max_workers=self.num_workers, thread_name_prefix="CacheWorker"
        )
        # Create a random subdirectory name to avoid collisions with other processes
        # As the global random generator is cloned across processes, we need to use a process-specific seed
        self.cache_dir = (self.parent_cache_dir / f"cache_{random_suffix}").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # if initial:
        #     print(
        #         f"FSCP r={torch.distributed.get_rank()}, pid={os.getpid()}: Init oid={id(self)} random_suffix={random_suffix!r}\n",
        #         end="",
        #     )
        # else:
        #     print(
        #         f"FSCP r={torch.distributed.get_rank()}, pid={os.getpid()}: After fork for pid={os.getpid()} oid={id(self)} random_suffix={random_suffix!r}\n",
        #         end="",
        #     )

    def __str__(self):
        return f"FileStoreCachePool(cache_dir={self.cache_dir}, max_cache_size={self.max_cache_size}, max_cache_count={self.max_cache_count}, method={self.method}, current_cache_size={self.current_cache_size}, current_cache_count={self.current_cache_count})"
