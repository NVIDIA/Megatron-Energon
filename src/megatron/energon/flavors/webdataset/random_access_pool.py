# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import hashlib
import os
import random
import string
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from megatron.energon.flavors.base_dataset import RandomAccessDataset


# -----------------------------------------------------------------------
# Lazy
# -----------------------------------------------------------------------
@dataclass
class Lazy:
    """
    Represents a reference to a background prefetch. Each Lazy has:
      - The dataset (`ds`)
      - The sample name (`fname`)
      - The cache path
      - A back-reference to the pool, so we can do bookkeeping on get().
      - A key for looking up the shared future+refcount in `_pending_tasks`.
    """

    ds: RandomAccessDataset
    fname: str
    cache_path: Path
    pool: RandomAccessDatasetCachePool
    key: Tuple[RandomAccessDataset, str]

    def get(self) -> bytes:
        """
        Returns the data. If the background job hasn't started, we cancel it,
        do a direct read, and remove ourselves from the pool's references.
        Otherwise, we wait for the job to finish, read from cache, and remove ourselves.
        """
        with self.pool._lock:
            entry = self.pool._pending_tasks.get(self.key)
            if not entry:
                # Should not normally happen, but fallback to direct read
                return self.ds[self.fname]

            future = entry.future

            # Attempt to cancel if the job hasn't started
            if future.cancel():
                # Cancelled => job never ran. We'll do a direct read.
                data = self.ds[self.fname]
                # Decrement refcount and possibly remove from pool
                self._decrement_refcount_and_cleanup()
                return data
            else:
                # Future is already running or done.
                # Release the lock so the background job can proceed,
                # then reacquire it after waiting. Otherwise we might block the worker.
                self.pool._lock.release()

                future.result()  # can raise exception if job fails

                self.pool._lock.acquire()
                # The job is complete; read from cache
                data = self.pool._read_from_cache(self.cache_path)

                # Decrement refcount; if zero, remove file
                self._decrement_refcount_and_cleanup(remove_file_if_last=True)
                return data

    def _decrement_refcount_and_cleanup(self, remove_file_if_last: bool = False) -> None:
        """
        Decrement the reference count in `_pending_tasks`.
        If it hits zero, remove the entry. Optionally remove the file if so.
        Assumes the caller holds `self.pool._lock`.
        """
        entry = self.pool._pending_tasks.get(self.key)
        if not entry:
            # Already cleaned up
            return

        entry.refcount -= 1
        if entry.refcount <= 0:
            # No more references to this background job
            del self.pool._pending_tasks[self.key]

            if remove_file_if_last:
                self.pool._remove_cached_file(entry.cache_path)


# -----------------------------------------------------------------------
# Dataclass for storing a pending background task
# -----------------------------------------------------------------------
@dataclass
class PendingTask:
    future: Future
    refcount: int
    cache_path: Path


# -----------------------------------------------------------------------
# RandomAccessDatasetCachePool
# -----------------------------------------------------------------------
class RandomAccessDatasetCachePool:
    """
    Manages a thread pool to pre-fetch data onto an SSD cache.
    Each (ds, fname) has one Future (one read). Multiple requests
    share that same future. We track usage with a refcount.

    To avoid multi-process collisions, we generate a random subfolder
    for each instance.
    """

    def __init__(
        self,
        parent_cache_dir: Optional[Path] = None,
        num_workers: int = 8,
        max_cache_size_gbytes: int = 1024,  # 1 TB
        max_cache_count: int = 10_000_000,  # 10 M files
    ):
        # If no parent directory is given, create a temp directory
        if parent_cache_dir is None:
            parent_cache_dir = Path(tempfile.gettempdir())

        # Create a random subdirectory name to avoid collisions with other processes
        random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        self.cache_dir = (parent_cache_dir / f"radcp_{random_suffix}").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.worker_pool = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="RadcpCacheWorker"
        )

        # We'll store _pending_tasks in the form:
        #   (ds, fname) -> PendingTask
        self._pending_tasks: Dict[Tuple[RandomAccessDataset, str], PendingTask] = {}

        # Cache size management
        self.max_cache_size = max_cache_size_gbytes * (1024**3)
        self.max_cache_count = max_cache_count
        self.current_cache_size: int = 0
        self.current_cache_count: int = 0

        # We'll rely on dict insertion order to track oldest file for eviction
        # path -> file_size
        self._cache_records: Dict[Path, int] = {}

        # A lock to protect all shared structures
        self._lock = threading.Lock()

    def get(self, ds: RandomAccessDataset, fname: str) -> bytes:
        """
        Synchronous read from the dataset (no cache usage).
        """
        return ds[fname]

    def get_lazy(self, ds: RandomAccessDataset, fname: str) -> Lazy:
        """
        Schedule a background pre-fetch. If multiple calls come in for the same (ds, fname),
        they'll share the same Future and increment reference counts.
        """
        key = (ds, fname)
        with self._lock:
            entry = self._pending_tasks.get(key)
            if entry:
                # Already have a background task for this (ds, fname)
                entry.refcount += 1
                future = entry.future
                cache_path = entry.cache_path
            else:
                # Create a new background task
                cache_path = self._make_cache_path(ds, fname)

                def background_task():
                    # Perform the data read
                    data = ds[fname]
                    file_size = len(data)

                    # Evict if needed
                    self._maintain_cache_limits(file_size)

                    # Write to cache
                    self._write_to_cache(cache_path, data)

                    # Update stats
                    with self._lock:
                        self._cache_records[cache_path] = file_size
                        self.current_cache_size += file_size
                        self.current_cache_count += 1

                future = self.worker_pool.submit(background_task)
                self._pending_tasks[key] = PendingTask(
                    future=future,
                    refcount=1,
                    cache_path=cache_path,
                )

        return Lazy(ds=ds, fname=fname, cache_path=cache_path, pool=self, key=key)

    def close(self) -> None:
        """
        Shutdown the pool, wait for tasks, and clear our structures.
        """
        self.worker_pool.shutdown(wait=True)
        with self._lock:
            self._pending_tasks.clear()
            # We won't explicitly remove the disk files here,
            # but you could if you want a full cleanup.

    # ------------------------------------------------------------------------
    # Internal cache management
    # ------------------------------------------------------------------------

    def _make_cache_path(self, ds: RandomAccessDataset, fname: str) -> Path:
        ds_hash = hashlib.md5(ds.get_path().encode("utf-8")).hexdigest()
        fn_hash = hashlib.md5(fname.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{ds_hash}_{fn_hash}.bin"

    def _write_to_cache(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def _read_from_cache(self, path: Path) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def _remove_cached_file(self, path: Path) -> None:
        """
        Removes a file from disk and updates size counters.
        """
        if not path.exists():
            return

        file_size = self._cache_records.get(path, 0)
        try:
            os.remove(path)
        except OSError:
            pass

        if path in self._cache_records:
            del self._cache_records[path]

        if file_size > 0:
            self.current_cache_size -= file_size
            self.current_cache_count -= 1

    def _maintain_cache_limits(self, new_file_size: int) -> None:
        """
        Remove oldest files until we have room for `new_file_size`.
        """
        final_size = self.current_cache_size + new_file_size
        final_count = self.current_cache_count + 1

        while final_count > self.max_cache_count or final_size > self.max_cache_size:
            oldest_path = self._find_oldest_file()
            if oldest_path is None:
                break
            self._remove_cached_file(oldest_path)
            final_size = self.current_cache_size + new_file_size
            final_count = self.current_cache_count + 1

    def _find_oldest_file(self) -> Optional[Path]:
        """
        Return the path of the oldest cached file by insertion order, or None if empty.
        """
        if not self._cache_records:
            return None

        # First key is oldest in Python 3.7+
        return next(iter(self._cache_records))
