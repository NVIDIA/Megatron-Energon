import mmap
import os
import time
from typing import Optional, Tuple

CLEAR_OS_FS_CACHE = os.environ.get("CLEAR_OS_FS_CACHE", "0") == "1"

NUMBER_OF_OPENS = 0
NUMBER_OF_CLOSES = 0
NUMBER_OF_READS = 0
READ_BYTES = 0
READ_TIME_NS = 0
OPEN_TIME_NS = 0


class FileRandomReadPool:
    def __init__(self, cache_size: Optional[int] = None):
        self.pool = {}
        self.mmap_cache = {}
        self.size_cache = {}
        self.cache_size = cache_size

    def _open(self, path: str) -> Tuple[int, bytes]:
        global NUMBER_OF_OPENS, OPEN_TIME_NS, NUMBER_OF_CLOSES
        # print(f"Opening file {path}")
        NUMBER_OF_OPENS += 1
        start_time = time.perf_counter_ns()
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        if CLEAR_OS_FS_CACHE:
            # TODO: Remove for optimizations. Discard the cache for benchmarking.
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        # Advide that access is random.
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        OPEN_TIME_NS += time.perf_counter_ns() - start_time
        self.pool[path] = fd

        while self.cache_size is not None and len(self.pool) > self.cache_size:
            key = next(iter(self.pool))
            close_fd = self.pool.pop(key)
            self.mmap_cache.pop(key, None)
            self.size_cache.pop(key, None)
            # print(f"Closing file {key}")
            os.close(close_fd)
            NUMBER_OF_CLOSES += 1

        # self.mmap_cache[path] = mm = mmap.mmap(fd, length=0, offset=0, prot=mmap.PROT_READ)
        # self.size_cache[path] = len(mm)
        # return fd, mm
        return fd

    def get_handle(self, path: str) -> int:
        fd = self.pool.get(path)
        if fd is None:
            fd = self._open(path)
        return fd

    def willread(self, path: str, offset: int, length: int) -> None:
        fd = self.get_handle(path)
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_SEQUENTIAL)

    def get_size(self, path: str) -> int:
        if path not in self.size_cache:
            self.size_cache[path] = os.path.getsize(path)
        return self.size_cache[path]

    def read(self, path: str, offset: int, length: int) -> bytes:
        global READ_BYTES, READ_TIME_NS, NUMBER_OF_READS
        # print(f"Reading file {path} at offset {offset} with length {length}")
        mmap = self.mmap(path)
        assert len(mmap) >= offset + length, (
            f"mmap length: {len(mmap)}, offset: {offset}, length: {length}"
        )
        READ_BYTES += length
        NUMBER_OF_READS += 1
        start = time.perf_counter_ns()
        res = bytes(mmap[offset : offset + length])
        READ_TIME_NS += time.perf_counter_ns() - start
        return res

    def read2(self, path: str, offset: int, length: int) -> bytes:
        global READ_BYTES, READ_TIME_NS, NUMBER_OF_READS
        fd = self.get_handle(path)
        offset_alignment = offset % mmap.PAGESIZE
        offset = offset - offset_alignment
        read_end = length + offset_alignment
        length += offset_alignment
        start = time.perf_counter_ns()
        buf = mmap.mmap(fd, length, offset=offset, prot=mmap.PROT_READ)
        READ_BYTES += length
        NUMBER_OF_READS += 1
        res = bytes(buf[offset_alignment:read_end])
        READ_TIME_NS += time.perf_counter_ns() - start
        return res

    def mmap(self, path: str) -> bytes:
        mm = self.mmap_cache.get(path)
        if mm is None:
            fd = self.get_handle(path)
            self.mmap_cache[path] = mm = mmap.mmap(fd, length=0, offset=0, prot=mmap.PROT_READ)
        return mm
        # fd = self.get_handle(path)
        # buf = mmap.mmap(fd, self.get_size(path), offset=0, prot=mmap.PROT_READ)
        # return buf

    def close(self):
        for fd in self.pool.values():
            os.close(fd)
        self.pool.clear()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
