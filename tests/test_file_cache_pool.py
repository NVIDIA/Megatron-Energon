# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

from megatron.energon.cache import DecodeFileStore, FileCacheLazy, FileStore, FileStoreCachePool
from megatron.energon.flavors.webdataset.sample_decoder import SampleDecoder


class MockFileStore(FileStore):
    """Mock implementation of FileStore for testing"""

    def __init__(self, data: Optional[Dict[str, Any]] = None, path: str = "mock_store"):
        self._data = data if data is not None else {}
        self._path = path

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get_path(self) -> str:
        return self._path


class MockDecoder(SampleDecoder):
    """Mock decoder for DecodeFileStore"""

    def decode(self, fname: str, raw: bytes) -> Any:
        return f"{fname}: {raw.decode()}"


class TestFileStoreCachePool(unittest.TestCase):
    """Test cases for FileStoreCachePool"""

    def setUp(self):
        """Setup test environment before each test"""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up after each test"""
        self.temp_dir.cleanup()

    def test_get_method(self):
        """Test the synchronous get method"""
        # Create mock file stores
        mock_raw_file_store = MockFileStore(
            {
                "file1": b"test data 1",
                "file2": b"test data 2",
                "file3": b"test data 3",
            }
        )

        mock_decode_file_store = DecodeFileStore(
            decoder=MockDecoder(),
            inner_reader=mock_raw_file_store,
        )
        pool = FileStoreCachePool(parent_cache_dir=self.temp_path)
        try:
            # get should directly read from the dataset without caching
            result = pool.get(mock_raw_file_store, "file1")
            assert result == b"test data 1"

            # get should directly read from the dataset without caching
            result = pool.get(mock_decode_file_store, "file1")
            assert result == "file1: test data 1", result
        finally:
            pool.close()

    def test_get_lazy_method(self):
        """Test the lazy get method for background prefetching"""
        pool = FileStoreCachePool(parent_cache_dir=self.temp_path)
        # Create mock file stores
        mock_raw_file_store = MockFileStore(
            {
                "file1": b"test data 1",
            }
        )
        try:
            # Request lazy loading
            lazy_ref = pool.get_lazy(mock_raw_file_store, "file1")

            # Verify the return type
            assert isinstance(lazy_ref, FileCacheLazy)

            # Wait for the background task
            lazy_ref.entry.send_to_cache_future.result()

            # Check that the file exists in the cache directory
            cache_files = list(pool.cache_dir.glob("*"))
            assert len(cache_files) == 1

            # Get the data
            result = lazy_ref.get()
            assert result == b"test data 1"
        finally:
            pool.close()

    def test_shared_references(self):
        """Test that multiple references share the same background task"""
        pool = FileStoreCachePool(parent_cache_dir=self.temp_path)
        # Create mock file stores
        mock_raw_file_store = MockFileStore(
            {
                "file1": b"test data 1",
            }
        )
        try:
            # Check that the file exists in the cache directory
            cache_files = list(pool.cache_dir.rglob("*"))
            assert len(cache_files) == 0

            # Request lazy loading for the same file twice
            lazy_ref1 = pool.get_lazy(mock_raw_file_store, "file1")
            lazy_ref2 = pool.get_lazy(mock_raw_file_store, "file1")

            # Check that they share the same entry
            assert lazy_ref1.entry is lazy_ref2.entry

            # Check that refcount is 2
            assert lazy_ref1.entry.refcount == 2

            # Wait for the background task
            lazy_ref1.entry.send_to_cache_future.result()

            # Check that the file exists in the cache directory
            cache_files = list(pool.cache_dir.rglob("*"))
            assert len(cache_files) == 1, cache_files

            # Get data from both references
            result1 = lazy_ref1.get()
            assert lazy_ref1.entry.refcount == 1
            result2 = lazy_ref2.get()
            assert lazy_ref1.entry.refcount == 0

            # Check that the file exists in the cache directory
            cache_files = list(pool.cache_dir.rglob("*"))
            assert len(cache_files) == 0

            assert result1 == b"test data 1"
            assert result2 == b"test data 1"
        finally:
            pool.close()

    def test_cache_size_management(self):
        """Test that the cache respects size limits and evicts files"""
        # Create a cache pool with strict limits
        pool = FileStoreCachePool(
            parent_cache_dir=self.temp_path,
            max_cache_size_gbytes=0.0001,  # ~100KB
            max_cache_count=2,
            num_workers=1,
        )
        # Set to a safe byte size
        pool.max_cache_size = 75_000

        mock_raw_file_store = MockFileStore(
            {
                "large_file1": b"a" * 50_000,
                "large_file2": b"b" * 50_000,
                "large_file3": b"c" * 50_000,
                "large_file4": b"d" * 25_000,
                "large_file5": b"e" * 25_000,
                "large_file6": b"f" * 25_000,
            }
        )

        try:
            # Enqueue all fetches
            lazy1 = pool.get_lazy(mock_raw_file_store, "large_file1")
            lazy2 = pool.get_lazy(mock_raw_file_store, "large_file2")
            lazy3 = pool.get_lazy(mock_raw_file_store, "large_file3")
            lazy4 = pool.get_lazy(mock_raw_file_store, "large_file4")
            lazy2_2 = pool.get_lazy(mock_raw_file_store, "large_file2")
            lazy2_3 = pool.get_lazy(mock_raw_file_store, "large_file2")
            lazy3_2 = pool.get_lazy(mock_raw_file_store, "large_file3")
            lazy5 = pool.get_lazy(mock_raw_file_store, "large_file4")
            lazy6 = pool.get_lazy(mock_raw_file_store, "large_file5")

            def status():
                return [
                    lazy1.entry.send_to_cache_future.done(),
                    lazy2.entry.send_to_cache_future.done(),
                    lazy3.entry.send_to_cache_future.done(),
                    lazy4.entry.send_to_cache_future.done(),
                    lazy5.entry.send_to_cache_future.done(),
                    lazy6.entry.send_to_cache_future.done(),
                ]

            # lazy2_2 and lazy2_3 should share the same entry as lazy2
            assert lazy2_2.entry is lazy2.entry
            assert lazy2_3.entry is lazy2.entry

            lazy1.entry.send_to_cache_future.result(timeout=1)
            # Wait for the background tasks to finish
            time.sleep(0.5)

            print("Checking cache status")
            # They should not be able to finish, because the cache is full
            assert status() == [True, False, False, False, False, False], status()

            # Check cache count and size before second file
            assert pool.current_cache_count == 1, pool.current_cache_count
            assert pool.current_cache_size == 50_000, pool.current_cache_size

            print("Fetching lazy2_3")
            # Now, fetching the second file should still work directly and ignore the caching
            # But it will requeue fetching the second file to the background thread for the remaining lazies.
            result2_3 = lazy2_3.get()
            assert result2_3 == b"b" * 50_000

            # They should not be able to finish, because the cache is full
            assert status() == [True, False, False, False, False, False], status()

            print("Fetching lazy1")
            # Fetch
            result1 = lazy1.get()
            assert result1 == b"a" * 50_000

            lazy3.entry.send_to_cache_future.result(timeout=1)

            time.sleep(0.5)

            # Second file is now queued at the end.
            # File 3 and 4 should now be cached.
            assert status() == [True, False, True, True, False, False], status()

            print("Fetching lazy3")
            assert pool.current_cache_count == 2
            assert pool.current_cache_size == 50_000
            result3 = lazy3.get()
            assert result3 == b"c" * 50_000

            time.sleep(0.5)

            # Space by large_file3 is still occupied in cache
            assert status() == [True, False, True, True, False, False], status()

            result3_2 = lazy3_2.get()
            assert result3_2 == b"b" * 50_000

            time.sleep(0.5)

            # Space by large_file3 was freed now, 4, 5, and 6 should fit now, large_file2 not yet
            assert status() == [True, False, True, True, True, True], status()

            result4 = lazy4.get()
            assert result4 == b"d" * 25_000

            time.sleep(0.5)
            # Nothing changed, no space for large_file2 still
            assert status() == [True, False, True, True, True, True], status()

            result5 = lazy5.get()
            assert result5 == b"e" * 25_000

            time.sleep(0.5)

            # Now large_file2 can be cached
            assert status() == [True, True, True, True, True, True], status()

            result6 = lazy6.get()
            assert result6 == b"f" * 25_000

            result2 = lazy2.get()
            assert result2 == b"b" * 50_000

            # Cache should be empty now
            assert pool.current_cache_count == 0
            assert pool.current_cache_size == 0
        finally:
            pool.close()

    def test_raw_method(self):
        """Test the 'raw' caching method with DecodeFileStore"""
        pool = FileStoreCachePool(parent_cache_dir=self.temp_path, method="raw")
        mock_raw_file_store = MockFileStore(
            {
                "file1": b"test data 1",
            }
        )
        mock_decode_file_store = DecodeFileStore(
            decoder=MockDecoder(),
            inner_reader=mock_raw_file_store,
        )
        try:
            # Request lazy loading
            lazy_ref = pool.get_lazy(mock_decode_file_store, "file1")

            # Wait for background task
            time.sleep(0.5)

            # Get the data - should be decoded
            result = lazy_ref.get()
            assert result == "file1: test data 1"
        finally:
            pool.close()

    def test_pickle_method(self):
        """Test the 'pickle' caching method"""
        pool = FileStoreCachePool(parent_cache_dir=self.temp_path, method="pickle")
        mock_raw_file_store = MockFileStore(
            {
                "complex": b"test data 1",
            }
        )
        mock_decode_file_store = DecodeFileStore(
            decoder=MockDecoder(),
            inner_reader=mock_raw_file_store,
        )
        try:
            # Request lazy loading
            lazy_ref = pool.get_lazy(mock_decode_file_store, "file1")

            # Wait for background task
            lazy_ref.entry.send_to_cache_future.result()

            # Get the data - should be unpickled correctly
            result = lazy_ref.get()
            assert result == "file1: test data 1"

            # Request lazy loading
            lazy_ref = pool.get_lazy(mock_raw_file_store, "file1")

            # Wait for background task
            lazy_ref.entry.send_to_cache_future.result()

            # Get the data - should be unpickled correctly
            result = lazy_ref.get()
            assert result == b"test data 1"
        finally:
            pool.close()

    def test_concurrent_access(self):
        """Test concurrent access to the cache pool"""
        pool = FileStoreCachePool(parent_cache_dir=self.temp_path)
        mock_raw_file_store = MockFileStore(
            {
                "file1": b"test data 1",
            }
        )
        results = []

        def worker(filename):
            lazy_ref = pool.get_lazy(mock_raw_file_store, filename)
            result = lazy_ref.get()
            results.append(result)

        try:
            # Start multiple threads accessing the same file
            threads = []
            for i in range(5):
                t = threading.Thread(target=worker, args=("file1",))
                threads.append(t)
                t.start()

            # Wait for all threads to complete
            for t in threads:
                t.join()

            # All threads should get the correct result
            for r in results:
                assert r == b"test data 1"
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()
