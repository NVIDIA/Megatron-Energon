# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from megatron.energon.cache.base import FileStoreDecoder
from megatron.energon.cache.file_cache_pool import FileStoreCachePool
from megatron.energon.cache.file_store import DecodeFileStore, SystemFileStore
from megatron.energon.cache.no_cache import NoCachePool


class JsonDecoder(FileStoreDecoder):
    def decode(self, fname, data):
        return json.loads(data)


class TestRepeatedLazyCacheReads(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "sample.txt").write_bytes(b"cached sample")
        self.store = SystemFileStore(self.root)

    def pool(self, method):
        pool = FileStoreCachePool(
            parent_cache_dir=self.root / "cache", num_workers=1, method=method
        )
        self.addCleanup(pool.close)
        return pool

    def test_completed_cache_returns_the_same_value_on_every_read(self):
        for method in ("raw", "pickle"):
            with self.subTest(method=method):
                pool = self.pool(method)
                lazy = pool.get_lazy(self.store, "sample.txt")
                self.assertTrue(lazy.entry.send_to_cache_future.result(timeout=10))
                first = lazy.get()
                self.assertEqual(first, b"cached sample")
                self.assertEqual(lazy.entry.refcount, 0)
                self.assertEqual(list(pool.cache_dir.iterdir()), [])
                for _ in range(3):
                    self.assertIs(lazy.get(), first)
                    self.assertEqual(lazy.entry.refcount, 0)

    def test_provenance_is_added_to_each_recipient(self):
        for method in ("raw", "pickle"):
            for recipient_type in (dict, lambda: SimpleNamespace(__sources__=None)):
                with self.subTest(method=method, recipient=recipient_type):
                    lazy = self.pool(method).get_lazy(self.store, "sample.txt")
                    self.assertTrue(lazy.entry.send_to_cache_future.result(timeout=10))
                    first, second = recipient_type(), recipient_type()
                    lazy.get(first)
                    lazy.get(second)
                    if isinstance(first, dict):
                        self.assertIn("__sources__", second)
                        sources = second["__sources__"]
                        self.assertEqual(sources, first["__sources__"])
                    else:
                        sources = second.__sources__
                        self.assertEqual(sources, first.__sources__)
                    self.assertEqual(len(sources), 1)
                    self.assertEqual(sources[0].file_names, ("sample.txt",))
                    self.assertEqual(str(sources[0].dataset_path), str(self.root))

    def test_repeated_read_does_not_release_another_reference(self):
        for method in ("raw", "pickle"):
            with self.subTest(method=method):
                pool = self.pool(method)
                first = pool.get_lazy(self.store, "sample.txt")
                second = pool.get_lazy(self.store, "sample.txt")
                self.assertIs(first.entry, second.entry)
                self.assertTrue(first.entry.send_to_cache_future.result(timeout=10))
                value = first.get()
                self.assertIs(first.get(), value)
                self.assertEqual(second.entry.refcount, 1)
                self.assertTrue(second.entry.cache_path.is_file())
                self.assertEqual(second.get(), value)
                self.assertEqual(second.entry.refcount, 0)
                self.assertEqual(list(pool.cache_dir.iterdir()), [])

    def test_direct_read_when_prefetch_is_queued_remains_repeatable(self):
        pool = self.pool("raw")
        started, release = threading.Event(), threading.Event()

        def occupy_worker():
            started.set()
            release.wait(timeout=10)

        blocker = pool._worker_pool.submit(occupy_worker)
        try:
            self.assertTrue(started.wait(timeout=5))
            lazy = pool.get_lazy(self.store, "sample.txt")
            self.assertFalse(lazy.entry.send_to_cache_future.running())
            first = lazy.get()
            self.assertTrue(lazy.entry.send_to_cache_future.cancelled())
            self.assertEqual(first, b"cached sample")
            self.assertIs(lazy.get(), first)
            self.assertEqual(lazy.entry.refcount, 0)
        finally:
            release.set()
            blocker.result(timeout=10)

    def test_decoded_values_keep_their_type_including_none(self):
        for method in ("raw", "pickle"):
            for expected in ({"number": 3}, None, False):
                with self.subTest(method=method, value=expected):
                    (self.root / "sample.json").write_text(json.dumps(expected), encoding="utf-8")
                    store = DecodeFileStore(self.store, decoder=JsonDecoder())
                    lazy = self.pool(method).get_lazy(store, "sample.json")
                    self.assertTrue(lazy.entry.send_to_cache_future.result(timeout=10))
                    first = lazy.get()
                    self.assertEqual(first, expected)
                    self.assertIs(lazy.get(), first)

    def test_repeated_sample_provenance_matches_direct_lazy(self):
        cached = self.pool("raw").get_lazy(self.store, "sample.txt")
        direct = NoCachePool().get_lazy(self.store, "sample.txt")
        self.assertTrue(cached.entry.send_to_cache_future.result(timeout=10))
        actual_sample, expected_sample = {}, {}
        for _ in range(3):
            self.assertEqual(cached.get(actual_sample), direct.get(expected_sample))
            self.assertEqual(actual_sample, expected_sample)


if __name__ == "__main__":
    unittest.main()
