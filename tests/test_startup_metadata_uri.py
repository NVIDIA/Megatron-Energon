# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from megatron.energon.cache.file_store import SystemFileStore


class OptimizationsTest(unittest.TestCase):
    def test_metadata_resolves_dss_and_escapes_uri(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index #with? chars.sqlite"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE media_metadata (entry_key TEXT)")
                connection.execute("INSERT INTO media_metadata VALUES ('sample')")

            class MappedPath:
                def __truediv__(self, _):
                    return self

                def __str__(self):
                    return "dss://test@v0/.nv-meta/index.sqlite"

                def is_file(self):
                    return True

                def local_path(self):
                    return path

            store = SystemFileStore()
            store.base_dir = MappedPath()
            with patch(
                "megatron.energon.cache.file_store.ensure_local_copy",
                side_effect=lambda p: p,
            ):
                reader = store._ensure_media_metadata_reader()
                self.assertEqual(
                    reader.select_one("SELECT entry_key FROM media_metadata"),
                    ("sample",),
                )
                with self.assertRaises(sqlite3.OperationalError):
                    reader.select_one("INSERT INTO media_metadata VALUES ('forbidden')")
                reader.thread_close()

    def test_missing_metadata_does_not_create_database(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SystemFileStore(directory)
            with self.assertRaisesRegex(RuntimeError, "database missing"):
                store._ensure_media_metadata_reader()
            self.assertFalse((Path(directory) / ".nv-meta").exists())


if __name__ == "__main__":
    unittest.main()
