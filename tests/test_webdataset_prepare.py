# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import unittest
from pathlib import Path

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.indexing import (
    DuplicateSampleKeyError,
    SqliteIndexWriter,
)


class TestSqliteIndexWriterBatching(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sqlite_path = EPath(Path(self.temp_dir.name) / "index.sqlite")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_append_samples_batch_reports_duplicate_sample_key(self):
        with SqliteIndexWriter(self.sqlite_path) as writer:
            with self.assertRaises(DuplicateSampleKeyError) as exc:
                writer.append_samples(
                    [
                        (0, "first", 0, 0, 10),
                        (0, "duplicate", 1, 10, 10),
                        (1, "duplicate", 0, 20, 10),
                    ]
                )
            assert writer.db is not None
            row_count = writer.db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

        self.assertEqual(exc.exception.sample_key, "duplicate")
        self.assertEqual(row_count, 0)


if __name__ == "__main__":
    unittest.main()
