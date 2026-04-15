# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import sqlite3
import tempfile
import unittest
from pathlib import Path

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.indexing import (
    DuplicateSampleKeyError,
    SqliteIndexReader,
    SqliteIndexWriter,
)


class TestSqliteIndexWriter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sqlite_path = Path(self.temp_dir.name) / "index.sqlite"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_close_does_not_create_duplicate_sample_key_index(self):
        writer = SqliteIndexWriter(EPath(self.sqlite_path))
        writer.append_sample(
            tar_file_id=0,
            sample_key="sample-0",
            sample_index=0,
            byte_offset=0,
            byte_size=16,
        )
        writer.close()

        with sqlite3.connect(self.sqlite_path) as db:
            sample_indexes = {
                row[1]: row
                for row in db.execute("PRAGMA index_list(samples)")
            }
            unique_sample_key_indexes = []
            for index_name, row in sample_indexes.items():
                is_unique = bool(row[2])
                if not is_unique:
                    continue
                index_columns = [
                    index_info_row[2]
                    for index_info_row in db.execute(
                        f"PRAGMA index_info('{index_name}')"
                    )
                ]
                if "sample_key" in index_columns:
                    unique_sample_key_indexes.append(index_name)

        self.assertIn("idx_samples_by_tar_and_idx", sample_indexes)
        self.assertNotIn("idx_samples_sample_key", sample_indexes)
        self.assertTrue(
            unique_sample_key_indexes,
            "Expected a UNIQUE index on samples that includes sample_key",
        )

    def test_sample_key_lookup_still_works_with_implicit_unique_index(self):
        writer = SqliteIndexWriter(EPath(self.sqlite_path))
        writer.append_sample(
            tar_file_id=3,
            sample_key="sample-lookup",
            sample_index=7,
            byte_offset=128,
            byte_size=64,
        )
        writer.close()

        reader = SqliteIndexReader(EPath(self.sqlite_path))
        try:
            pointer = reader.get_sample_pointer_by_key("sample-lookup")
        finally:
            reader.close()

        self.assertEqual(pointer.tar_file_id, 3)
        self.assertEqual(pointer.byte_offset, 128)
        self.assertEqual(pointer.byte_size, 64)

    def test_duplicate_sample_key_still_raises(self):
        writer = SqliteIndexWriter(EPath(self.sqlite_path))
        writer.append_sample(
            tar_file_id=0,
            sample_key="sample-dup",
            sample_index=0,
            byte_offset=0,
            byte_size=16,
        )

        with self.assertRaises(DuplicateSampleKeyError):
            writer.append_sample(
                tar_file_id=0,
                sample_key="sample-dup",
                sample_index=1,
                byte_offset=16,
                byte_size=16,
            )

        writer.close()
