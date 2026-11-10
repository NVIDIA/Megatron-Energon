# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.jsonl.ijsonl import IJsonlIndexReader
from megatron.energon.flavors.webdataset.itar import TarIndexReader


class OptimizationsTest(unittest.TestCase):
    def test_index_random_access_iteration_boundaries_and_remote_fallback(self):
        values = list(range(0, 30000, 3))
        for reader_type, suffix in (
            (IJsonlIndexReader, ".jsonl"),
            (TarIndexReader, ".tar"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                data = Path(directory) / ("data" + suffix)
                data.with_suffix(suffix + ".idx").write_bytes(
                    struct.pack(f"{len(values)}Q", *values)
                )
                with reader_type(str(data)) as reader:
                    for index in [9999, 1023, 1024, 0, 1, 9998, 2048]:
                        self.assertEqual(reader[index], values[index])
                    self.assertEqual(list(reader), values)
                    for invalid in (-1, len(values)):
                        with self.assertRaises(IndexError):
                            reader[invalid]
                # A non-POSIX backend must still use its storage-client opener.
                with patch.object(EPath, "is_local", return_value=False):
                    with reader_type(str(data)) as reader:
                        self.assertEqual(reader[1024], values[1024])

    def test_index_buffer_preserves_storage_client(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.jsonl"
            data.with_suffix(".jsonl.idx").write_bytes(struct.pack("3Q", 0, 12, 24))
            # The bounded reader must use MSC, whose own path resolution and
            # instrumentation cannot be replaced by a direct EPath.local_path open.
            with patch.object(EPath, "local_path", side_effect=AssertionError("bypassed MSC")):
                with IJsonlIndexReader(str(data)) as reader:
                    self.assertEqual(reader[1], 12)
                    self.assertEqual(type(reader.ijsonl).__module__, "multistorageclient.file")


if __name__ == "__main__":
    unittest.main()
