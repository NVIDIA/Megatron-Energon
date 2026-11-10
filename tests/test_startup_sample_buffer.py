# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import io
import json
import random
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from megatron.energon.epathlib import EPath
from megatron.energon.epathlib.bounded_io import open_sample_file
from megatron.energon.flavors.jsonl.ijsonl import IJsonlFile
from megatron.energon.flavors.webdataset.itar import ITarFile


class SampleBufferTest(unittest.TestCase):
    def test_sample_buffer_preserves_storage_client(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            path.write_bytes(b"{}\n")
            with patch.object(EPath, "local_path", side_effect=AssertionError("bypassed MSC")):
                with open_sample_file(EPath(path)) as stream:
                    self.assertEqual(stream.read(), b"{}\n")
                    self.assertEqual(type(stream).__module__, "multistorageclient.file")

    def test_jsonl_sparse_and_adjacent_records_including_large_payload(self):
        records = [
            json.dumps({"key": index, "value": "x" * size}).encode() + b"\n"
            for index, size in enumerate([1, 32000, 200, 64000, 3, 3 * 1024 * 1024, 8000])
        ]
        offsets = [sum(map(len, records[:index])) for index in range(len(records))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            path.write_bytes(b"".join(records))
            with IJsonlFile(open_sample_file(EPath(path))) as stream:
                for index in [5, 0, 1, 2, 6, 3, 4, 5]:
                    self.assertEqual(
                        stream.next(offsets[index], len(records[index])), records[index]
                    )
                stream.seek(0)
                self.assertEqual(list(stream), records)

    def test_tar_headers_payloads_random_access_and_independent_handles(self):
        rng = random.Random(42)
        values = {
            f"{index}.bin": rng.randbytes(size)
            for index, size in enumerate((1, 512, 65000, 3 * 1024 * 1024))
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.tar"
            with tarfile.open(path, "w") as archive:
                for name, value in values.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(value)
                    archive.addfile(info, io.BytesIO(value))
            with (
                open_sample_file(EPath(path)) as stream,
                open_sample_file(EPath(path)) as other,
            ):
                stream.seek(100)
                self.assertEqual(other.tell(), 0)
                stream.seek(0)
                with ITarFile.open(fileobj=stream, mode="r:") as archive:
                    for name in reversed(values):
                        self.assertEqual(archive.extractfile(name).read(), values[name])

    def test_non_posix_uses_original_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            path.write_bytes(b'{"sample": true}\n')
            with patch.object(EPath, "is_local", return_value=False):
                with open_sample_file(EPath(path)) as stream:
                    self.assertEqual(stream.read(), path.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
