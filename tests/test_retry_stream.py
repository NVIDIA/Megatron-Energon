# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from megatron.energon.epathlib import EPath
from megatron.energon.retry_stream import RetryReadStream


class ShortReadFile:
    """Cap reads from a real file and optionally inject a shared transient error."""

    def __init__(self, file, limit, failures):
        self.file = file
        self.limit = limit
        self.failures = failures

    def read(self, size=-1):
        if self.file.tell() in self.failures:
            self.failures.remove(self.file.tell())
            raise OSError("transient read failure")
        return self.file.read(self.limit if size < 0 else min(size, self.limit))

    def __getattr__(self, name):
        return getattr(self.file, name)


class TestRetryReadStream(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.local_path = Path(self.directory.name) / "payload.bin"
        self.payload = bytes(range(128))
        self.local_path.write_bytes(self.payload)
        self.path = EPath(self.local_path)

    def short_reads(self, limit=7, failures=()):
        original_open = self.path.open
        pending = set(failures)
        return patch.object(
            EPath,
            "open",
            side_effect=lambda *args, **kwargs: ShortReadFile(
                original_open(*args, **kwargs), limit, pending
            ),
        )

    def test_sized_reads_preserve_every_chunk_and_do_not_overread(self):
        with self.short_reads(), RetryReadStream(self.path) as stream:
            self.assertEqual(stream.read(20), self.payload[:20])
            self.assertEqual(stream.tell(), 20)
            self.assertEqual(stream.read(9), self.payload[20:29])
            self.assertEqual(stream.tell(), 29)
            self.assertEqual(stream.read(0), b"")
            self.assertEqual(stream.tell(), 29)
            self.assertEqual(stream.read(), self.payload[29:])
            self.assertEqual(stream.tell(), len(self.payload))
            self.assertEqual(stream.read(5), b"")

    def test_more_than_ten_successful_short_reads_are_not_retry_exhaustion(self):
        with self.short_reads(limit=1), RetryReadStream(self.path) as stream:
            self.assertEqual(stream.read(), self.payload)
            self.assertEqual(stream.tell(), len(self.payload))

    def test_transient_read_failure_preserves_prefix_and_remaining_request(self):
        with self.short_reads(limit=7, failures=(14, 28)), RetryReadStream(self.path) as stream:
            self.assertEqual(stream.read(36), self.payload[:36])
            self.assertEqual(stream.tell(), 36)
            self.assertEqual(stream.read(200), self.payload[36:])
            self.assertEqual(stream.tell(), len(self.payload))

    def test_transient_open_failure_retries_without_a_handle(self):
        original_open = self.path.open
        calls = []

        def open_after_failure(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("transient open failure")
            return original_open(*args, **kwargs)

        with (
            patch.object(EPath, "open", side_effect=open_after_failure),
            RetryReadStream(self.path) as stream,
        ):
            self.assertEqual(stream.read(8), self.payload[:8])
        self.assertEqual(len(calls), 2)

    def test_persistent_io_failure_keeps_ten_attempt_bound(self):
        with patch.object(EPath, "open", side_effect=OSError("persistent failure")) as opened:
            with RetryReadStream(self.path) as stream:
                with self.assertRaisesRegex(OSError, "persistent failure"):
                    stream.read(5)
                self.assertEqual(stream.tell(), 0)
        self.assertEqual(opened.call_count, 10)

    def test_seeks_and_empty_files_keep_binary_stream_behavior(self):
        with self.short_reads(limit=5), RetryReadStream(self.path) as stream:
            self.assertEqual(stream.seek(9), 9)
            self.assertEqual(stream.read(13), self.payload[9:22])
            self.assertEqual(stream.seek(-3, 1), 19)
            self.assertEqual(stream.read(4), self.payload[19:23])
            self.assertEqual(stream.seek(-6, 2), 122)
            self.assertEqual(stream.read(20), self.payload[-6:])
        self.local_path.write_bytes(b"")
        with RetryReadStream(self.path) as stream:
            self.assertEqual(stream.read(), b"")
            self.assertEqual(stream.tell(), 0)

    def test_real_tar_members_survive_fragmented_reads_and_retries(self):
        members = {"sample.txt": b"training example" * 80, "other.bin": self.payload * 8}
        with tarfile.open(self.local_path, "w") as archive:
            for name, data in members.items():
                metadata = tarfile.TarInfo(name)
                metadata.size = len(data)
                archive.addfile(metadata, io.BytesIO(data))
        with self.short_reads(limit=31, failures=(62,)), RetryReadStream(self.path) as stream:
            with tarfile.open(fileobj=stream, mode="r:") as archive:
                for name, data in members.items():
                    self.assertEqual(archive.extractfile(name).read(), data)


if __name__ == "__main__":
    unittest.main()
