# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import io
import os
import random
import struct
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from multistorageclient.file import PosixFile

from megatron.energon.epathlib import EPath
from megatron.energon.epathlib.index_io import open_sized_index
from megatron.energon.flavors.jsonl import ijsonl
from megatron.energon.flavors.webdataset import itar


class IndexReaderReuseTest(unittest.TestCase):
    def test_random_offsets_lifetime_and_backend_fallback(self):
        counts = Counter()
        opened = []
        original_open, original_size = EPath.open, EPath.size

        def counted_open(self, *a, **kw):
            counts["opens"] += 1
            stream = original_open(self, *a, **kw)
            opened.append(stream)
            return stream

        def counted_size(self):
            counts["size_queries"] += 1
            return original_size(self)

        self.addCleanup(setattr, EPath, "open", original_open)
        self.addCleanup(setattr, EPath, "size", original_size)
        EPath.open, EPath.size = counted_open, counted_size

        rng = random.Random(20261111)
        with tempfile.TemporaryDirectory(prefix="owned-index-test-") as temporary:
            root = Path(temporary)
            values = [0]
            for _ in range(256):
                values.append(values[-1] + rng.randrange(1, 2000))
            blob = struct.pack("Q" * len(values), *values)
            tar_paths = [root / f"part-{i}.tar" for i in range(9)]
            for path in tar_paths:
                path.with_suffix(".tar.idx").write_bytes(blob)
            jsonl = root / "text.jsonl"
            jsonl.with_suffix(".jsonl.idx").write_bytes(blob)
            tar = itar.CachedItarOffsetReader(cache_size=3)
            js = ijsonl.CachedIJsonlOffsetReader(jsonl, cache_size=3)
            calls = 0
            # Interleave distant positions in one file, then exceed the file cache limit.
            requests = [(tar_paths[0], n) for n in (0, 100, 1, 101, 2, 102) * 10]
            requests += [(rng.choice(tar_paths), rng.randrange(len(values))) for _ in range(2500)]
            requests += [(tar_paths[0], len(values) - 1), (tar_paths[0], 0), (tar_paths[0], 1)]
            max_tar_entries = max_json_entries = 0
            for path, offset in requests:
                expected = (
                    values[offset],
                    values[offset + 1] - values[offset] if offset + 1 < len(values) else 0,
                )
                assert tar.get_itar_byte_offset(path, offset) == expected
                assert js.get_ijsonl_byte_offset(offset) == expected
                max_tar_entries = max(max_tar_entries, len(tar.tar_index_reader_cache))
                max_json_entries = max(max_json_entries, len(js.ijsonl_index_reader_cache))
                assert max_tar_entries <= 3 and max_json_entries <= 3
                assert sum(not stream.closed for stream in opened) <= 6
                calls += 2
            assert len(js) == len(values) - 1
            assert js.get_total_size() == values[-1]
            for entry in tar.tar_index_reader_cache.values():
                entry.tar_index_reader.close()
            tar.tar_index_reader_cache.clear()
            js.close()
            assert all(stream.closed for stream in opened)
            for cls, path in (
                (itar.TarIndexReader, tar_paths[0]),
                (ijsonl.IJsonlIndexReader, jsonl),
            ):
                with cls(path) as reader:
                    assert list(reader) == values
                    assert reader[0] == 0 and reader[len(values) - 1] == values[-1]
                    for bad in (-1, len(values), len(values) + 10):
                        try:
                            reader[bad]
                        except IndexError:
                            pass
                        else:
                            raise AssertionError("Missing index bounds check")

            class FakePath:
                def __init__(self, stream, size=24, error=None):
                    self.stream, self.reported_size, self.error = stream, size, error
                    self.queries = 0

                def open(self, *args, **kwargs):
                    return self.stream

                def size(self):
                    self.queries += 1
                    if self.error:
                        raise self.error
                    return self.reported_size

            # Object-backed streams can expose a dummy descriptor. Never fstat it.
            class FakeObjectFile(io.BytesIO):
                def fileno(self):
                    raise AssertionError("ObjectFile descriptor must not be inspected")

            fallback_checks = 0
            for stream in (io.BytesIO(b"abcdefgh"), FakeObjectFile(b"abcdefgh")):
                path = FakePath(stream)
                result, length = open_sized_index(path)
                assert result is stream and length == 3 and path.queries == 1
                assert stream.tell() == 0
                stream.close()
                fallback_checks += 1

            # Real MSC POSIX stream, but manifest/metadata-provider size is authoritative.
            stream = original_open(EPath(tar_paths[0].with_suffix(".tar.idx")), "rb")
            stream._storage_client = SimpleNamespace(_metadata_provider=object())
            path = FakePath(stream, size=16)
            result, length = open_sized_index(path)
            assert length == 2 and path.queries == 1
            stream.close()
            fallback_checks += 1

            # Failed size lookup must not leak a newly opened stream.
            stream = io.BytesIO(b"abcdefgh")
            try:
                open_sized_index(FakePath(stream, error=OSError("injected metadata failure")))
            except OSError:
                assert stream.closed
            else:
                raise AssertionError("Missing expected error")
            fallback_checks += 1

            # Plain POSIX sizing preserves the stream cursor and avoids full metadata.
            stream = original_open(EPath(tar_paths[0].with_suffix(".tar.idx")), "rb")
            assert type(stream) is PosixFile
            path = FakePath(stream, error=AssertionError("Unnecessary metadata lookup"))
            result, length = open_sized_index(path)
            assert length == len(values) and stream.tell() == 0 and path.queries == 0
            stream.close()
            fallback_checks += 1

            # A renamed path does not redirect sizing away from the opened descriptor.
            physical = root / "rename.idx"
            physical.write_bytes(blob)
            stream = original_open(EPath(physical), "rb")
            replacement = root / "replacement.idx"
            replacement.write_bytes(b"12345678")
            os.replace(replacement, physical)
            result, length = open_sized_index(FakePath(stream, size=8))
            assert length == len(values)
            stream.close()
            fallback_checks += 1
