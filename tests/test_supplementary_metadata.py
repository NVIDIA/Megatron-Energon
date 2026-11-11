# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from megatron.energon.cache import supplementary_metadata as lookup
from megatron.energon.cache.file_store import SystemFileStore, WebdatasetFileStore


class SupplementaryMetadataTest(unittest.TestCase):
    def test_dimensions_payload_identity_and_read_only_fallback(self):
        originals = [cls.get_media_metadata for cls in (SystemFileStore, WebdatasetFileStore)]
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = root / "data"
                data.mkdir()
                (data / "image.bin").write_bytes(b"original media payload")
                prepared = root / "prepared"
                (prepared / "databases").mkdir(parents=True)
                database = prepared / "databases/complete.sqlite"
                with sqlite3.connect(database) as writer:
                    writer.execute(
                        "CREATE TABLE media_metadata (entry_key TEXT, metadata_type TEXT, metadata_json TEXT)"
                    )
                    writer.execute(
                        "INSERT INTO media_metadata VALUES (?, ?, ?)",
                        (
                            "prefix/image.bin",
                            "image",
                            json.dumps(dict(width=13, height=7, format="PNG", mode="RGB")),
                        ),
                    )
                before = hashlib.sha256(database.read_bytes()).hexdigest()
                (prepared / "ready-map.json").write_text(
                    json.dumps(
                        dict(
                            total_sources=1,
                            sources={str(data): dict(id="complete", key_prefix="prefix/")},
                        )
                    )
                )
                store = SystemFileStore(str(data))
                payload_before, source_before = store["image.bin"]
                lookup.install(prepared)
                metadata = store.get_media_metadata("image.bin")
                self.assertEqual((metadata.width, metadata.height), (13, 7))
                payload_after, source_after = store["image.bin"]
                self.assertEqual(payload_before, payload_after)
                self.assertEqual(source_before, source_after)
                with self.assertRaises(RuntimeError):
                    store.get_media_metadata("uncovered.bin")
                with self.assertRaises(RuntimeError):
                    lookup.install(prepared)
                self.assertEqual(before, hashlib.sha256(database.read_bytes()).hexdigest())
        finally:
            for cls, original in zip((SystemFileStore, WebdatasetFileStore), originals):
                cls.get_media_metadata = original
            for connection in getattr(lookup.STATE, "connections", {}).values():
                connection.close()
            lookup.STATE.connections = {}
