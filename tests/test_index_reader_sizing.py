# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import io
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from multistorageclient import StorageClient, StorageClientConfig
from multistorageclient.providers.posix_file import PosixFileStorageProvider
from multistorageclient.types import MetadataProvider, ResolvedPath, ResolvedPathState

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.jsonl.ijsonl import IJsonlIndexReader
from megatron.energon.flavors.webdataset.itar import TarIndexReader


class TestIndexReaderSizing(unittest.TestCase):
    def test_posix_sizes_the_open_file(self):
        offsets = (0, 42, 80)
        with tempfile.TemporaryDirectory() as directory:
            for suffix, reader_type in ((".tar", TarIndexReader), (".jsonl", IJsonlIndexReader)):
                with self.subTest(suffix=suffix):
                    data_path = Path(directory) / ("data" + suffix)
                    index_path = data_path.with_suffix(suffix + ".idx")
                    index_path.write_bytes(struct.pack("3Q", *offsets))
                    replacement = Path(directory) / "replacement.idx"
                    replacement.write_bytes(struct.pack("Q", 0))
                    original_open = EPath.open

                    def open_then_replace(path, mode):
                        stream = original_open(path, mode)
                        os.replace(replacement, index_path)
                        return stream

                    with (
                        patch.object(EPath, "open", open_then_replace),
                        patch.object(EPath, "size", side_effect=AssertionError("path size lookup")),
                        reader_type(data_path) as reader,
                    ):
                        self.assertEqual(len(reader), len(offsets))
                        self.assertEqual(reader[2], offsets[2])
                        self.assertEqual(reader[0], offsets[0])

    def test_composite_posix_uses_manifest_size(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "physical.idx").write_bytes(struct.pack("3Q", 0, 42, 80))
            manifest = Mock(spec=MetadataProvider)
            manifest.realpath.return_value = ResolvedPath(
                "physical.idx", ResolvedPathState.EXISTS, "disk"
            )
            manifest.get_object_metadata.return_value.content_length = 16
            disk = StorageClientConfig("disk", storage_provider=PosixFileStorageProvider(directory))
            composite = StorageClientConfig(
                "composite",
                storage_provider_profiles=["disk"],
                child_configs={"disk": disk},
                metadata_provider=manifest,
            )
            path = EPath(Path(directory) / "logical.tar")
            path.fs = StorageClient(composite)
            with TarIndexReader(path) as reader:
                self.assertEqual(len(reader), 2)
                self.assertEqual(reader[1], 42)
                with self.assertRaises(IndexError):
                    reader[2]

    def test_object_stream_uses_path_size(self):
        stream = io.BytesIO(struct.pack("3Q", 0, 42, 80))
        with (
            patch.object(EPath, "open", return_value=stream),
            patch.object(EPath, "size", return_value=16),
            IJsonlIndexReader("data.jsonl") as reader,
        ):
            self.assertEqual(len(reader), 2)
            self.assertEqual(reader[1], 42)
        self.assertTrue(stream.closed)

    def test_failed_size_lookup_closes_stream(self):
        stream = io.BytesIO()
        with (
            patch.object(EPath, "open", return_value=stream),
            patch.object(EPath, "size", side_effect=OSError("metadata unavailable")),
            self.assertRaisesRegex(OSError, "metadata unavailable"),
        ):
            TarIndexReader("data.tar")
        self.assertTrue(stream.closed)
