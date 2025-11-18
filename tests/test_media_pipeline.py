# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from megatron.energon.cache.file_store import SystemFileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from megatron.energon.media.extractor import (
    MediaFilterConfig,
    MediaFilterStrategy,
    extract_metadata,
)
from megatron.energon.media.filesystem_prepare import prepare_filesystem_dataset
from megatron.energon.media.metadata import (
    AVMetadata,
    ImageMetadata,
    MediaMetadataType,
    deserialize_media_metadata,
    serialize_media_metadata,
)


class MediaPipelineTests(unittest.TestCase):
    def _create_sample_image(self, path: Path, *, size: tuple[int, int] = (32, 16)) -> bytes:
        image = Image.new("RGB", size, color="blue")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        with path.open("wb") as handle:
            handle.write(buffer.getvalue())
        return buffer.getvalue()

    def test_av_metadata_roundtrip(self) -> None:
        metadata = AVMetadata(
            video_duration=1.5,
            video_num_frames=45,
            video_fps=30.0,
            video_width=1920,
            video_height=1080,
            audio_duration=1.5,
            audio_channels=2,
            audio_sample_rate=48000,
        )

        metadata_type, payload = serialize_media_metadata(metadata)

        self.assertIs(metadata_type, MediaMetadataType.AV)
        restored = deserialize_media_metadata(metadata_type, payload)
        self.assertEqual(restored, metadata)

    def test_image_metadata_roundtrip(self) -> None:
        metadata = ImageMetadata(width=64, height=32, format="PNG", mode="RGB")

        metadata_type, payload = serialize_media_metadata(metadata)
        self.assertIs(metadata_type, MediaMetadataType.IMAGE)
        restored = deserialize_media_metadata(metadata_type, payload)
        self.assertEqual(restored, metadata)

    def test_extract_metadata_extension_strategy(self) -> None:
        config = MediaFilterConfig.parse(None)
        image_bytes = self._create_in_memory_image()

        result = extract_metadata(image_bytes, config, filename="sample.png")
        self.assertIsNotNone(result)
        metadata_type, metadata = result
        self.assertEqual(metadata_type, MediaMetadataType.IMAGE)
        self.assertEqual(metadata.width, 16)
        self.assertEqual(metadata.height, 8)

    def test_extract_metadata_pattern_strategy(self) -> None:
        config = MediaFilterConfig.parse("*.png")
        image_bytes = self._create_in_memory_image()

        result = extract_metadata(image_bytes, config, filename="folder/sample.png")
        self.assertIsNotNone(result)

    def test_extract_metadata_type_strategy(self) -> None:
        config = MediaFilterConfig.parse(MediaFilterStrategy.TYPE.value)
        image_bytes = self._create_in_memory_image()

        result = extract_metadata(image_bytes, config, filename="ignored.bin")
        self.assertIsNotNone(result)
        metadata_type, _ = result
        self.assertEqual(metadata_type, MediaMetadataType.IMAGE)

    def test_prepare_filesystem_dataset_and_retrieve_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            image_path = tmp_root / "sample.jpg"
            self._create_sample_image(image_path)

            config = MediaFilterConfig.parse(None)
            stored = prepare_filesystem_dataset(EPath(tmp_root), config, progress=False)
            self.assertEqual(stored, 1)

            sqlite_path = tmp_root / MAIN_FOLDER_NAME / "index.sqlite"
            self.assertTrue(sqlite_path.is_file())

            with sqlite3.connect(sqlite_path) as connection:
                row = connection.execute(
                    "SELECT metadata_type, metadata_json FROM media_metadata WHERE entry_key = ?",
                    ("sample.jpg",),
                ).fetchone()
                filter_rows = connection.execute(
                    "SELECT strategy, pattern FROM media_filters"
                ).fetchall()

            self.assertIsNotNone(row)
            self.assertEqual(json.loads(row[1])["width"], 32)
            self.assertEqual(filter_rows, [(MediaFilterStrategy.EXT.value, None)])

            store = SystemFileStore(tmp_root)
            metadata = store.get_media_metadata("sample.jpg")
            self.assertIsInstance(metadata, ImageMetadata)
            self.assertEqual(metadata.width, 32)
            self.assertEqual(metadata.height, 16)

            image_path.unlink()

    def _create_in_memory_image(self) -> bytes:
        image = Image.new("RGB", (16, 8), color="red")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
