# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from megatron.energon import WorkerConfig, get_savable_loader, get_train_dataset
from megatron.energon.epathlib import EPath
from megatron.energon.epathlib.metadata_snapshot import (
    MetadataSnapshot,
    use_metadata_snapshot,
)
from tests import test_jsonl_dataset
from tests.capture_metadata import capture_metadata


class SnapshotTest(unittest.TestCase):
    def test_public_loader_order_and_resume_with_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "samples.jsonl"
            test_jsonl_dataset.TestJsonlDataset.create_text_test_dataset(
                source, range(32), range(32)
            )
            for num_workers in (0, 2):
                with self.subTest(num_workers=num_workers):
                    worker = WorkerConfig(rank=0, world_size=1, num_workers=num_workers)

                    def dataset():
                        return get_train_dataset(
                            source,
                            worker_config=worker,
                            batch_size=None,
                            shuffle_buffer_size=None,
                            max_samples_per_sequence=None,
                            task_encoder=test_jsonl_dataset.SimpleCookingTaskEncoder(),
                        )

                    package = root / "metadata.json"
                    with capture_metadata(package, "recipe"):
                        original = dataset()
                    snapshot = MetadataSnapshot.load(
                        package,
                        sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
                        recipe_identity="recipe",
                    )
                    with use_metadata_snapshot(snapshot):
                        replayed = dataset()
                    with get_savable_loader(original) as baseline:
                        iterator = iter(baseline)
                        expected = [
                            (sample.idx, sample.text)
                            for sample in (next(iterator) for _ in range(16))
                        ]
                    with get_savable_loader(replayed) as loader:
                        iterator = iter(loader)
                        prefix = [
                            (sample.idx, sample.text)
                            for sample in (next(iterator) for _ in range(5))
                        ]
                        state = loader.save_state_rank()
                        self.assertEqual(prefix, expected[:5])
                    with use_metadata_snapshot(snapshot):
                        restored = dataset()
                    with get_savable_loader(restored) as loader:
                        loader.restore_state_rank(state)
                        iterator = iter(loader)
                        suffix = [
                            (sample.idx, sample.text)
                            for sample in (next(iterator) for _ in range(11))
                        ]
                        self.assertEqual(suffix, expected[5:])

    def test_capture_replay_without_storage_calls_and_scope_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recipe.yaml"
            source.write_text("dataset: αβ\n")
            path = EPath(source)
            missing = EPath(root / "absent.yaml")
            package = root / "metadata.json"
            with capture_metadata(package, "recipe-sha"):
                self.assertTrue(path.is_file())
                self.assertFalse(missing.is_file())
                expected_size = path.size()
                self.assertEqual(path.read_text(), source.read_text())
            snapshot = MetadataSnapshot.load(
                package,
                sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
                recipe_identity="recipe-sha",
            )
            with use_metadata_snapshot(snapshot):
                with (
                    patch.object(
                        type(path.fs),
                        "open",
                        side_effect=AssertionError("storage open"),
                    ),
                    patch.object(
                        type(path.fs),
                        "info",
                        side_effect=AssertionError("storage info"),
                    ),
                    patch.object(
                        type(path.fs),
                        "is_file",
                        side_effect=AssertionError("storage is_file"),
                    ),
                ):
                    self.assertTrue(path.is_file())
                    self.assertFalse(missing.is_file())
                    self.assertEqual(path.size(), expected_size)
                    with path.open("rb") as first, path.open("rb") as second:
                        self.assertEqual(first.read(3), b"dat")
                        self.assertEqual(second.read(), source.read_bytes())
                    self.assertEqual(path.read_text(), source.read_text())
            source.write_text("new dataset revision\n")
            self.assertEqual(path.read_text(), "new dataset revision\n")

    def test_digest_and_recipe_identity_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "metadata.json"
            with capture_metadata(package, "recipe-sha"):
                pass
            digest = hashlib.sha256(package.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "SHA256"):
                MetadataSnapshot.load(package, sha256="wrong", recipe_identity="recipe-sha")
            with self.assertRaisesRegex(ValueError, "identity"):
                MetadataSnapshot.load(package, sha256=digest, recipe_identity="wrong")


if __name__ == "__main__":
    unittest.main(verbosity=2)
