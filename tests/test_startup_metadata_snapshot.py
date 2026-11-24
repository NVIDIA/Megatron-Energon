# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import json
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
from megatron.energon.flavors.jsonl.ijsonl import IJsonlIndexReader
from tests import test_jsonl_dataset


class SnapshotTest(unittest.TestCase):
    def test_public_loader_order_and_resume_with_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "samples.jsonl"
            test_jsonl_dataset.TestJsonlDataset.create_text_test_dataset(
                source, range(64), range(64)
            )
            for num_workers, rank in ((0, 0), (0, 1), (2, 0), (2, 1)):
                with self.subTest(num_workers=num_workers, rank=rank):
                    worker = WorkerConfig(rank=rank, world_size=2, num_workers=num_workers)

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
                    original = dataset()
                    with use_metadata_snapshot(package, create=True):
                        captured = dataset()
                    with use_metadata_snapshot(package):
                        replayed = dataset()
                    with get_savable_loader(original) as baseline:
                        iterator = iter(baseline)
                        expected = [
                            (sample.idx, sample.text)
                            for sample in (next(iterator) for _ in range(16))
                        ]
                    with get_savable_loader(captured) as loader:
                        iterator = iter(loader)
                        self.assertEqual(
                            [
                                (sample.idx, sample.text)
                                for sample in (next(iterator) for _ in range(16))
                            ],
                            expected,
                        )
                    with get_savable_loader(replayed) as loader:
                        iterator = iter(loader)
                        prefix = [
                            (sample.idx, sample.text)
                            for sample in (next(iterator) for _ in range(5))
                        ]
                        state = loader.save_state_rank()
                        self.assertEqual(prefix, expected[:5])
                    with use_metadata_snapshot(package, create=False):
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
            samples = root / "samples.jsonl"
            test_jsonl_dataset.TestJsonlDataset.create_text_test_dataset(
                samples, range(4), range(4)
            )
            samples_path = EPath(samples)
            index_path = samples_path.with_suffix(".idx", replace=False)
            package = root / "metadata.json"
            with use_metadata_snapshot(package, create=True):
                self.assertTrue(path.is_file())
                self.assertFalse(missing.is_file())
                expected_size = path.size()
                self.assertEqual(path.read_text(), source.read_text())
                self.assertEqual(IJsonlIndexReader.size(samples_path), samples.stat().st_size)
            with use_metadata_snapshot(
                package, sha256=hashlib.sha256(package.read_bytes()).hexdigest()
            ):
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
                    self.assertEqual(IJsonlIndexReader.size(samples_path), samples.stat().st_size)
                    self.assertEqual(index_path.size(), (root / "samples.jsonl.idx").stat().st_size)
            source.write_text("new dataset revision\n")
            self.assertEqual(path.read_text(), "new dataset revision\n")

    def test_optional_checks_and_missing_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "metadata.json"
            with self.assertRaises(FileNotFoundError):
                with use_metadata_snapshot(package):
                    self.fail("Reading a missing snapshot must not create it")
            # Creation ignores a supplied checksum, even when it is incorrect.
            with use_metadata_snapshot(package, create=True, sha256="wrong"):
                pass
            digest = hashlib.sha256(package.read_bytes()).hexdigest()
            for checksum in (None, digest):
                with use_metadata_snapshot(package, sha256=checksum):
                    pass
            with self.assertRaisesRegex(ValueError, "SHA256"):
                with use_metadata_snapshot(package, sha256="wrong"):
                    self.fail("Checksum must be checked before construction")
            with self.assertRaisesRegex(ValueError, "identity"):
                MetadataSnapshot.load(package, sha256=digest, recipe_identity="wrong")
            # Existing applications can still distribute a decoded snapshot in memory.
            snapshot = MetadataSnapshot.load(package, sha256=digest, recipe_identity="")
            with use_metadata_snapshot(snapshot):
                pass
            for options in ({"create": True}, {"sha256": digest}):
                with self.assertRaisesRegex(ValueError, "file path"):
                    with use_metadata_snapshot(snapshot, **options):
                        self.fail("File-mode options need a path")
            document = json.loads(package.read_bytes())
            document["format_version"] = 999
            package.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "format"):
                with use_metadata_snapshot(package):
                    self.fail("Format validation is independent of checksum validation")

    def test_failed_creation_preserves_snapshot_and_nested_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recipe.yaml"
            source.write_text("old revision")
            path = EPath(source)
            package = root / "metadata.json"
            with use_metadata_snapshot(package, create=True):
                self.assertEqual(path.read_text(), "old revision")
                self.assertFalse(package.exists())
            original_package = package.read_bytes()
            source.write_text("new revision")
            with use_metadata_snapshot(package):
                for target in (package, root / "new-snapshot.json"):
                    with self.assertRaisesRegex(RuntimeError, "construction failed"):
                        with use_metadata_snapshot(target, create=True):
                            self.assertEqual(path.read_text(), "new revision")
                            raise RuntimeError("construction failed")
                    self.assertEqual(path.read_text(), "old revision")
            self.assertEqual(package.read_bytes(), original_package)
            self.assertFalse((root / "new-snapshot.json").exists())
            self.assertEqual(path.read_text(), "new revision")
            # A supplied checksum also cannot prevent replacing an existing package.
            with use_metadata_snapshot(package):
                with use_metadata_snapshot(package, create=True, sha256="wrong"):
                    self.assertEqual(path.read_text(), "new revision")
                self.assertEqual(path.read_text(), "old revision")
            with use_metadata_snapshot(package):
                self.assertEqual(path.read_text(), "new revision")
            self.assertEqual(
                sorted(p.name for p in root.iterdir()), ["metadata.json", "recipe.yaml"]
            )

    def test_uncaptured_reads_remain_live_and_capture_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "recipe.yaml"
            media = root / "image.bin"
            unseen = root / "unseen.yaml"
            metadata.write_text("first revision")
            media.write_bytes(b"first media")
            unseen.write_text("not captured")
            package = root / "metadata.json"
            with use_metadata_snapshot(package, create=True):
                EPath(metadata).read_text()
                with EPath(media).open("rb") as stream:
                    self.assertEqual(stream.read(), b"first media")
            original_package = package.read_bytes()
            media.write_bytes(b"second media")
            with use_metadata_snapshot(package):
                with EPath(media).open("rb") as stream:
                    self.assertEqual(stream.read(), b"second media")
                self.assertEqual(EPath(unseen).read_text(), "not captured")
                self.assertTrue(EPath(root).is_dir())
            self.assertEqual(package.read_bytes(), original_package)
            with patch("megatron.energon.epathlib.metadata_snapshot._MAX_CONTENT_BYTES", 4):
                with self.assertRaisesRegex(ValueError, "exceeded"):
                    with use_metadata_snapshot(package, create=True):
                        EPath(metadata).read_text()
            self.assertEqual(package.read_bytes(), original_package)


if __name__ == "__main__":
    unittest.main(verbosity=2)
