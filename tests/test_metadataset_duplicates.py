# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import tempfile
import unittest
import warnings
from copy import deepcopy
from pathlib import Path

import yaml

from megatron.energon import (
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    load_dataset,
)
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from megatron.energon.task_encoder.base import DefaultTaskEncoder
from tests import test_metadataset_v2 as fixtures


class TestDuplicateBlendDatasets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.worker = WorkerConfig(rank=0, world_size=1, num_workers=0)
        for name, start in (("ds1", 0), ("ds2", 100)):
            fixtures.TestDataset.create_text_test_dataset(
                self.root / name, range(start, start + 100), range(100)
            )

    def write_blend(self, name, entries, version=2, mode="blend"):
        path = self.root / name
        path.write_text(
            yaml.safe_dump(
                {
                    "__module__": "megatron.energon",
                    "__class__": "MetadatasetV2" if version == 2 else "Metadataset",
                    "splits": {
                        "train": {
                            mode if version == 2 else "datasets": [
                                deepcopy(entry) for entry in entries
                            ]
                        }
                    },
                }
            )
        )
        return path

    def load(self, path):
        return load_dataset(path).get_datasets(
            training=True, split_part="train", worker_config=self.worker
        )

    def test_nested_duplicates_preserve_combined_weights(self):
        for version in (1, 2):
            with self.subTest(version=version):
                self.write_blend(
                    "inner.yaml",
                    [
                        {"path": "./ds1", "weight": 1},
                        {"path": "./ds2", "weight": 3},
                    ],
                    version,
                )
                path = self.write_blend(
                    "outer.yaml",
                    [
                        {"path": "./ds1", "weight": 2},
                        {"path": "./inner.yaml", "weight": 4},
                    ],
                    version,
                )
                with warnings.catch_warnings(record=True) as caught:
                    result = self.load(path)
                self.assertEqual(len(result.datasets), 2)
                self.assertAlmostEqual(result.datasets[0].weight, 0.5)
                self.assertAlmostEqual(result.datasets[1].weight, 0.5)
                self.assertTrue(any("duplicate" in str(w.message) for w in caught))

    def test_nested_duplicate_has_one_sample_stream(self):
        self.write_blend("inner.yaml", [{"path": "./ds1"}])
        path = self.write_blend(
            "outer.yaml",
            [
                {"path": "./ds1"},
                {"path": "./inner.yaml"},
            ],
        )
        with warnings.catch_warnings(record=True) as caught:
            dataset = get_train_dataset(
                path,
                worker_config=self.worker,
                batch_size=1,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                task_encoder=DefaultTaskEncoder(),
            )
        loader = get_loader(dataset)
        iterator = iter(loader)
        values = [next(iterator).text[0] for _ in range(100)]
        self.assertEqual(len(set(values)), 100)
        self.assertTrue(any("duplicate" in str(w.message) for w in caught))
        del iterator, loader, dataset
        gc.collect()

    def test_distinct_leaf_settings_remain_separate(self):
        meta = self.root / "ds1" / MAIN_FOLDER_NAME
        (meta / "other-dataset.yaml").write_text((meta / "dataset.yaml").read_text())
        (meta / "other-split.yaml").write_text((meta / "split.yaml").read_text())
        split = yaml.safe_load((meta / "split.yaml").read_text())
        split["split_parts"]["val"] = list(split["split_parts"]["train"])
        (meta / "split.yaml").write_text(yaml.safe_dump(split))
        cases = [
            {"split_part": "val"},
            {"subset": {"range": [0, 50]}},
            {"subflavors": {"purpose": ["distinct", "training"]}},
            {"shuffle_over_epochs_multiplier": 2},
            {"dataset_config": "other-dataset.yaml"},
            {"split_config": "other-split.yaml"},
            {"aux": {"data": {"fs_path": "."}}},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                path = self.write_blend(
                    "distinct.yaml",
                    [
                        {"path": "./ds1"},
                        {"path": "./ds1", **overrides},
                    ],
                )
                with warnings.catch_warnings(record=True) as caught:
                    result = self.load(path)
                self.assertEqual(len(result.datasets), 2)
                self.assertFalse(any("duplicate" in str(w.message) for w in caught))

    def test_matching_auxiliary_data_merges(self):
        entry = {"path": "./ds1", "aux": {"data": {"fs_path": "."}}}
        path = self.write_blend("aux.yaml", [entry, entry])
        with self.assertWarnsRegex(UserWarning, "duplicate"):
            result = self.load(path)
        self.assertEqual(len(result.datasets), 1)
        self.assertEqual(result.datasets[0].weight, 1.0)
        self.assertEqual(result.datasets[0].aux["data"].get_path(), str(self.root))

    def test_epochized_repetitions_remain_separate(self):
        path = self.write_blend(
            "epochized.yaml",
            [
                {"path": "./ds1", "repetitions": 1},
                {"path": "./ds1", "repetitions": 2},
            ],
            mode="blend_epochized",
        )
        result = self.load(path)
        self.assertEqual(len(result.datasets), 2)
        self.assertEqual([d.repetitions for d in result.datasets], [1, 2])

    def test_merged_blend_save_restore_preserves_sequence(self):
        path = self.write_blend(
            "resume.yaml",
            [
                {"path": "./ds1"},
                {"path": "./ds2"},
                {"path": "./ds1"},
            ],
        )

        def make_loader():
            with self.assertWarnsRegex(UserWarning, "duplicate"):
                dataset = get_train_dataset(
                    path,
                    worker_config=self.worker,
                    batch_size=1,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                    task_encoder=DefaultTaskEncoder(),
                )
            return get_savable_loader(dataset)

        loader = make_loader()
        list(zip(range(20), loader))
        state = loader.save_state_rank()
        expected = [batch.text for _, batch in zip(range(40), loader)]
        restored = make_loader()
        restored.restore_state_rank(state)
        actual = [batch.text for _, batch in zip(range(40), restored)]
        self.assertEqual(actual, expected)
        del loader, restored
        gc.collect()
