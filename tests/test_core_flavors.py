# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the dataset type detection of the core flavors."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import webdataset as wds

from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory
from megatron.energon.flavors.common.manifest.paths import MAIN_FOLDER_NAME
from megatron.energon.flavors.dataset_type import EnergonDatasetType, get_dataset_type
from megatron.energon.flavors.jsonl.crude_jsonl_dataset import CrudeJsonlDatasetFactory


def create_crude_webdataset(path: Path, num_samples: int) -> None:
    (path / "parts").mkdir(parents=True)
    with wds.ShardWriter(f"{path}/parts/data-%d.tar", maxcount=5) as shard_writer:
        for idx in range(num_samples):
            shard_writer.write(
                {"__key__": f"{idx:06d}", "txt": f"{idx}".encode(), "pkl": pickle.dumps(idx)}
            )
        total_shards = shard_writer.shard
    BaseWebdatasetFactory.prepare_dataset(
        path,
        [f"parts/data-{{0..{total_shards - 1}}}.tar"],
        split_parts_ratio=[("train", 1.0)],
        shuffle_seed=None,
        workers=1,
    )
    (path / MAIN_FOLDER_NAME / "dataset.yaml").write_text(
        "__module__: megatron.energon\n__class__: CrudeWebdataset\n"
    )


def create_jsonl_dataset(path: Path, num_samples: int) -> None:
    path.write_text("".join(json.dumps({"idx": idx}) + "\n" for idx in range(num_samples)))
    CrudeJsonlDatasetFactory.prepare_dataset(path)


class TestDatasetType(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_detects_dataset_types(self):
        create_crude_webdataset(self.root / "wds", 6)
        create_jsonl_dataset(self.root / "data.jsonl", 3)
        (self.root / "recipe.yaml").write_text("splits: {}\n")
        (self.root / "fs_only" / MAIN_FOLDER_NAME).mkdir(parents=True)
        (self.root / "fs_only" / MAIN_FOLDER_NAME / "index.sqlite").write_text("")
        (self.root / "empty_dir").mkdir()

        for name, expected in (
            ("wds", EnergonDatasetType.MANIFEST_DATASET),
            ("data.jsonl", EnergonDatasetType.JSONL),
            ("recipe.yaml", EnergonDatasetType.METADATASET),
            ("fs_only", EnergonDatasetType.FILESYSTEM),
            ("empty_dir", EnergonDatasetType.INVALID),
            ("missing", EnergonDatasetType.INVALID),
        ):
            with self.subTest(name=name):
                assert get_dataset_type(EPath(self.root / name)) == expected


if __name__ == "__main__":
    unittest.main()
