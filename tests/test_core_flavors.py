# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the core flavor helpers: dataset type detection, deprecated import paths,
EPath.with_suffix and Cooker.part_filter."""

import gc
import importlib
import json
import pickle
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import webdataset as wds

from megatron.energon import (
    Cooker,
    DefaultTaskEncoder,
    TextSample,
    WorkerConfig,
    basic_sample_keys,
    get_loader,
    get_train_dataset,
    stateless,
)
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory
from megatron.energon.flavors.common.dataset_sampler import SliceState
from megatron.energon.flavors.common.manifest.paths import MAIN_FOLDER_NAME
from megatron.energon.flavors.common.manifest.types import ShardInfo
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


class TestDeprecatedImports(unittest.TestCase):
    def test_deprecated_module_resolves_to_new_object(self):
        parent_name = "megatron.energon.flavors.webdataset"
        module_name = f"{parent_name}.structs"
        parent = importlib.import_module(parent_name)
        # Deprecated modules only warn on their first import, import a fresh copy and restore the
        # previous module (also bound as attribute of the parent package) afterwards
        with (
            mock.patch.dict(sys.modules),
            mock.patch.dict(parent.__dict__),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            sys.modules.pop(module_name, None)
            old_shard_info = importlib.import_module(module_name).ShardInfo
        assert old_shard_info is ShardInfo
        messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any(m.startswith(f"{module_name} is deprecated") for m in messages), messages
        assert any(m.startswith(f"{module_name}.ShardInfo is deprecated") for m in messages), (
            messages
        )

    def test_deprecated_attribute_resolves_to_new_value(self):
        config = importlib.import_module("megatron.energon.flavors.webdataset.config")
        with self.assertWarnsRegex(DeprecationWarning, r"webdataset\.config\.MAIN_FOLDER_NAME"):
            main_folder_name = config.MAIN_FOLDER_NAME
        assert main_folder_name == MAIN_FOLDER_NAME

    def test_old_pickle_paths_load(self):
        # Pickles written before the restructuring reference the old module paths
        for module_name, name, expected in (
            ("megatron.energon.flavors.webdataset.structs", "ShardInfo", ShardInfo),
            ("megatron.energon.flavors.webdataset.sample_loader", "SliceState", SliceState),
        ):
            with self.subTest(module=module_name, name=name):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    cls = pickle.loads(f"c{module_name}\n{name}\n.".encode())
                assert cls is expected


class TestWithSuffix(unittest.TestCase):
    def test_replace_last_suffix(self):
        assert EPath("/data/train.tar.gz").with_suffix(".idx") == EPath("/data/train.tar.idx")
        assert EPath("/data/train.jsonl").with_suffix(".idx") == EPath("/data/train.idx")

    def test_append_suffix(self):
        assert EPath("/data/train.jsonl").with_suffix(".idx", replace=False) == EPath(
            "/data/train.jsonl.idx"
        )
        assert EPath("/data/train.tar.gz").with_suffix(".idx", replace=False) == EPath(
            "/data/train.tar.gz.idx"
        )


class TestCookerPartFilter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _cooked_part_keys(self, path: Path, part_filter) -> list[set[str]]:
        seen_parts = []

        @stateless
        def cook(sample: dict) -> TextSample:
            seen_parts.append({key for key in sample if not key.startswith("__")})
            return TextSample(**basic_sample_keys(sample), text="")

        class PartFilterTaskEncoder(DefaultTaskEncoder):
            cookers = [Cooker(cook, part_filter=part_filter)]

        loader = get_loader(
            get_train_dataset(
                path,
                batch_size=None,
                worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
                task_encoder=PartFilterTaskEncoder(),
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
            )
        )
        for _ in zip(range(4), loader):
            pass
        # Release the loader and its open files before the temporary dataset is removed
        del loader
        gc.collect()
        return seen_parts

    def test_webdataset_part_filter(self):
        create_crude_webdataset(self.root / "wds", 6)
        assert self._cooked_part_keys(self.root / "wds", None) == [{"txt", "pkl"}] * 4
        assert (
            self._cooked_part_keys(self.root / "wds", lambda part: part == "txt") == [{"txt"}] * 4
        )

    def test_jsonl_part_filter(self):
        create_jsonl_dataset(self.root / "data.jsonl", 6)
        assert self._cooked_part_keys(self.root / "data.jsonl", None) == [{"json"}] * 4
        assert self._cooked_part_keys(self.root / "data.jsonl", lambda part: False) == [set()] * 4


if __name__ == "__main__":
    unittest.main()
