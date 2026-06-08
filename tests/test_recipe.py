# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for recipes."""

import gc
import logging
import random
import sys
import tempfile
import time
import unittest
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable
from unittest.mock import patch

import torch
import webdataset as wds

from megatron.energon import (
    PackingGroupConfig,
    Sample,
    TextSample,
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_dataset,
    load_dataset,
)
from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.common.manifest.paths import MAIN_FOLDER_NAME
from megatron.energon.recipe.loader import prepare_recipe, traverse_recipe
from megatron.energon.recipe.loader_interface import DatasetBlendMode, LoadedDataset
from megatron.energon.task_encoder.base import DefaultTaskEncoder, PackingBufferSize, stateless
from megatron.energon.wrappers.blend_dataset import BlendDataset
from megatron.energon.wrappers.watchdog_dataset import WatchdogDataset
from tests.epath_s3_emulator import setup_s3_emulator

# Speed up tests significantly by reducing the torch status check interval for broken worker shutdown
try:
    torch.utils.data._utils.worker.MP_STATUS_CHECK_INTERVAL = 0.1
    torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 0.1
except AttributeError:
    pass


def _norng_state(state):
    if isinstance(state, bytes):
        if len(state) > 100:
            return state[:5] + f"...<len={len(state)}>".encode()
        return state
    elif isinstance(state, str):
        if len(state) > 100:
            return state[:5] + f"...<len={len(state)}>"
        return state
    elif isinstance(state, dict):
        return {k: _norng_state(v) for k, v in state.items()}
    elif isinstance(state, (list, tuple)):
        if len(state) > 100:
            state = state[:5]
        return type(state)(_norng_state(v) for v in state)
    else:
        return state


@edataclass
class TestJoinedSample(Sample):
    text1: torch.Tensor
    text2: torch.Tensor

    @staticmethod
    def from_joined(ds1: TextSample, ds2: TextSample) -> "TestJoinedSample":
        return TestJoinedSample.derive_from(
            ds1,
            text1=ds1.text,
            text2=ds2.text,
        )


def test_joiner(text1: TextSample, text2: TextSample) -> TestJoinedSample:
    return TestJoinedSample.derive_from(text1, text1=f"j{text1.text}", text2=f"j{text2.text}")


def get_blend_dataset(ds):
    if isinstance(ds, BlendDataset):
        return ds
    if hasattr(ds, "dataset"):
        return get_blend_dataset(ds.dataset)
    raise ValueError("No blend dataset found")


class SizeBlendTaskEncoder(DefaultTaskEncoder):
    blend_sample_size_alpha: float = 0.8
    blend_sample_size_epsilon: float = 0.7

    @stateless
    def blend_sample_size(self, sample: TextSample) -> int:
        return len(sample.text)


class TestDataset(unittest.TestCase):
    # Set up the test fixture
    def setUp(self):
        random.seed(42)

        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        warnings.simplefilter("ignore", ResourceWarning)

        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name)
        # self.dataset_path = Path("./test_dataset")

        self.dataset_path.mkdir(exist_ok=True, parents=True)

        # Create a small dummy datasets
        self.create_text_test_dataset(self.dataset_path / "ds1", range(55), range(55))
        self.create_text_test_dataset(self.dataset_path / "ds2", range(100, 155), range(100, 155))
        self.create_text_test_dataset(self.dataset_path / "ds3", range(200, 255), range(55))

        # Create a shuffled dataset for joining with the ds1. It has  overlap but includes more samples
        shuffled_range_100 = list(range(100))
        random.shuffle(shuffled_range_100)

        self.create_text_test_dataset(
            self.dataset_path / "ds1b", shuffled_range_100, shuffled_range_100, prefix="B"
        )

        shuffled_range_100 = list(range(100))
        random.shuffle(shuffled_range_100)
        self.create_text_test_dataset(
            self.dataset_path / "ds1c", shuffled_range_100, shuffled_range_100, prefix="C"
        )

        self.recipe_path = self.dataset_path / "recipe.yaml"
        with open(self.recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    blend:",
                        "      - weight: 1",
                        "        path: ds1",
                        "        subflavors:",
                        "          source: recipe.yaml",
                        "          number: 43",
                        "          recipe: recipe",
                        "        shuffle_over_epochs_multiplier: 3",
                        "      - weight: 1",
                        "        path: ds2",
                        "        subflavors:",
                        "          source: recipe.yaml",
                        "          number: 44",
                        "          recipe: recipe",
                        "  val:",
                        "    blend:",
                        "      - weight: 1",
                        "        path: ds1",
                        "        split_part: train",
                        "      - weight: 1",
                        "        path: ds2",
                        "        split_part: train",
                    ]
                )
            )
        self.nested_recipe_path = self.dataset_path / "nested_recipe.yaml"
        with open(self.nested_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    blend:",
                        "      - weight: 4",
                        "        path: ./recipe.yaml",
                        "        split_part: train",
                        "        subflavors:",
                        "          source: nested_recipe.yaml",
                        "          recipe: nested_train",
                        "      - path: ./recipe.yaml",
                        "        split_part: val",
                        "        subflavors:",
                        "          source: nested_recipe.yaml",
                        "          recipe: nested_val",
                    ]
                )
            )
        print(self.dataset_path)

    def tearDown(self):
        # Remove all temporary files
        gc.collect()
        self.temp_dir.cleanup()

    @staticmethod
    def create_text_test_dataset(
        path: Path, txt_range: Iterable[int], key_range: Iterable[int], prefix: str = ""
    ):
        """Creates a small dummy test dataset for testing purposes."""

        # Create num_samples unique captions
        (path / "parts").mkdir(exist_ok=True, parents=True)

        # Initialize the ShardWriter
        with wds.ShardWriter(f"{path}/parts/data-%d.tar", maxcount=10) as shard_writer:
            for key, txt in zip(key_range, txt_range):
                # Write individual files to shards
                shard_writer.write(
                    {
                        "__key__": f"{key:06d}",
                        "txt": f"{prefix}{txt}".encode(),
                    },
                )
            total_shards = shard_writer.shard

        from megatron.energon.flavors import BaseWebdatasetFactory

        BaseWebdatasetFactory.prepare_dataset(
            path,
            [f"parts/data-{{0..{total_shards - 1}}}.tar"],
            split_parts_ratio=[("train", 1.0)],
            shuffle_seed=None,
        )

        with open(path / MAIN_FOLDER_NAME / "dataset.yaml", "w") as f:
            f.write(
                "\n".join(
                    [
                        "sample_type:",
                        "  __module__: megatron.energon",
                        "  __class__: TextSample",
                        "field_map:",
                        "  text: txt",
                        "subflavors:",
                        "  source: dataset.yaml",
                        "  dataset.yaml: true",
                        "  number: 42",
                    ]
                )
            )

    @staticmethod
    def create_fixed_size_text_dataset(path: Path, key_range: Iterable[int], text_size: int):
        """Creates a text dataset where every sample has the same byte length."""
        (path / "parts").mkdir(exist_ok=True, parents=True)
        with wds.ShardWriter(f"{path}/parts/data-%d.tar", maxcount=10) as shard_writer:
            for key in key_range:
                shard_writer.write(
                    {
                        "__key__": f"{key:06d}",
                        "txt": ("x" * text_size).encode(),
                    },
                )
            total_shards = shard_writer.shard

        from megatron.energon.flavors import BaseWebdatasetFactory

        BaseWebdatasetFactory.prepare_dataset(
            path,
            [f"parts/data-{{0..{total_shards - 1}}}.tar"],
            split_parts_ratio=[("train", 1.0)],
            shuffle_seed=None,
        )

        with open(path / MAIN_FOLDER_NAME / "dataset.yaml", "w") as f:
            f.write(
                "\n".join(
                    [
                        "sample_type:",
                        "  __module__: megatron.energon",
                        "  __class__: TextSample",
                        "field_map:",
                        "  text: txt",
                    ]
                )
            )

    def test_recipe(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Train mode dataset
        train_dataset = get_train_dataset(
            self.recipe_path,
            worker_config=worker_config,
            batch_size=10,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
        )
        print(len(train_dataset))
        assert len(train_dataset) == 11

        train_loader1 = get_loader(train_dataset)

        train_order1 = [
            text for idx, data in zip(range(55 * 10), train_loader1) for text in data.text
        ]
        print(train_order1[:10])
        print(Counter(train_order1))
        assert len(Counter(train_order1)) == 110
        assert all(48 <= v <= 52 for v in Counter(train_order1).values())

    def test_deprecated_v2_aliases(self):
        from megatron.energon.recipe.recipe import Recipe

        with self.assertWarns(DeprecationWarning):
            from megatron.energon import MetadatasetV2
        with self.assertWarns(DeprecationWarning):
            from megatron.energon import prepare_metadataset
        with self.assertWarns(DeprecationWarning):
            from megatron.energon import traverse_metadataset
        with self.assertWarns(DeprecationWarning):
            from megatron.energon.flavors.dataset_type import is_metadataset
        from megatron.energon.flavors.dataset_type import is_recipe

        legacy_recipe_path = self.dataset_path / "legacy_recipe_alias.yaml"
        legacy_recipe_path.write_text(
            "\n".join(
                [
                    "__module__: megatron.energon",
                    "__class__: MetadatasetV2",
                    "splits:",
                    "  train:",
                    "    path: ds1",
                ]
            )
        )

        with self.assertWarns((DeprecationWarning, FutureWarning)):
            recipe = load_dataset(legacy_recipe_path)
        assert isinstance(recipe, Recipe)
        assert isinstance(recipe, MetadatasetV2)

        refs = traverse_metadataset(self.recipe_path, split_part="train")
        assert [ref.path.name for ref in refs] == ["ds1", "ds2"]
        assert is_metadataset(EPath(self.recipe_path)) == is_recipe(EPath(self.recipe_path))

        prepare_metadataset(EPath(self.recipe_path))

    def test_group(self):
        """Task-defined packing groups keep returned samples source-homogeneous."""
        recipe_path = self.dataset_path / "group_blend.yaml"
        with open(recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    blend:",
                        "      - weight: 1",
                        "        path: ds1",
                        "        subflavors:",
                        "          packing_source: ds1",
                        "      - weight: 1",
                        "        path: ds2",
                        "        subflavors:",
                        "          packing_source: ds2",
                    ]
                )
            )

        leaves = traverse_recipe(recipe_path, split_part="train")
        assert len(leaves) == 2
        assert {ref.subflavors["packing_source"] for ref in leaves} == {"ds1", "ds2"}

        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=0)
        loaded = load_dataset(recipe_path).get_datasets(
            training=True,
            split_part="train",
            worker_config=worker_config,
        )
        assert loaded.blend_mode == DatasetBlendMode.DATASET_WEIGHT
        assert {d.dataset.subflavors["packing_source"] for d in loaded.datasets} == {"ds1", "ds2"}

        class GroupIsolationEncoder(DefaultTaskEncoder):
            """Each returned packed sample must come from exactly one packing source."""

            def build_packing_groups(
                self,
                datasets: list[LoadedDataset],
                packing_buffer_size: PackingBufferSize,
                shuffle_buffer_size: int | None,
            ) -> list[PackingGroupConfig]:
                return [
                    PackingGroupConfig(
                        datasets=[
                            dataset
                            for dataset in datasets
                            if dataset.dataset.subflavors["packing_source"] == packing_source
                        ],
                        packing_buffer_size=packing_buffer_size,
                        shuffle_buffer_size=shuffle_buffer_size,
                    )
                    for packing_source in sorted(
                        {dataset.dataset.subflavors["packing_source"] for dataset in datasets}
                    )
                ]

            @stateless
            def encode_sample(self, sample: TextSample) -> TextSample:
                return sample

            def select_samples_to_pack(self, samples: list[TextSample]) -> list[list[TextSample]]:
                return [samples]

            @stateless
            def pack_selected_samples(self, samples: list[TextSample]) -> TextSample:
                packing_sources = {
                    sample.__subflavors__.get("packing_source")
                    for sample in samples
                    if sample.__subflavors__ is not None
                }
                assert len(packing_sources) == 1, (
                    "Mixed sources in one packed sample: "
                    f"{packing_sources} for keys {[sample.__key__ for sample in samples]}"
                )
                return TextSample.derive_from(
                    samples[0],
                    __key__=",".join(s.__key__ for s in samples),
                    __restore_key__=(),
                    text=f"{next(iter(packing_sources))}:" + " | ".join(s.text for s in samples),
                )

        torch.manual_seed(42)
        packed_ds = get_train_dataset(
            recipe_path,
            worker_config=worker_config,
            batch_size=2,
            packing_buffer_size=8,
            shuffle_buffer_size=8,
            max_samples_per_sequence=None,
            task_encoder=GroupIsolationEncoder(),
            virtual_epoch_length=10,
        )
        list(get_loader(packed_ds))

    def test_nested_recipe(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
        )

        dataset = load_dataset(self.nested_recipe_path)

        raw_datasets = dataset.get_datasets(
            training=False, split_part="train", worker_config=worker_config
        )
        assert raw_datasets.blend_mode == DatasetBlendMode.DATASET_WEIGHT
        assert [raw_dataset.weight for raw_dataset in raw_datasets.datasets] == [
            0.4,
            0.4,
            0.1,
            0.1,
        ], [raw_dataset.weight for raw_dataset in raw_datasets.datasets]
        assert [raw_dataset.dataset.paths[0].name for raw_dataset in raw_datasets.datasets] == [
            "ds1",
            "ds2",
            "ds1",
            "ds2",
        ]
        print([raw_dataset.dataset.subflavors for raw_dataset in raw_datasets.datasets])
        assert [raw_dataset.dataset.subflavors for raw_dataset in raw_datasets.datasets] == [
            {
                "source": "nested_recipe.yaml",
                "dataset.yaml": True,
                "number": 43,
                "recipe": "nested_train",
            },
            {
                "source": "nested_recipe.yaml",
                "dataset.yaml": True,
                "number": 44,
                "recipe": "nested_train",
            },
            {
                "source": "nested_recipe.yaml",
                "dataset.yaml": True,
                "number": 42,
                "recipe": "nested_val",
            },
            {
                "source": "nested_recipe.yaml",
                "dataset.yaml": True,
                "number": 42,
                "recipe": "nested_val",
            },
        ]

    def test_traverse_recipe_recurses_nested_v2_references(self):
        """Traversed subflavors only reflect recipe hierarchy merges.

        They intentionally do not include the leaf dataset's own `dataset.yaml` subflavors, which
        are only applied later when `get_datasets()` loads the concrete dataset factory.
        """

        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
        )
        refs = traverse_recipe(self.nested_recipe_path, split_part="train")
        dataset = load_dataset(self.nested_recipe_path)
        raw_datasets = dataset.get_datasets(
            training=False,
            split_part="train",
            worker_config=worker_config,
        )

        assert [ref.path for ref in refs] == [
            EPath(self.dataset_path / "ds1"),
            EPath(self.dataset_path / "ds2"),
            EPath(self.dataset_path / "ds1"),
            EPath(self.dataset_path / "ds2"),
        ]
        assert [ref.split_part for ref in refs] == ["train", "train", "train", "train"]
        assert all(ref.aux == {} for ref in refs)
        # Traversal records only hierarchy-derived subflavors, not the loaded leaf dataset.yaml.
        assert [ref.subflavors for ref in refs] == [
            {
                "source": "nested_recipe.yaml",
                "number": 43,
                "recipe": "nested_train",
            },
            {
                "source": "nested_recipe.yaml",
                "number": 44,
                "recipe": "nested_train",
            },
            {
                "source": "nested_recipe.yaml",
                "recipe": "nested_val",
            },
            {
                "source": "nested_recipe.yaml",
                "recipe": "nested_val",
            },
        ]

        for ref, raw_dataset in zip(refs, raw_datasets.datasets):
            for key, value in ref.subflavors.items():
                assert raw_dataset.dataset.subflavors[key] == value

    def test_traverse_recipe_preserves_missing_v2_leaf_and_aux(self):
        missing_leaf_recipe_path = self.dataset_path / "missing_leaf_recipe.yaml"
        missing_leaf_recipe_path.write_text(
            "\n".join(
                [
                    "__module__: megatron.energon",
                    "__class__: Recipe",
                    "splits:",
                    "  train:",
                    "    path: missing_ds",
                    "    subflavors:",
                    "      source: missing_leaf_recipe.yaml",
                    "      number: 42",
                    "      recipe: nested_val",
                    "    aux:",
                    "      labels: missing_aux",
                    "      media: filesystem://media",
                    "      blobs: byterange://byte_blobs",
                    "    shuffle_over_epochs_multiplier: 2",
                ]
            ),
            encoding="utf-8",
        )

        refs = traverse_recipe(missing_leaf_recipe_path, split_part="train")

        assert len(refs) == 1
        assert refs[0].path == EPath(self.dataset_path / "missing_ds")
        assert refs[0].split_part == "train"
        assert refs[0].aux == {
            "labels": EPath(self.dataset_path / "missing_aux"),
            "media": EPath(self.dataset_path / "media"),
            "blobs": EPath(self.dataset_path / "byte_blobs"),
        }
        assert refs[0].subflavors == {
            "source": "missing_leaf_recipe.yaml",
            "number": 42,
            "recipe": "nested_val",
        }
        assert refs[0].shuffle_over_epochs_multiplier == 2

    def test_joined_recipe(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Create a joined dataset configuration
        joined_recipe_path = self.dataset_path / "joined_recipe.yaml"
        with open(joined_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    join:",
                        "      ds1:",
                        "        path: ds1",
                        "        subflavors:",
                        "          source1: ds1",
                        "          number: 43",
                        "      ds2:",
                        "        path: ds3",
                        "        subflavors:",
                        "          source2: ds3",
                        "          number: 44",
                        "    joiner:",
                        f"      __module__: {TestJoinedSample.__module__}",
                        f"      __class__: {TestJoinedSample.__name__}",
                    ]
                )
            )
        prepare_recipe(EPath(joined_recipe_path))

        # Train mode dataset
        train_dataset = get_train_dataset(
            joined_recipe_path,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
        )
        print(len(train_dataset))
        assert len(train_dataset) == 55

        train_loader = get_savable_loader(
            train_dataset,
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )

        data = list(zip(range(2 * 55), train_loader))
        txt1_order = [data.text1[0] for idx, data in data]
        txt2_order = [data.text2[0] for idx, data in data]
        key_order = [data.__key__[0] for idx, data in data]
        # ds1 has 55 samples, key range 0:55, txt range 0:55
        # ds3 has 28 samples, key range 0:55, txt range 200:255
        # Joining results in: 0:55
        print("txt1:", txt1_order)
        # Joining results in: 200:255
        print("txt2:", txt2_order)
        # Joining results in: 0:55
        print("key:", key_order)
        # Check matching
        assert all(int(txt1) + 200 == int(txt2) for txt1, txt2 in zip(txt1_order, txt2_order))
        # Check frequency
        assert set(txt1_order) == set(str(i) for i in range(0, 55))
        assert set(txt2_order) == set(str(i) for i in range(200, 255))
        # Every item must occurr 2 times (2*55).
        assert Counter(txt1_order).most_common(1)[0][1] == 2

        state = train_loader.save_state_rank()

        # Iterate 60 more items
        data = list(zip(range(60), train_loader))
        txt1_order = [data.text1 for idx, data in data]
        txt2_order = [data.text2 for idx, data in data]
        key_order = [data.__key__ for idx, data in data]

        # Restore state
        train_loader = get_savable_loader(
            get_train_dataset(
                joined_recipe_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )

        train_loader.restore_state_rank(state)

        # Iterate 360 more items
        data = list(zip(range(60), train_loader))
        txt1_order_rest = [data.text1 for idx, data in data]
        txt2_order_rest = [data.text2 for idx, data in data]
        key_order_rest = [data.__key__ for idx, data in data]

        # Verify matching
        assert txt1_order == txt1_order_rest
        assert txt2_order == txt2_order_rest
        assert key_order == key_order_rest

    def test_joined_recipe_joiner(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Create a joined dataset configuration
        joined_recipe_path = self.dataset_path / "joined_recipe_joiner.yaml"
        with open(joined_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    blend:",
                        "      - weight: 1",
                        "        join:",
                        "          text1:",
                        "            path: ds1",
                        "            subflavors:",
                        "              source1: ds1",
                        "              number: 43",
                        "          text2:",
                        "            path: ds3",
                        "            subflavors:",
                        "              source2: ds3",
                        "              number: 44",
                        "        joiner:",
                        f"          __module__: {test_joiner.__module__}",
                        f"          __function__: {test_joiner.__name__}",
                    ]
                )
            )
        prepare_recipe(EPath(joined_recipe_path))

        # Train mode dataset
        train_dataset = get_train_dataset(
            joined_recipe_path,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
        )
        print(len(train_dataset))
        assert len(train_dataset) == 55

        train_loader = get_savable_loader(
            train_dataset,
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )

        data = list(zip(range(2 * 55), train_loader))
        txt1_order = [data.text1[0] for idx, data in data]
        txt2_order = [data.text2[0] for idx, data in data]
        key_order = [data.__key__[0] for idx, data in data]
        # ds1 has 55 samples, key range 0:55, txt range 0:55
        # ds3 has 28 samples, key range 0:55, txt range 200:255
        # Joining results in: 0:55, with prefix "j"
        print("txt1:", txt1_order)
        # Joining results in: 200:255, with prefix "j"
        print("txt2:", txt2_order)
        # Joining results in: 0:55
        print("key:", key_order)
        # Check matching
        assert all(
            int(txt1[1:]) + 200 == int(txt2[1:]) for txt1, txt2 in zip(txt1_order, txt2_order)
        )
        # Check frequency
        assert set(txt1_order) == set(f"j{i}" for i in range(0, 55))
        assert set(txt2_order) == set(f"j{i}" for i in range(200, 255))
        # Every item must occurr 2 times (2*55).
        assert Counter(txt1_order).most_common(1)[0][1] == 2

    def test_left_join(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Create a joined dataset configuration
        joined_recipe_path = self.dataset_path / "left_join.yaml"
        with open(joined_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    blend:",
                        "      - weight: 1",
                        "        join:",
                        "          text1:",
                        "            path: ds1",
                        "            subflavors:",
                        "              source1: ds1",
                        "              number: 43",
                        "          text2:",
                        "            path: ds1b",
                        "            nonmatch: skip",
                        "            subflavors:",
                        "              source2: ds1b",
                        "              number: 44",
                        "        joiner:",
                        f"          __module__: {test_joiner.__module__}",
                        f"          __function__: {test_joiner.__name__}",
                    ]
                )
            )
        prepare_recipe(EPath(joined_recipe_path))

        # Train mode dataset
        train_dataset = get_train_dataset(
            joined_recipe_path,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
        )
        print(len(train_dataset))
        assert len(train_dataset) == 55, len(train_dataset)

        train_loader = get_savable_loader(
            train_dataset,
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )

        data = list(zip(range(2 * 55), train_loader))
        txt1_order = [data.text1[0] for idx, data in data]
        txt2_order = [data.text2[0] for idx, data in data]
        key_order = [data.__key__[0] for idx, data in data]
        # ds1 has 55 samples, key range 0:55, txt range 0:55
        # ds3 has 28 samples, key range 0:55, txt range 200:255
        # Joining results in: 0:55, with prefix "j"
        print("txt1:", txt1_order)
        # Joining results in: 200:255, with prefix "j"
        print("txt2:", txt2_order)
        # Joining results in: 0:55
        print("key:", key_order)
        # Check matching
        assert all(int(txt1[1:]) == int(txt2[2:]) for txt1, txt2 in zip(txt1_order, txt2_order))
        # Check frequency
        assert set(txt1_order) == set(f"j{i}" for i in range(55))
        assert set(txt2_order) == set(f"jB{i}" for i in range(55))
        # Every item must occurr 2 times (2*55).
        assert Counter(txt1_order).most_common(1)[0][1] == 2

        # Test that changing the file works as expected
        with open(joined_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    blend:",
                        "      - weight: 1",
                        "        join:",
                        "          text1:",
                        "            path: ds1c",
                        "            subflavors:",
                        "              source1: ds1c",
                        "              number: 43",
                        "          text2:",
                        "            path: ds1b",
                        "            nonmatch: skip",
                        "            subflavors:",
                        "              source2: ds1b",
                        "              number: 44",
                        "        joiner:",
                        f"          __module__: {test_joiner.__module__}",
                        f"          __function__: {test_joiner.__name__}",
                        "      - weight: 1",
                        "        join:",
                        "          text1:",
                        "            path: ds1b",
                        "          text2:",
                        "            path: ds1",
                        "            nonmatch: skip",
                        "        joiner:",
                        f"          __module__: {test_joiner.__module__}",
                        f"          __function__: {test_joiner.__name__}",
                    ]
                )
            )

        # Expect this to fail. Preparation does not match!
        with self.assertRaises(Exception):
            # Train mode dataset
            train_dataset = get_train_dataset(
                joined_recipe_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
            )

        # Shall succeed after preparation
        prepare_recipe(EPath(joined_recipe_path))
        train_dataset = get_train_dataset(
            joined_recipe_path,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
        )
        # Check that there are no remainder files
        cache_folder = joined_recipe_path.with_name(joined_recipe_path.name + ".cache")
        assert sum(1 for f in cache_folder.iterdir() if f.is_file()) == 2, list(
            cache_folder.iterdir()
        )

    def test_left_join_exclude(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Create a joined dataset configuration
        orig_split_path = self.dataset_path / "ds1" / ".nv-meta" / "split.yaml"
        exclude_split_path = self.dataset_path / "ds1" / ".nv-meta" / "exclude_split.yaml"
        with open(exclude_split_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        orig_split_path.read_text(),
                        "exclude:",
                        ' - "parts/data-0.tar/000000"',
                        ' - "parts/data-0.tar/000001"',
                        ' - "parts/data-0.tar/000002"',
                        ' - "parts/data-0.tar/000003"',
                        ' - "parts/data-0.tar/000004"',
                        ' - "parts/data-1.tar"',
                        ' - "parts/data-2.tar/000029"',
                    ]
                )
            )

        # Create a joined dataset configuration
        joined_recipe_path = self.dataset_path / "left_join.yaml"
        with open(joined_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    blend:",
                        "      - weight: 1",
                        "        join:",
                        "          text1:",
                        "            path: ds1",
                        "            split_config: exclude_split.yaml",
                        "          text2:",
                        "            path: ds1b",
                        "            nonmatch: skip",
                        "        joiner:",
                        f"          __module__: {test_joiner.__module__}",
                        f"          __function__: {test_joiner.__name__}",
                    ]
                )
            )
        prepare_recipe(EPath(joined_recipe_path))

        # Train mode dataset
        train_dataset = get_train_dataset(
            joined_recipe_path,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
        )
        print(len(train_dataset))
        assert len(train_dataset) == 55 - 16, len(train_dataset)

        train_loader = get_savable_loader(
            train_dataset,
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )

        data = list(zip(range(2 * 55), train_loader))
        txt1_order = [data.text1[0] for idx, data in data]
        txt2_order = [data.text2[0] for idx, data in data]
        key_order = [data.__key__[0] for idx, data in data]
        # ds1 has 55 samples, key range 0:55, txt range 0:55
        # ds3 has 28 samples, key range 0:55, txt range 200:255
        # Joining results in: 0:55, with prefix "j"
        print("txt1:", txt1_order)
        # Joining results in: 200:255, with prefix "j"
        print("txt2:", txt2_order)
        # Joining results in: 0:55
        print("key:", key_order)
        # Check matching
        assert all(int(txt1[1:]) == int(txt2[2:]) for txt1, txt2 in zip(txt1_order, txt2_order))
        # Check frequency
        set_filtered_nums = set(range(5, 10)) | set(range(20, 29)) | set(range(30, 55))
        assert set(txt1_order) == set(f"j{i}" for i in set_filtered_nums)
        assert set(txt2_order) == set(f"jB{i}" for i in set_filtered_nums)

    def test_joined_recipe_prepare_mock(self):
        torch.manual_seed(42)

        # Create a joined dataset configuration
        joined_recipe_path = self.dataset_path / "joined_recipe_prepare_mock.yaml"
        with open(joined_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    join:",
                        "      - path: ds1",
                        "      - path: ds3",
                        "    joiner:",
                        "      __module__: __main__",
                        "      __class__: NonExistantSample",
                    ]
                )
            )
        prepare_recipe(EPath(joined_recipe_path))

        # Create a joined dataset configuration
        joined_recipe_path = self.dataset_path / "joined_recipe_prepare_mock2.yaml"
        with open(joined_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    join:",
                        "      - path: ds1",
                        "      - path: ds3",
                        "    joiner:",
                        "      __module__: non_existant_module",
                        "      __class__: MyCaptioningSample",
                    ]
                )
            )
        prepare_recipe(EPath(joined_recipe_path))

    def test_recipe_fixed_epochs(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Create a joined dataset configuration
        fixed_epochs_recipe_path = self.dataset_path / "recipe_fixed_epochs.yaml"
        with open(fixed_epochs_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    blend_epochized:",
                        "      - repetitions: 2",
                        "        path: ds1",
                        "        subflavors:",
                        "          source: ds1",
                        "          number: 43",
                        "      - repetitions: 3",
                        "        path: ds2",
                        "        subflavors:",
                        "          source: ds2",
                        "          number: 42",
                    ]
                )
            )

        # Train mode dataset
        train_dataset = get_train_dataset(
            fixed_epochs_recipe_path,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            repeat=False,
        )
        print(len(train_dataset))
        assert len(train_dataset) == 5 * 55, len(train_dataset)

        train_loader = get_savable_loader(
            train_dataset,
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )

        data = list(enumerate(train_loader))
        txt_order = [data.text[0] for idx, data in data]
        key_order = [data.__subflavors__[0]["source"] + "/" + data.__key__[0] for idx, data in data]
        print("txt1:", txt_order)
        print("key:", key_order)
        assert len(txt_order) == 5 * 55, Counter(txt_order)
        ds1_keys = [key for key in key_order if key.startswith("ds1/")]
        ds2_keys = [key for key in key_order if key.startswith("ds2/")]
        txt_cnt = Counter(txt_order)
        ds1_key_cnt = Counter(ds1_keys)
        ds2_key_cnt = Counter(ds2_keys)
        assert len(ds1_keys) == 2 * 55, (len(ds1_keys), ds1_key_cnt)
        assert len(ds2_keys) == 3 * 55, (len(ds2_keys), ds2_key_cnt)
        assert all(ds1_key_cnt[key] == 2 for key in ds1_keys)
        assert all(ds2_key_cnt[key] == 3 for key in ds2_keys)
        assert all(txt_cnt[key] in (2, 3) for key in txt_order)

        # Next epoch
        data = list(enumerate(train_loader))
        print([data.text[0] for idx, data in data])
        assert len(data) == 5 * 55, len(data)

        # Next epoch
        data1 = list(zip(range(3 * 55), train_loader))
        assert len(data1) == 3 * 55, len(data1)
        # Save state mid epoch
        state1 = train_loader.save_state_rank()
        print(state1)

        data2 = list(enumerate(train_loader))
        assert len(data2) == 2 * 55
        txt_order = [data.text[0] for idx, data in data1 + data2]
        key_order = [
            data.__subflavors__[0]["source"] + "/" + data.__key__[0] for idx, data in data1 + data2
        ]
        assert len(txt_order) == 5 * 55, Counter(txt_order)
        ds1_keys = [key for key in key_order if key.startswith("ds1/")]
        ds2_keys = [key for key in key_order if key.startswith("ds2/")]
        txt_cnt = Counter(txt_order)
        ds1_key_cnt = Counter(ds1_keys)
        ds2_key_cnt = Counter(ds2_keys)
        assert len(ds1_keys) == 2 * 55, (len(ds1_keys), ds1_key_cnt)
        assert len(ds2_keys) == 3 * 55, (len(ds2_keys), ds2_key_cnt)
        assert all(ds1_key_cnt[key] == 2 for key in ds1_keys)
        assert all(ds2_key_cnt[key] == 3 for key in ds2_keys)
        assert all(txt_cnt[key] in (2, 3) for key in txt_order)

        # Restore state
        train_loader = get_savable_loader(
            get_train_dataset(
                fixed_epochs_recipe_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )
        train_loader.restore_state_rank(state1)
        data2_restore = list(enumerate(train_loader))
        assert len(data2_restore) == 2 * 55
        txt_order_rst = [data.text[0] for idx, data in data1 + data2_restore]
        key_order_rst = [
            data.__subflavors__[0]["source"] + "/" + data.__key__[0]
            for idx, data in data1 + data2_restore
        ]
        assert len(txt_order_rst) == 5 * 55, Counter(txt_order_rst)
        assert txt_order_rst == txt_order
        assert key_order_rst == key_order
        ds1_keys_rst = [key for key in key_order_rst if key.startswith("ds1/")]
        ds2_keys_rst = [key for key in key_order_rst if key.startswith("ds2/")]
        txt_cnt_rst = Counter(txt_order_rst)
        ds1_key_cnt_rst = Counter(ds1_keys_rst)
        ds2_key_cnt_rst = Counter(ds2_keys_rst)
        assert len(ds1_keys_rst) == 2 * 55, (len(ds1_keys_rst), ds1_key_cnt_rst)
        assert len(ds2_keys_rst) == 3 * 55, (len(ds2_keys_rst), ds2_key_cnt_rst)
        assert all(ds1_key_cnt_rst[key] == 2 for key in ds1_keys_rst)
        assert all(ds2_key_cnt_rst[key] == 3 for key in ds2_keys_rst)
        assert all(txt_cnt_rst[key] in (2, 3) for key in txt_order_rst)

    def test_recipe_fixed_fractional_epochs(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Create a joined dataset configuration
        fixed_epochs_recipe_path = self.dataset_path / "recipe_fixed_epochs.yaml"
        with open(fixed_epochs_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    blend_epochized:",
                        "      - repetitions: 0.7",
                        "        path: ds1",
                        "        subflavors:",
                        "          source: ds1",
                        "          number: 43",
                        "      - repetitions: 1.5",
                        "        path: ds2",
                        "        subflavors:",
                        "          source: ds2",
                        "          number: 42",
                    ]
                )
            )

        # ===== Part 1: Verify fractions =====

        # Train mode dataset
        train_dataset = get_train_dataset(
            fixed_epochs_recipe_path,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=None,
            shuffle_over_epochs_multiplier=None,
            parallel_shard_iters=1,
            max_samples_per_sequence=None,
            repeat=False,
        )

        train_loader = get_savable_loader(
            train_dataset,
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )

        assert len(train_loader) == 38 + 55 + 27, len(train_loader)

        data = list(enumerate(train_loader))

        # Check the overall number of samples
        # Should be 0.7*len(ds1) + 1.5*len(ds2) = 0.7*55 + 1.5*55 = 38 + 55 + 27 (floor rounding)
        assert len(data) == 38 + 55 + 27, len(data)

        sample_counts = Counter([int(s[1].text[0]) for s in data])

        # The first 70% of samples from ds1 (0 to incl. 37) should be repeated only once
        assert all(sample_counts[sample] == 1 for sample in range(38))

        # Since ds2 is repeated 1.5 times, the first 50% of samples from ds2 (100 to incl. 126) should be repeated twice
        assert all(sample_counts[sample] == 2 for sample in range(100, 127))

        # The remaining samples from ds2 (127 to incl. 154) should be repeated only once
        assert all(sample_counts[sample] == 1 for sample in range(127, 155))

        # ===== Part 2: Save and restore state =====

        # Now let's check if the state is stored and restored correctly

        train_loader = get_savable_loader(
            get_train_dataset(
                fixed_epochs_recipe_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                shuffle_over_epochs_multiplier=None,
                parallel_shard_iters=1,
                max_samples_per_sequence=None,
                repeat=False,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )

        data1 = list(zip(range(95), train_loader))
        state1 = train_loader.save_state_rank()

        train_loader = get_savable_loader(
            get_train_dataset(
                fixed_epochs_recipe_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                shuffle_over_epochs_multiplier=None,
                parallel_shard_iters=1,
                max_samples_per_sequence=None,
                repeat=False,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )
        train_loader.restore_state_rank(state1)
        data2_restore = list(enumerate(train_loader))

        total_samples_save_restore = len(data1) + len(data2_restore)

        assert total_samples_save_restore == len(data), (
            "Total number of samples do not match when using save/restore"
        )

        sample_counts_save_restore = Counter(
            [int(s[1].text[0]) for d in [data1, data2_restore] for s in d]
        )

        assert sample_counts_save_restore == sample_counts, (
            "Sample counts do not match when using save/restore"
        )

        # ===== Part 3: Check if the state is restored correctly when saving right at the end of a dataset =====

        torch.manual_seed(42)

        train_loader = get_savable_loader(
            get_train_dataset(
                fixed_epochs_recipe_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                shuffle_over_epochs_multiplier=None,
                parallel_shard_iters=1,
                max_samples_per_sequence=None,
                repeat=False,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )

        ds1_counter = 0
        data1 = []
        for idx, sample in enumerate(train_loader):
            data1.append((idx, sample))
            if sample.__subflavors__[0]["source"] == "ds1":
                ds1_counter += 1
                if ds1_counter == 38:
                    # Stop right after the last sample from ds1
                    break

        state1 = train_loader.save_state_rank()

        train_loader = get_savable_loader(
            get_train_dataset(
                fixed_epochs_recipe_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                shuffle_over_epochs_multiplier=None,
                parallel_shard_iters=1,
                max_samples_per_sequence=None,
                repeat=False,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )
        train_loader.restore_state_rank(state1)
        data2_restore = list(enumerate(train_loader))

        total_samples_save_restore = len(data1) + len(data2_restore)

        assert total_samples_save_restore == len(data), (
            "Total number of samples do not match when using save/restore"
        )

        sample_counts_save_restore = Counter(
            [int(s[1].text[0]) for d in [data1, data2_restore] for s in d]
        )

        assert sample_counts_save_restore == sample_counts, (
            "Sample counts do not match when using save/restore"
        )

        # Try in repeat mode
        # Train mode dataset
        train_loader = get_savable_loader(
            get_train_dataset(
                fixed_epochs_recipe_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                shuffle_over_epochs_multiplier=None,
                parallel_shard_iters=1,
                max_samples_per_sequence=None,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )

        data = list(zip(range(200), train_loader))
        assert len(train_loader) == 38 + 55 + 27, len(train_loader)

        # Check the overall number of samples
        # Should be 0.7*len(ds1) + 1.5*len(ds2) = 38 + 55 + 27 (floor rounding)
        assert len(data) == 200, len(data)

        # ===== Part 4: Test count for multiple workers =====

        worker_config = WorkerConfig(
            rank=0,
            world_size=2,
            num_workers=2,
            seed_offset=42,
        )

        # Train mode dataset
        train_loader = get_savable_loader(
            get_train_dataset(
                fixed_epochs_recipe_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                shuffle_over_epochs_multiplier=None,
                parallel_shard_iters=1,
                max_samples_per_sequence=None,
                repeat=False,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )

        # TODO: This should be exactly 60. There is a corresponding TODO in the repeat_dataset.py
        assert len(train_loader) == 58, len(train_loader)

        data = list(enumerate(train_loader))

        # Check the overall number of samples
        # Should be 0.7*len(ds1)55 + 1.5*len(ds2)55 = 38 + 55 + 27 (floor rounding)
        # TODO: This should be exactly 60. There is a corresponding TODO in the repeat_dataset.py
        assert len(data) == 58, len(data)

    @patch.object(WatchdogDataset, "_watchdog_trigger")
    def test_watchdog_dataset(self, mock_watchdog_trigger):
        class TestTaskEncoder(DefaultTaskEncoder):
            def __init__(self):
                super().__init__()
                self.did_sleep = False

            def encode_sample(self, sample: TextSample) -> TextSample:
                if sample.text == "13":
                    import time

                    if not self.did_sleep:
                        print("Sleeping for 5 seconds on encode_sample to simulate stuck worker")
                        time.sleep(5)
                        self.did_sleep = True

                return sample

        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Train mode dataset
        train_dataset = get_train_dataset(
            self.recipe_path,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            task_encoder=TestTaskEncoder(),
        )

        train_loader = get_loader(
            train_dataset,
            watchdog_timeout_seconds=3,
            fail_on_timeout=False,
        )

        for idx, data in enumerate(train_loader):
            print(idx, data.text[0])
            if idx > 255:
                break

        mock_watchdog_trigger.assert_called()

    def test_worker_sample_balance(self):
        torch.manual_seed(42)

        for num_workers in [6, 30]:
            samples_per_global_worker = Counter()

            for rank in range(2):
                wc = WorkerConfig(
                    rank=rank,
                    world_size=2,
                    num_workers=num_workers,
                )

                train_dataset = get_train_dataset(
                    self.nested_recipe_path,
                    worker_config=wc,
                    batch_size=1,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                )

                blend_dataset = get_blend_dataset(train_dataset)
                assert isinstance(blend_dataset, BlendDataset)

                ds_weights = blend_dataset.dataset_weights
                assert len(ds_weights) == 4  # 4 datasets

                # We are now going to count the number of samples that was assigned to each
                # globally unique worker. This corresponds to the shard_ranges that energon
                # prints out when the dataset is built.

                for ds, w in ds_weights:
                    worker_slice_offsets = ds.dataset.dataset.workers_slice_offsets
                    assert len(worker_slice_offsets) == num_workers

                    for worker_idx, slice_offsets in enumerate(worker_slice_offsets):
                        samples_per_global_worker[(rank, worker_idx)] += (
                            slice_offsets[-1] - slice_offsets[0]
                        )
            print(samples_per_global_worker)

            # Check the sample assignnent is balanced across all global workers
            if num_workers == 6:
                assert list(samples_per_global_worker.values()) == [
                    19,  # rank 0
                    18,
                    18,
                    19,
                    18,
                    18,
                    19,  # rank 1
                    18,
                    18,
                    19,
                    18,
                    18,
                ]
            elif num_workers == 30:
                # This should match the pattern of the first 40 items of a generalized bit
                # reversal sequence of length 60.
                # Given 4 * 55 = 220 samples modulo 60 workers, is 40 remaining samples
                assert list(samples_per_global_worker.values()) == [
                    4,
                    4,
                    4,
                    4,
                    3,
                    4,
                    3,
                    4,
                    4,
                    4,
                    3,
                    4,
                    3,
                    4,
                    3,
                    4,
                    4,
                    4,
                    4,
                    3,
                    4,
                    3,
                    4,
                    4,
                    4,
                    3,
                    4,
                    3,
                    4,
                    3,
                    4,
                    4,
                    4,
                    4,
                    3,
                    4,
                    3,
                    4,
                    4,
                    4,
                    3,
                    4,
                    3,
                    4,
                    3,
                    4,
                    4,
                    4,
                    4,
                    3,
                    4,
                    3,
                    4,
                    4,
                    4,
                    3,
                    4,
                    3,
                    4,
                    3,
                ]

    def test_save_restore_state_train(self):
        torch.manual_seed(42)

        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        def new_loader():
            return get_savable_loader(
                get_train_dataset(
                    self.recipe_path,
                    worker_config=worker_config,
                    batch_size=10,
                    parallel_shard_iters=2,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                    shuffle_over_epochs_multiplier=2,
                ),
            )

        # Train mode dataset
        loader = new_loader()
        state_0 = loader.save_state_rank()
        order_0 = [data.text for idx, data in zip(range(10), loader)]
        state_1 = loader.save_state_rank()
        # print("save state done")
        order_1 = [data.text for idx, data in zip(range(20), loader)]

        state_2 = loader.save_state_rank()
        # print("save state done")
        # Iterated 30 samples, afterwards 50 samples. Checkpoint should be around that
        order_2 = [data.text for idx, data in zip(range(20), loader)]

        state_3 = loader.save_state_rank()
        # print("save state done")
        # Iterated 50 samples, afterwards 53 samples. Checkpoint should be around that
        order_3 = [data.text for idx, data in zip(range(3), loader)]

        state_4 = loader.save_state_rank()
        # print("save state done")
        # Dataset size is 55, want to save one sample before end of epoch
        # Iterated 53 samples, afterwards 54 samples. Checkpoint should be around that
        order_4 = [data.text for idx, data in zip(range(1), loader)]

        state_5 = loader.save_state_rank()
        # print("save state done")
        # Dataset size is 55, want to save one sample before end of epoch
        # Iterated 54 samples, afterwards 55 samples. Checkpoint should be around that
        order_5 = [data.text for idx, data in zip(range(1), loader)]

        state_6 = loader.save_state_rank()
        # print("save state done")
        # Dataset size is 55, want to save one sample before end of epoch
        # Iterated 55 samples, afterwards 75 samples. Checkpoint should be around that
        order_6 = [data.text for idx, data in zip(range(70), loader)]

        loader = new_loader()
        print("state_1:", _norng_state(state_1))
        loader.restore_state_rank(state_1)
        order_1_rest = [data.text for idx, data in zip(range(len(order_1)), loader)]
        assert order_1 == order_1_rest

        loader = new_loader()
        loader.restore_state_rank(state_0)
        order_0_rest = [data.text for idx, data in zip(range(len(order_0)), loader)]
        assert order_0 == order_0_rest

        loader = new_loader()
        print("state_2:", _norng_state(state_2))
        loader.restore_state_rank(state_2)
        order_2_rest = [data.text for idx, data in zip(range(len(order_2)), loader)]
        print("order_2:", order_2)
        print("order_2_rest:", order_2_rest)
        assert order_2 == order_2_rest

        loader = new_loader()
        print("state_3:", _norng_state(state_3))
        loader.restore_state_rank(state_3)
        order_3_rest = [data.text for idx, data in zip(range(len(order_3)), loader)]
        print("order_3:", order_3)
        print("order_3_rest:", order_3_rest)
        assert order_3 == order_3_rest

        loader = new_loader()
        print("state_4:", _norng_state(state_4))
        loader.restore_state_rank(state_4)
        order_4_rest = [data.text for idx, data in zip(range(len(order_4)), loader)]
        print("order_4:", order_4)
        print("order_4_rest:", order_4_rest)
        assert order_4 == order_4_rest

        loader = new_loader()
        print("state_5:", _norng_state(state_5))
        loader.restore_state_rank(state_5)
        order_5_rest = [data.text for idx, data in zip(range(len(order_5)), loader)]
        print("order_5:", order_5)
        print("order_5_rest:", order_5_rest)
        assert order_5 == order_5_rest

        loader = new_loader()
        print("state_6:", _norng_state(state_6))
        loader.restore_state_rank(state_6)
        order_6_rest = [data.text for idx, data in zip(range(len(order_6)), loader)]
        print("order_6:", order_6)
        print("order_6_rest:", order_6_rest)
        assert order_6 == order_6_rest

    def test_save_restore_state_train_workers(self):
        torch.manual_seed(42)

        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=1,
            seed_offset=42,
        )

        def new_loader():
            return get_savable_loader(
                get_train_dataset(
                    self.recipe_path,
                    worker_config=worker_config,
                    batch_size=10,
                    parallel_shard_iters=2,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                ),
                checkpoint_every_sec=0.5,
                checkpoint_every_min_n_samples=1,
            )

        # Train mode dataset
        loader = new_loader()
        state_0 = loader.save_state_rank()
        order_0 = [data.text for idx, data in zip(range(10), loader)]
        time.sleep(0.5)
        state_1 = loader.save_state_rank()
        # print("save state done")
        order_1 = [data.text for idx, data in zip(range(20), loader)]

        # Ensure a checkpoint is created on next()
        time.sleep(1.5)

        state_2 = loader.save_state_rank()
        # print("save state done")
        # Iterated 30 samples, afterwards 50 samples. Checkpoint should be around that
        order_2 = [data.text for idx, data in zip(range(20), loader)]

        state_3 = loader.save_state_rank()
        # print("save state done")
        # Iterated 50 samples, afterwards 53 samples. Checkpoint should be around that
        order_3 = [data.text for idx, data in zip(range(3), loader)]

        # Ensure a checkpoint is created on next()
        time.sleep(1.5)

        state_4 = loader.save_state_rank()
        # print("save state done")
        # Dataset size is 55, want to save one sample before end of epoch
        # Iterated 1 samples, afterwards 54 samples. Checkpoint should be around that
        order_4 = [data.text for idx, data in zip(range(1), loader)]

        # Ensure a checkpoint is created on next()
        time.sleep(1.5)

        state_5 = loader.save_state_rank()
        # print("save state done")
        # Dataset size is 55, want to save one sample before end of epoch
        # Iterated 1 samples, afterwards 55 samples. Checkpoint should be around that
        order_5 = [data.text for idx, data in zip(range(1), loader)]

        # Ensure a checkpoint is created on next()
        time.sleep(1.5)

        state_6 = loader.save_state_rank()
        # print("save state done")
        # Dataset size is 55, want to save one sample before end of epoch
        # Iterated 1 samples, afterwards 55 samples. Checkpoint should be around that
        order_6 = [data.text for idx, data in zip(range(10), loader)]

        loader = new_loader()
        print("state_1:", _norng_state(state_1))
        loader.restore_state_rank(state_1)
        order_1_rest = [data.text for idx, data in zip(range(len(order_1)), loader)]
        print("order_1:", order_1)
        print("order_1_rest:", order_1_rest)
        assert order_1 == order_1_rest

        loader = new_loader()
        loader.restore_state_rank(state_0)
        order_0_rest = [data.text for idx, data in zip(range(len(order_0)), loader)]
        assert order_0 == order_0_rest

        loader = new_loader()
        print("state_2:", _norng_state(state_2))
        loader.restore_state_rank(state_2)
        order_2_rest = [data.text for idx, data in zip(range(len(order_2)), loader)]
        print("order_2:", order_2)
        print("order_2_rest:", order_2_rest)
        assert order_2 == order_2_rest

        loader = new_loader()
        print("state_3:", _norng_state(state_3))
        loader.restore_state_rank(state_3)
        order_3_rest = [data.text for idx, data in zip(range(len(order_3)), loader)]
        print("order_3:", order_3)
        print("order_3_rest:", order_3_rest)
        assert order_3 == order_3_rest

        loader = new_loader()
        print("state_4:", _norng_state(state_4))
        loader.restore_state_rank(state_4)
        order_4_rest = [data.text for idx, data in zip(range(len(order_4)), loader)]
        print("order_4:", order_4)
        print("order_4_rest:", order_4_rest)
        assert order_4 == order_4_rest

        loader = new_loader()
        print("state_5:", _norng_state(state_5))
        loader.restore_state_rank(state_5)
        order_5_rest = [data.text for idx, data in zip(range(len(order_5)), loader)]
        print("order_5:", order_5)
        print("order_5_rest:", order_5_rest)
        assert order_5 == order_5_rest

        loader = new_loader()
        print("state_6:", _norng_state(state_6))
        loader.restore_state_rank(state_6)
        order_6_rest = [data.text for idx, data in zip(range(len(order_6)), loader)]
        print("order_6:", order_6)
        print("order_6_rest:", order_6_rest)
        assert order_6 == order_6_rest

    def test_save_restore_state_train_epochize_workers(self):
        torch.manual_seed(42)
        psi = 2
        vel = 19
        sbs = 10

        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=2,
            seed_offset=42,
        )

        # Train mode dataset
        torch.manual_seed(42)
        loader = get_savable_loader(
            get_train_dataset(
                self.recipe_path,
                worker_config=worker_config,
                batch_size=1,
                parallel_shard_iters=psi,
                virtual_epoch_length=vel,
                shuffle_buffer_size=sbs,
                max_samples_per_sequence=sbs,
            ),
        )
        state_0 = loader.save_state_rank()
        order_1 = [data.text[0] for data in loader]
        state_1 = loader.save_state_rank()
        order_2 = [data.text[0] for data in loader]
        state_2 = loader.save_state_rank()
        order_3 = [data.text[0] for idx, data in zip(range(17), loader)]

        torch.manual_seed(42)
        loader = get_savable_loader(
            get_train_dataset(
                self.recipe_path,
                worker_config=worker_config,
                batch_size=1,
                parallel_shard_iters=psi,
                virtual_epoch_length=vel,
                shuffle_buffer_size=sbs,
                max_samples_per_sequence=sbs,
            ),
        )
        print("state_0:", _norng_state(state_0))
        loader.restore_state_rank(state_0)
        order_5 = [data.text[0] for data in loader]
        print("order_1:", order_1)
        print("order_5:", order_5)
        assert order_1 == order_5

        torch.manual_seed(42)
        loader = get_savable_loader(
            get_train_dataset(
                self.recipe_path,
                worker_config=worker_config,
                batch_size=1,
                parallel_shard_iters=psi,
                virtual_epoch_length=vel,
                shuffle_buffer_size=sbs,
                max_samples_per_sequence=sbs,
            ),
        )
        print("state_1:", _norng_state(state_1))
        loader.restore_state_rank(state_1)
        order_6 = [data.text[0] for data in loader]
        print("order_2:", order_2)
        print("order_6:", order_6)
        assert order_2 == order_6

        torch.manual_seed(42)
        loader = get_savable_loader(
            get_train_dataset(
                self.recipe_path,
                worker_config=worker_config,
                batch_size=1,
                parallel_shard_iters=psi,
                virtual_epoch_length=vel,
                shuffle_buffer_size=sbs,
                max_samples_per_sequence=sbs,
            ),
        )
        print("state_2:", _norng_state(state_2))
        loader.restore_state_rank(state_2)
        order_7 = [data.text[0] for idx, data in zip(range(17), loader)]
        print("order_3:", order_3)
        print("order_7:", order_7)
        assert order_3 == order_7

    def test_save_restore_state_val(self):
        torch.manual_seed(42)

        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Train mode dataset
        loader = get_savable_loader(
            get_val_dataset(self.recipe_path, worker_config=worker_config, batch_size=10),
        )
        state_0 = loader.save_state_rank()
        order_1 = [data.text for idx, data in zip(range(55 * 20), loader)]
        state_1 = loader.save_state_rank()
        # print("save state done")
        order_2 = [data.text for idx, data in zip(range(55 * 20), loader)]

        loader = get_savable_loader(
            get_val_dataset(self.recipe_path, worker_config=worker_config, batch_size=10),
        )
        loader.restore_state_rank(state_1)
        order_3 = [data.text for idx, data in zip(range(55 * 20), loader)]
        assert order_2 == order_3

        loader = get_savable_loader(
            get_val_dataset(self.recipe_path, worker_config=worker_config, batch_size=10),
        )
        loader.restore_state_rank(state_0)
        order_4 = [data.text for idx, data in zip(range(55 * 20), loader)]
        assert order_1 == order_4

    def test_blending_randomness(self):
        import random

        import numpy

        for num_workers in [0, 1, 2]:  # Especially also check the num_workers=0 case
            world_size = 4
            micro_batch_size = 1
            seed = 42

            configs = (
                WorkerConfig(rank=0, world_size=world_size, num_workers=num_workers),
                WorkerConfig(rank=1, world_size=world_size, num_workers=num_workers),
                WorkerConfig(rank=2, world_size=world_size, num_workers=num_workers),
            )

            all_ranks_subflavors = []
            for rank_config in configs:
                torch.manual_seed(seed)
                numpy.random.seed(seed)
                random.seed(seed)

                ds = get_train_dataset(
                    self.recipe_path,
                    split_part="train",
                    worker_config=rank_config,
                    batch_size=micro_batch_size,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                )
                loader = get_loader(ds)

                subflavors = [
                    data.__subflavors__[0].get("number") for idx, data in zip(range(25), loader)
                ]

                all_ranks_subflavors.append(subflavors)

                print(f"Subflavors for rank {rank_config.rank}:", subflavors)

            # Assert that all ranks got different data
            for i in range(len(all_ranks_subflavors)):
                for j in range(i + 1, len(all_ranks_subflavors)):
                    assert all_ranks_subflavors[i] != all_ranks_subflavors[j], (
                        f"Rank {i} and rank {j} got the same subflavors."
                    )

            # Delete all locals, otherwise loaders might be kept alive
            locals().clear()
            gc.collect()

    def test_slice_iter_shuffle_over_epochs(self):
        torch.manual_seed(42)

        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        def new_loader():
            return get_savable_loader(
                get_train_dataset(
                    self.recipe_path,
                    worker_config=worker_config,
                    batch_size=10,
                    parallel_shard_iters=2,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                    shuffle_over_epochs_multiplier=-1,
                ),
            )

        # Train mode dataset
        loader = new_loader()
        _ = [data.text for idx, data in zip(range(1000), loader)]

    def test_save_restore_next(self):
        torch.manual_seed(42)

        wc = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=6,
        )

        initial_loader = get_savable_loader(
            get_train_dataset(
                self.nested_recipe_path,
                worker_config=wc,
                batch_size=1,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=0,
        )
        skip_initial = 9

        previous_cp = initial_loader.save_state_rank()
        print("initial_samples:")
        for i, sample in zip(range(skip_initial), initial_loader):
            print(f"sample[@{i}]: {sample.text}")
            print("previous_cp:", previous_cp)
            rst_loader = get_savable_loader(
                get_train_dataset(
                    self.nested_recipe_path,
                    worker_config=wc,
                    batch_size=1,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                ),
                checkpoint_every_sec=0,
                checkpoint_every_min_n_samples=0,
            )
            rst_loader.restore_state_rank(previous_cp)
            for i, rst_sample in zip(range(1), rst_loader):
                print(f"rst_sample[@{i}]: {rst_sample.text}")
            assert sample.text == rst_sample.text, f"{sample} != {rst_sample}"
            assert sample.__key__ == rst_sample.__key__, f"{sample} != {rst_sample}"
            assert sample.__restore_key__ == rst_sample.__restore_key__, f"{sample} != {rst_sample}"
            previous_cp = initial_loader.save_state_rank()

        # Iterate 10 samples, the save state and store the next 10 samples for reference.
        state_initial = initial_loader.save_state_rank()
        print("state_initial:", str(state_initial))
        initial_samples = [sample for _, sample in zip(range(20), initial_loader)]
        print(
            "initial_samples:"
            + "".join(
                f"\n [@{idx}] {sample.text}"
                for idx, sample in enumerate(initial_samples, start=skip_initial)
            )
        )

        del initial_loader
        gc.collect()

        second_loader = get_savable_loader(
            get_train_dataset(
                self.nested_recipe_path,
                worker_config=wc,
                batch_size=1,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=0,
        )
        second_loader.restore_state_rank(state_initial)

        # Save the state again, to check that it is the same as the just restored state
        same_state = second_loader.save_state_rank()
        print("same_state:", same_state)
        assert same_state == state_initial

        for offset in range(10):
            try:
                # Save state and restore in next loader
                state_offset = second_loader.save_state_rank()
                # Get 1 sample from the current loader
                samples = [sample for _, sample in zip(range(1), second_loader)]
                assert len(samples) == 1
                sample = samples[0]

                # Check that the sample is the same as the initial loader's reference sample
                print(f"sample[@{offset + skip_initial}]: {sample.text}")
                try:
                    assert sample.text == initial_samples[offset].text, (
                        f"{sample} != {initial_samples[offset]}"
                    )
                    assert sample.__key__ == initial_samples[offset].__key__, (
                        f"{sample} != {initial_samples[offset]}"
                    )
                    assert sample.__restore_key__ == initial_samples[offset].__restore_key__, (
                        f"{sample} != {initial_samples[offset]}"
                    )
                except Exception as e:
                    print(
                        "samples:"
                        + f"\n [@{offset + skip_initial}] {sample.text}"
                        + "".join(
                            f"\n [@{idx}] {sample.text}"
                            for idx, sample in zip(
                                range(skip_initial + offset + 1, skip_initial + offset + 6),
                                second_loader,
                            )
                        )
                    )
                    raise ValueError(f"Failed to iterate @{offset + skip_initial} samples") from e

                # Restore state in a new loader
                ref_loader = get_savable_loader(
                    get_train_dataset(
                        self.nested_recipe_path,
                        worker_config=wc,
                        batch_size=1,
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=None,
                    ),
                    checkpoint_every_sec=0,
                    checkpoint_every_min_n_samples=0,
                )
                ref_loader.restore_state_rank(state_offset)

                # Get 1 sample from the restored loader
                next_loader_samples = [sample for _, sample in zip(range(6), ref_loader)]
                assert len(next_loader_samples) == 6
                next_loader_sample = next_loader_samples[0]
                print(
                    "next_loader_samples:"
                    + f"\n [@{offset + skip_initial}] {sample.text}"
                    + "".join(
                        f"\n [@{idx}] {sample}"
                        for idx, sample in zip(
                            range(skip_initial + offset, skip_initial + offset + 6),
                            next_loader_samples,
                        )
                    )
                )
                assert next_loader_sample.text == sample.text, f"{next_loader_sample} != {sample}"
                assert next_loader_sample.__key__ == sample.__key__, (
                    f"{next_loader_sample} != {sample}"
                )
                assert next_loader_sample.__restore_key__ == sample.__restore_key__, (
                    f"{next_loader_sample} != {sample}"
                )
            except Exception as e:
                raise ValueError(f"Failed to iterate @{skip_initial}+{offset} samples") from e

    def test_dataset_absolute_nested_subset_fail(self):
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )
        ratio_recipe_path = self.dataset_path / "recipe_ratio.yaml"
        with open(ratio_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        # Absolute range on outer level should fail
                        "    subset: {range: [50, 55]}",
                        "    blend_epochized:",
                        "      - path: ds1",
                        "        subflavors:",
                        "          source: ds1",
                        "          number: 43",
                        "      - repetitions: 2",
                        "        path: ds2",
                        "        subflavors:",
                        "          source: ds2",
                        "          number: 42",
                    ]
                )
            )

        try:
            get_loader(
                get_train_dataset(
                    ratio_recipe_path,
                    worker_config=worker_config,
                    batch_size=1,
                    shuffle_buffer_size=None,
                    shuffle_over_epochs_multiplier=None,
                    parallel_shard_iters=1,
                    max_samples_per_sequence=None,
                    repeat=False,
                )
            )
            assert False, "Should have failed"
        except Exception as e:
            assert "only allowed for a leaf dataset" in str(
                e
            ) or "only use absolute subset ranges for a leaf dataset" in str(e), str(e)
            return

    def test_dataset_with_subset_end_keyword(self):
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )
        ratio_recipe_path = self.dataset_path / "recipe_ratio.yaml"
        with open(ratio_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        # Absolute range: [50, end]
                        # I.e. corresponds to sample range: [50, 55] (end is not included, so up to 54)
                        "    subset: {range: [50, end]}",
                        "    path: ds1",
                        "    subflavors:",
                        "      source: ds1",
                        "      number: 43",
                    ]
                )
            )

        loader = get_loader(
            get_train_dataset(
                ratio_recipe_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                shuffle_over_epochs_multiplier=None,
                parallel_shard_iters=1,
                max_samples_per_sequence=None,
                repeat=False,
            )
        )

        all_numbers = [int(s.text[0]) for s in loader]

        assert all_numbers == [50, 51, 52, 53, 54], "Subset range [50, end] should be [50, 55]"

    def test_dataset_with_subset_ratio(self):
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )
        ratio_recipe_path = self.dataset_path / "recipe_ratio.yaml"
        with open(ratio_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        # 20% of the dataset will be from ds1, 80% from ds2
                        # I.e. sample range: [0.2*55, 0.8*55] = [11, 44]
                        "    subset: {range: [20%, 80%]}",
                        "    blend_epochized:",
                        "      - path: ds1",
                        "        subflavors:",
                        "          source: ds1",
                        "          number: 43",
                        "      - repetitions: 2",
                        "        path: ds2",
                        "        subflavors:",
                        "          source: ds2",
                        "          number: 42",
                    ]
                )
            )

        loader = get_loader(
            get_train_dataset(
                ratio_recipe_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                shuffle_over_epochs_multiplier=None,
                parallel_shard_iters=1,
                max_samples_per_sequence=None,
                repeat=False,
            )
        )

        data = list(enumerate(loader))
        assert len(data) == 33 + 33 * 2, len(data)

        sample_counts = Counter([int(s[1].text[0]) for s in data])
        assert all(sample_counts[sample] == 0 for sample in range(11)), sample_counts
        assert all(sample_counts[sample] == 1 for sample in range(11, 44)), sample_counts
        assert all(sample_counts[sample] == 0 for sample in range(44, 55)), sample_counts
        assert all(sample_counts[sample] == 0 for sample in range(100, 111)), sample_counts
        assert all(sample_counts[sample] == 2 for sample in range(111, 144)), sample_counts
        assert all(sample_counts[sample] == 0 for sample in range(144, 155)), sample_counts
        assert sample_counts.total() == 33 + 33 * 2, sample_counts.total()

        # Combine with subset_samples

        ratio2_recipe_path = self.dataset_path / "recipe_ratio2.yaml"
        with open(ratio2_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        # take [10, 30] from ds1, [20, 40] from ds2 and then only [20%, 80%]
                        # I.e. sample range: [14, 26], 2 * [124, 136]
                        "    subset: {range: [20%, 80%]}",
                        "    blend_epochized:",
                        "      - path: ds1",
                        "        subset: {range: [10, 30]}",
                        "        subflavors:",
                        "          source: ds1",
                        "          number: 43",
                        "      - repetitions: 2",
                        "        subset: {range: [20, 40]}",
                        "        path: ds2",
                        "        subflavors:",
                        "          source: ds2",
                        "          number: 42",
                    ]
                )
            )

        loader = get_loader(
            get_train_dataset(
                ratio2_recipe_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                shuffle_over_epochs_multiplier=None,
                parallel_shard_iters=1,
                max_samples_per_sequence=None,
                repeat=False,
            )
        )

        data = list(enumerate(loader))
        assert len(data) == 12 + 12 * 2, len(data)

        sample_counts = Counter([int(s[1].text[0]) for s in data])
        assert all(sample_counts[sample] == 0 for sample in range(14)), sample_counts
        assert all(sample_counts[sample] == 1 for sample in range(14, 26)), sample_counts
        assert all(sample_counts[sample] == 0 for sample in range(26, 55)), sample_counts
        assert all(sample_counts[sample] == 0 for sample in range(100, 124)), sample_counts
        assert all(sample_counts[sample] == 2 for sample in range(124, 136)), sample_counts
        assert all(sample_counts[sample] == 0 for sample in range(136, 155)), sample_counts
        assert sample_counts.total() == 12 + 12 * 2, sample_counts.total()

        # Combine with subset_ratio and subset_samples and nested recipe
        nested_recipe_path = self.dataset_path / "recipe_nested_subset.yaml"
        with open(nested_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    subset: {range: [0%, 50%]}",
                        "    blend_epochized:",
                        "      - path: ds3",
                        # take [30, 50] from ds3, then first 50%, resulting in samples [230, 240]
                        "        subset: {range: [30, 50]}",
                        "        subflavors:",
                        "          source: ds3",
                        "          number: 45",
                        "      - repetitions: 2",
                        # Inner sample range: [14, 26], 2 * [124, 136], total=12*3=36
                        # Applying subset ratio 25%-75%: [17, 23], 2*[127, 133], total=3*6=18
                        # Applying outer 50%: [17, 20], 2*[127, 130], total=3*3=9
                        # Applying repetition: 2*[17, 20], 4*[127, 130], total=2*9=18
                        "        subset: {range: [25%, 75%]}",
                        "        path: recipe_ratio2.yaml",
                    ]
                )
            )

        loader = get_loader(
            get_train_dataset(
                nested_recipe_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                shuffle_over_epochs_multiplier=None,
                parallel_shard_iters=1,
                max_samples_per_sequence=None,
                repeat=False,
            )
        )

        data = list(enumerate(loader))
        assert len(data) == 10 + 9 * 2, len(data)
        sample_counts = Counter([int(s[1].text[0]) for s in data])
        assert all(sample_counts[sample] == 0 for sample in range(17)), sample_counts
        assert all(sample_counts[sample] == 2 for sample in range(17, 20)), sample_counts
        assert all(sample_counts[sample] == 0 for sample in range(20, 55)), sample_counts
        assert all(sample_counts[sample] == 0 for sample in range(100, 127)), sample_counts
        assert all(sample_counts[sample] == 4 for sample in range(127, 130)), sample_counts
        assert all(sample_counts[sample] == 0 for sample in range(130, 155)), sample_counts
        assert all(sample_counts[sample] == 0 for sample in range(200, 230)), sample_counts
        assert all(sample_counts[sample] == 1 for sample in range(230, 240)), sample_counts
        assert all(sample_counts[sample] == 0 for sample in range(240, 255)), sample_counts
        assert sample_counts.total() == 10 + 9 * 2, sample_counts.total()

    def test_blend_sample_size_based_distribution(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        short_size = 2
        long_size = 50
        self.create_fixed_size_text_dataset(
            self.dataset_path / "ds_short", range(55), text_size=short_size
        )
        self.create_fixed_size_text_dataset(
            self.dataset_path / "ds_long", range(100, 155), text_size=long_size
        )

        size_blend_recipe_path = self.dataset_path / "recipe_size_blend.yaml"
        with open(size_blend_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    blend:",
                        "      - weight: 1",
                        "        path: ds_short",
                        "        subflavors:",
                        "          source: ds_short",
                        "      - weight: 1",
                        "        path: ds_long",
                        "        subflavors:",
                        "          source: ds_long",
                    ]
                )
            )

        def load_samples(task_encoder: DefaultTaskEncoder, n_samples: int):
            loader = get_loader(
                get_train_dataset(
                    size_blend_recipe_path,
                    worker_config=worker_config,
                    batch_size=None,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                    task_encoder=task_encoder,
                )
            )
            return list(zip(range(n_samples), loader))

        n_samples = 2000
        default_samples = load_samples(DefaultTaskEncoder(), n_samples)
        size_samples = load_samples(SizeBlendTaskEncoder(), n_samples)

        def tally(samples):
            sample_counts: Counter[str] = Counter()
            size_totals: Counter[str] = Counter()
            for _, sample in samples:
                source = sample.__subflavors__["source"]
                sample_counts[source] += 1
                size_totals[source] += len(sample.text)
            return sample_counts, size_totals

        default_counts, default_sizes = tally(default_samples)
        size_counts, size_totals = tally(size_samples)

        default_sample_ratio = default_counts["ds_short"] / default_counts.total()
        size_sample_ratio = size_counts["ds_short"] / size_counts.total()
        default_size_ratio = default_sizes["ds_short"] / default_sizes.total()
        size_size_ratio = size_totals["ds_short"] / size_totals.total()
        assert 0.45 <= default_sample_ratio <= 0.55, default_counts
        assert default_size_ratio < 0.15, (default_sizes, default_size_ratio)
        assert 0.45 <= size_size_ratio <= 0.55, size_totals
        assert size_sample_ratio > 0.85, size_counts
        assert size_sample_ratio > default_sample_ratio + 0.15, (
            size_sample_ratio,
            default_sample_ratio,
        )

    def test_blend_sample_size_save_restore(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        short_size = 2
        long_size = 50
        self.create_fixed_size_text_dataset(
            self.dataset_path / "ds_short", range(55), text_size=short_size
        )
        self.create_fixed_size_text_dataset(
            self.dataset_path / "ds_long", range(100, 155), text_size=long_size
        )

        size_blend_recipe_path = self.dataset_path / "recipe_size_blend.yaml"
        with open(size_blend_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    blend:",
                        "      - weight: 1",
                        "        path: ds_short",
                        "        subflavors:",
                        "          source: ds_short",
                        "      - weight: 1",
                        "        path: ds_long",
                        "        subflavors:",
                        "          source: ds_long",
                    ]
                )
            )

        def new_loader():
            return get_savable_loader(
                get_train_dataset(
                    size_blend_recipe_path,
                    worker_config=worker_config,
                    batch_size=None,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                    task_encoder=SizeBlendTaskEncoder(),
                ),
                checkpoint_every_sec=0,
                checkpoint_every_min_n_samples=1,
            )

        loader = new_loader()
        list(zip(range(20), loader))
        state = loader.save_state_rank()
        order_1 = [sample.text for _, sample in zip(range(30), loader)]

        loader = new_loader()
        loader.restore_state_rank(state)
        order_1_rest = [sample.text for _, sample in zip(range(len(order_1)), loader)]
        assert order_1 == order_1_rest

        blend_dataset = get_blend_dataset(loader.dataset.dataset)
        assert isinstance(blend_dataset, BlendDataset)
        assert blend_dataset.sample_size_fn is not None
        assert sum(blend_dataset._emitted_sizes) > 0

    def test_s3(self):
        # Create a joined dataset configuration
        mixed_recipe_path = self.dataset_path / "recipe_mixed.yaml"
        with open(mixed_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    path: msc://s3test_recipe/test/dataset/nested_recipe.yaml",
                    ]
                )
            )

        with setup_s3_emulator(profile_name="s3test_recipe") as emu:
            # Upload the dataset to the S3 emulator
            # EPath(self.dataset_path).copy(EPath("msc://s3/test/dataset"))
            emu.add_file(self.dataset_path, "test/dataset")

            train_dataset = get_loader(
                get_train_dataset(
                    mixed_recipe_path,
                    worker_config=WorkerConfig(
                        rank=0,
                        world_size=1,
                        num_workers=2,
                    ),
                    batch_size=1,
                    shuffle_buffer_size=10,
                    max_samples_per_sequence=None,
                    virtual_epoch_length=10,
                )
            )

            data = list(enumerate(train_dataset))
            assert len(data) == 10, len(data)


if __name__ == "__main__":
    unittest.main()
