# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for meta datasets."""

import gc
import logging
import random
import sys
import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable
from unittest.mock import patch

import torch
import webdataset as wds

from megatron.energon import (
    Sample,
    TextSample,
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    load_dataset,
)
from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from megatron.energon.metadataset.loader import prepare_metadataset
from megatron.energon.metadataset.loader_interface import DatasetBlendMode
from megatron.energon.task_encoder.base import DefaultTaskEncoder
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

        self.mds_path = self.dataset_path / "metadataset_v2.yaml"
        with open(self.mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: MetadatasetV2",
                        "splits:",
                        "  train:",
                        "    blend:",
                        "      - weight: 1",
                        "        path: ds1",
                        "        subflavors:",
                        "          source: metadataset_v2.yaml",
                        "          number: 43",
                        "          mds: mds",
                        "        shuffle_over_epochs_multiplier: 3",
                        "      - weight: 1",
                        "        path: ds2",
                        "        subflavors:",
                        "          source: metadataset_v2.yaml",
                        "          number: 44",
                        "          mds: mds",
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
        self.nested_mds_path = self.dataset_path / "nested_metadataset_v2.yaml"
        with open(self.nested_mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: MetadatasetV2",
                        "splits:",
                        "  train:",
                        "    blend:",
                        "      - weight: 4",
                        "        path: ./metadataset_v2.yaml",
                        "        split_part: train",
                        "        subflavors:",
                        "          source: nested_metadataset.yaml",
                        "          mds: nested_train",
                        "      - path: ./metadataset_v2.yaml",
                        "        split_part: val",
                        "        subflavors:",
                        "          source: nested_metadataset.yaml",
                        "          mds: nested_val",
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

    def test_metadataset(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Train mode dataset
        train_dataset = get_train_dataset(
            self.mds_path,
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

    def test_nested_metadataset(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
        )

        dataset = load_dataset(self.nested_mds_path)

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
                "source": "nested_metadataset.yaml",
                "dataset.yaml": True,
                "number": 43,
                "mds": "nested_train",
            },
            {
                "source": "nested_metadataset.yaml",
                "dataset.yaml": True,
                "number": 44,
                "mds": "nested_train",
            },
            {
                "source": "nested_metadataset.yaml",
                "dataset.yaml": True,
                "number": 42,
                "mds": "nested_val",
            },
            {
                "source": "nested_metadataset.yaml",
                "dataset.yaml": True,
                "number": 42,
                "mds": "nested_val",
            },
        ]

    def test_joined_metadataset(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Create a joined dataset configuration
        joined_mds_path = self.dataset_path / "joined_metadataset_v2.yaml"
        with open(joined_mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: MetadatasetV2",
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
        prepare_metadataset(EPath(joined_mds_path))

        # Train mode dataset
        train_dataset = get_train_dataset(
            joined_mds_path,
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
                joined_mds_path,
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

    def test_joined_metadataset_joiner(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Create a joined dataset configuration
        joined_mds_path = self.dataset_path / "joined_metadataset_joiner.yaml"
        with open(joined_mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: MetadatasetV2",
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
        prepare_metadataset(EPath(joined_mds_path))

        # Train mode dataset
        train_dataset = get_train_dataset(
            joined_mds_path,
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
        joined_mds_path = self.dataset_path / "left_join.yaml"
        with open(joined_mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: MetadatasetV2",
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
        prepare_metadataset(EPath(joined_mds_path))

        # Train mode dataset
        train_dataset = get_train_dataset(
            joined_mds_path,
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
        with open(joined_mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: MetadatasetV2",
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
                joined_mds_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
            )

        # Shall succeed after preparation
        prepare_metadataset(EPath(joined_mds_path))
        train_dataset = get_train_dataset(
            joined_mds_path,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
        )
        # Check that there are no remainder files
        cache_folder = joined_mds_path.with_name(joined_mds_path.name + ".cache")
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
        joined_mds_path = self.dataset_path / "left_join.yaml"
        with open(joined_mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: MetadatasetV2",
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
        prepare_metadataset(EPath(joined_mds_path))

        # Train mode dataset
        train_dataset = get_train_dataset(
            joined_mds_path,
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

    def test_joined_metadataset_prepare_mock(self):
        torch.manual_seed(42)

        # Create a joined dataset configuration
        joined_mds_path = self.dataset_path / "joined_metadataset_prepare_mock.yaml"
        with open(joined_mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: MetadatasetV2",
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
        prepare_metadataset(EPath(joined_mds_path))

        # Create a joined dataset configuration
        joined_mds_path = self.dataset_path / "joined_metadataset_prepare_mock2.yaml"
        with open(joined_mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: MetadatasetV2",
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
        prepare_metadataset(EPath(joined_mds_path))

    def test_metadataset_fixed_epochs(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Create a joined dataset configuration
        fixed_epochs_mds_path = self.dataset_path / "metadataset_fixed_epochs.yaml"
        with open(fixed_epochs_mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: MetadatasetV2",
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
            fixed_epochs_mds_path,
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
                fixed_epochs_mds_path,
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

    def test_metadataset_fixed_fractional_epochs(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Create a joined dataset configuration
        fixed_epochs_mds_path = self.dataset_path / "metadataset_fixed_epochs.yaml"
        with open(fixed_epochs_mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: MetadatasetV2",
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
            fixed_epochs_mds_path,
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
                fixed_epochs_mds_path,
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
                fixed_epochs_mds_path,
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
                fixed_epochs_mds_path,
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
                fixed_epochs_mds_path,
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
                fixed_epochs_mds_path,
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
                fixed_epochs_mds_path,
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
            self.mds_path,
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

    def test_s3(self):
        # Create a joined dataset configuration
        mixed_mds_path = self.dataset_path / "metadataset_mixed.yaml"
        with open(mixed_mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: MetadatasetV2",
                        "splits:",
                        "  train:",
                        "    path: msc://s3test_metadataset/test/dataset/nested_metadataset_v2.yaml",
                    ]
                )
            )

        with setup_s3_emulator(profile_name="s3test_metadataset") as emu:
            # Upload the dataset to the S3 emulator
            # EPath(self.dataset_path).copy(EPath("msc://s3/test/dataset"))
            emu.add_file(self.dataset_path, "test/dataset")

            train_dataset = get_loader(
                get_train_dataset(
                    mixed_mds_path,
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
