# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for meta datasets."""

import gc
import logging
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Iterable

import torch
import webdataset as wds

from megatron.energon import (
    BlendDataset,
    SavableDataset,
    WorkerConfig,
    get_savable_loader,
    get_train_dataset,
)
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME

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


def get_blend_dataset(ds: SavableDataset):
    if isinstance(ds, BlendDataset):
        return ds
    else:
        if hasattr(ds, "dataset"):
            return get_blend_dataset(ds.dataset)
        else:
            raise ValueError("No blend dataset found")


class TestDataset(unittest.TestCase):
    # Set up the test fixture
    def setUp(self):
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        warnings.simplefilter("ignore", ResourceWarning)

        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name)
        # self.dataset_path = Path("./test_dataset")

        self.dataset_path.mkdir(exist_ok=True, parents=True)

        (self.dataset_path / "ds1").mkdir(exist_ok=True, parents=True)
        (self.dataset_path / "ds2").mkdir(exist_ok=True, parents=True)
        (self.dataset_path / "ds3").mkdir(exist_ok=True, parents=True)

        # Create a small dummy captioning dataset
        self.create_text_test_dataset(self.dataset_path / "ds1", range(55), range(55))
        self.create_text_test_dataset(self.dataset_path / "ds2", range(100, 107), range(100, 107))
        self.create_text_test_dataset(self.dataset_path / "ds3", range(200, 255), range(0, 55))

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
                        "      - weight: 1",
                        "        path: ds2",
                        "      - weight: 1",
                        "        path: ds3",
                    ]
                )
            )

        print(self.dataset_path)

    def tearDown(self):
        # Remove all temporary files
        gc.collect()
        self.temp_dir.cleanup()

    @staticmethod
    def create_text_test_dataset(path: Path, txt_range: Iterable[int], key_range: Iterable[int]):
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
                        "txt": f"{txt}".encode(),
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
                        "__module__: megatron.energon",
                        "__class__: TextWebdataset",
                        "field_map:",
                        "  text: txt",
                        "subflavors:",
                        "  source: dataset.yaml",
                        "  dataset.yaml: true",
                        "  number: 42",
                    ]
                )
            )

    def test_metadataset_few_samples_save_restore(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=32,
            num_workers=1,
            seed_offset=42,
        )

        # Train mode dataset
        train_dataset = get_train_dataset(
            self.mds_path,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=100,
            max_samples_per_sequence=None,
        )
        print(len(train_dataset))
        assert len(train_dataset) == 4

        # The middle dataset should have 0 samples assigned to this rank
        blend_ds = get_blend_dataset(train_dataset)
        assert len(blend_ds.dataset_weights[1][0].dataset.dataset.workers_slice_offsets[0]) == 1
        assert len(blend_ds.dataset_weights[1][0].dataset.dataset) == 0

        train_loader = get_savable_loader(
            train_dataset,
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )

        data1a = list(zip(train_loader, range(3)))  # noqa: F841. Load 3 samples

        # Save state mid epoch
        state1 = train_loader.save_state_rank()

        data1b = list(zip(train_loader, range(5)))  # Load 5 samples

        # Restore state
        train_loader = get_savable_loader(
            get_train_dataset(
                self.mds_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=100,
                max_samples_per_sequence=None,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )
        train_loader.restore_state_rank(state1)
        data2_restore = list(zip(train_loader, range(5)))  # Load 5 samples

        # Check that the restored state is the same
        order1b = [(s[0].__key__[0], int(s[0].text[0])) for s in data1b]
        order2 = [(s[0].__key__[0], int(s[0].text[0])) for s in data2_restore]

        print("order1b")
        print(order1b)
        print("order2")
        print(order2)

        assert order1b == order2, "The restored state does not match the original state."


if __name__ == "__main__":
    # unittest.main()
    ds = TestDataset()
    ds.setUp()
    ds.test_metadataset_few_samples_save_restore()
    ds.tearDown()
