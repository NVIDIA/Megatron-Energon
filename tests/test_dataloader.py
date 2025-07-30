# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for meta datasets."""

import gc
import logging
import sys
import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
import webdataset as wds

from megatron.energon import (
    WorkerConfig,
    get_train_dataset,
)
from megatron.energon.dataloader import DataLoader, ForkDataLoaderWorker, ThreadDataLoaderWorker
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME


class TestDataloader(unittest.TestCase):
    # Set up the test fixture
    def setUp(self):
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        warnings.simplefilter("ignore", ResourceWarning)

        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name)
        # self.dataset_path = Path("./test_dataset")

        self.dataset_path.mkdir(exist_ok=True, parents=True)

        self.ds1_path = self.dataset_path / "ds1"
        self.ds1_path.mkdir(exist_ok=True, parents=True)

        # Create a small dummy captioning dataset
        self.create_text_test_dataset(self.ds1_path, range(55), range(55))
        print(self.ds1_path)

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

    def test_dataloader_no_workers(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Train mode dataset
        with DataLoader(
            get_train_dataset(
                self.ds1_path,
                worker_config=worker_config,
                batch_size=10,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
            ),
        ) as train_loader:
            assert len(train_loader) == 6, len(train_loader)

            train_order1 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]
            print(train_order1[:10])
            print(Counter(train_order1))
            assert len(train_order1) == 55, len(train_order1)
            assert len(Counter(train_order1)) == 55, Counter(train_order1)
            assert all(v == 1 for v in Counter(train_order1).values()), Counter(train_order1)

            state1 = train_loader.save_state_rank()

            train_order2 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]

        with DataLoader(
            get_train_dataset(
                self.ds1_path,
                worker_config=worker_config,
                batch_size=10,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
            ),
        ).with_restored_state_rank(state1) as train_loader:
            cmp_order2 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]
            assert train_order2 == cmp_order2, (train_order1, cmp_order2)

    def test_dataloader_fork(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=2,
            seed_offset=42,
        )

        # Train mode dataset
        with DataLoader(
            get_train_dataset(
                self.ds1_path,
                worker_config=worker_config,
                batch_size=10,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
            ),
            prefetch_factor=2,
            worker_type=ForkDataLoaderWorker,
            gc_collect_every_n_steps=10,
            gc_freeze_at_start=True,
            watchdog_timeout_seconds=60,
            fail_on_timeout=True,
        ) as train_loader:
            assert len(train_loader) == 6, len(train_loader)

            train_order1 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]
            print(train_order1[:10])
            print(Counter(train_order1))
            assert len(train_order1) == 55, len(train_order1)
            assert len(Counter(train_order1)) == 55, Counter(train_order1)
            assert all(v == 1 for v in Counter(train_order1).values()), Counter(train_order1)

            state1 = train_loader.save_state_rank()

            train_order2 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]

            assert len(train_order1) == len(train_order2), (len(train_order1), len(train_order2))

        with DataLoader(
            get_train_dataset(
                self.ds1_path,
                worker_config=worker_config,
                batch_size=10,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
            ),
            prefetch_factor=2,
            worker_type=ForkDataLoaderWorker,
            gc_collect_every_n_steps=10,
            gc_freeze_at_start=True,
            watchdog_timeout_seconds=60,
            fail_on_timeout=True,
        ).with_restored_state_rank(state1) as train_loader:
            cmp_order2 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]
            assert train_order2 == cmp_order2, (train_order1, cmp_order2)

    def test_dataloader_thread(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=2,
            seed_offset=42,
        )

        # Train mode dataset
        with DataLoader(
            get_train_dataset(
                self.ds1_path,
                worker_config=worker_config,
                batch_size=10,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
            ),
            prefetch_factor=2,
            worker_type=ThreadDataLoaderWorker,
            gc_collect_every_n_steps=0,
            watchdog_timeout_seconds=60,
            fail_on_timeout=True,
        ) as train_loader:
            assert len(train_loader) == 6, len(train_loader)

            train_order1 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]
            print(train_order1[:10])
            print(Counter(train_order1))
            assert len(train_order1) == 55, len(train_order1)
            assert len(Counter(train_order1)) == 55, Counter(train_order1)
            assert all(v == 1 for v in Counter(train_order1).values()), Counter(train_order1)

            state1 = train_loader.save_state_rank()

            train_order2 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]

        with DataLoader(
            get_train_dataset(
                self.ds1_path,
                worker_config=worker_config,
                batch_size=10,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
            ),
            prefetch_factor=2,
            worker_type=ThreadDataLoaderWorker,
            gc_collect_every_n_steps=0,
            watchdog_timeout_seconds=60,
            fail_on_timeout=True,
        ).with_restored_state_rank(state1) as train_loader:
            cmp_order2 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]
            assert train_order2 == cmp_order2, (train_order1, cmp_order2)


if __name__ == "__main__":
    unittest.main()
