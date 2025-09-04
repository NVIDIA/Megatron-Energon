# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for meta datasets."""

import gc
import logging
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import pytest
import torch
import webdataset as wds

from megatron.energon import (
    WorkerConfig,
    get_train_dataset,
)
from megatron.energon.dataloader import DataLoader, ForkDataLoaderWorker, ThreadDataLoaderWorker
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from megatron.energon.task_encoder.base import DefaultTaskEncoder, stateless


class VerifyWorkerTaskEncoder(DefaultTaskEncoder):
    def __init__(self, expected_num_workers: int):
        self.expected_num_workers = expected_num_workers
        super().__init__()

    @stateless
    def encode_sample(self, sample):
        sample = super().encode_sample(sample)
        worker_info = torch.utils.data.get_worker_info()
        if self.expected_num_workers > 0:
            assert worker_info is not None
            assert worker_info.num_workers == self.expected_num_workers
        else:
            assert worker_info is None
        return sample


@pytest.fixture
def temp_dir():
    temp_dir = tempfile.TemporaryDirectory()
    yield temp_dir
    gc.collect()
    temp_dir.cleanup()


@pytest.fixture
def dataset_path(temp_dir):
    dataset_path = Path(temp_dir.name)
    dataset_path.mkdir(exist_ok=True, parents=True)
    return dataset_path


@pytest.fixture
def ds1_path(dataset_path):
    ds1_path = dataset_path / "ds1"
    ds1_path.mkdir(exist_ok=True, parents=True)
    create_text_test_dataset(ds1_path, range(55), range(55))
    print(ds1_path)
    return ds1_path


@pytest.fixture(autouse=True)
def setup_logging():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    warnings.simplefilter("ignore", ResourceWarning)


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


def test_dataloader_no_workers(ds1_path):
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
            ds1_path,
            worker_config=worker_config,
            batch_size=10,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            repeat=False,
            task_encoder=VerifyWorkerTaskEncoder(expected_num_workers=0),
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
            ds1_path,
            worker_config=worker_config,
            batch_size=10,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            repeat=False,
            task_encoder=VerifyWorkerTaskEncoder(expected_num_workers=0),
        ),
    ).with_restored_state_rank(state1) as train_loader:
        cmp_order2 = [text for idx, data in zip(range(55 * 10), train_loader) for text in data.text]
        assert train_order2 == cmp_order2, (train_order1, cmp_order2)


def test_dataloader_fork(ds1_path):
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
            ds1_path,
            worker_config=worker_config,
            batch_size=10,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            repeat=False,
            task_encoder=VerifyWorkerTaskEncoder(expected_num_workers=2),
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
            ds1_path,
            worker_config=worker_config,
            batch_size=10,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            repeat=False,
            task_encoder=VerifyWorkerTaskEncoder(expected_num_workers=worker_config.num_workers),
        ),
        prefetch_factor=2,
        worker_type=ForkDataLoaderWorker,
        gc_collect_every_n_steps=10,
        gc_freeze_at_start=True,
        watchdog_timeout_seconds=60,
        fail_on_timeout=True,
    ).with_restored_state_rank(state1) as train_loader:
        cmp_order2 = [text for idx, data in zip(range(55 * 10), train_loader) for text in data.text]
        assert train_order2 == cmp_order2, (train_order1, cmp_order2)


def test_dataloader_fork_multi_parallel(ds1_path):
    torch.manual_seed(42)
    worker_config_r0 = WorkerConfig(
        rank=0,
        world_size=2,
        num_workers=2,
        seed_offset=42,
    )
    worker_config_r1 = WorkerConfig(
        rank=1,
        world_size=2,
        num_workers=2,
        seed_offset=42,
    )

    # Train mode dataset
    train_loader_r0 = DataLoader(
        get_train_dataset(
            ds1_path,
            worker_config=worker_config_r0,
            batch_size=10,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            repeat=False,
            task_encoder=VerifyWorkerTaskEncoder(expected_num_workers=worker_config_r0.num_workers),
        ),
        prefetch_factor=2,
        worker_type=ForkDataLoaderWorker,
        gc_collect_every_n_steps=10,
        gc_freeze_at_start=True,
        watchdog_timeout_seconds=60,
        fail_on_timeout=True,
    )
    assert len(train_loader_r0) == 4, len(train_loader_r0)

    train_order1_r0 = [
        text for idx, data in zip(range(55 * 10), train_loader_r0) for text in data.text
    ]
    print(train_order1_r0[:10])
    print(Counter(train_order1_r0))
    assert len(train_order1_r0) == 28, len(train_order1_r0)
    assert len(Counter(train_order1_r0)) == 28, Counter(train_order1_r0)
    assert all(v == 1 for v in Counter(train_order1_r0).values()), Counter(train_order1_r0)

    train_loader_r1 = DataLoader(
        get_train_dataset(
            ds1_path,
            worker_config=worker_config_r1,
            batch_size=10,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            repeat=False,
            task_encoder=VerifyWorkerTaskEncoder(expected_num_workers=worker_config_r1.num_workers),
        ),
        prefetch_factor=2,
        worker_type=ForkDataLoaderWorker,
        gc_collect_every_n_steps=10,
        gc_freeze_at_start=True,
        watchdog_timeout_seconds=60,
        fail_on_timeout=True,
    )
    assert len(train_loader_r1) == 4, len(train_loader_r1)

    train_order1_r1 = [
        text for idx, data in zip(range(55 * 10), train_loader_r1) for text in data.text
    ]
    print(train_order1_r1[:10])
    print(Counter(train_order1_r1))
    assert len(train_order1_r1) == 27, len(train_order1_r1)
    assert len(Counter(train_order1_r1)) == 27, Counter(train_order1_r1)
    assert all(v == 1 for v in Counter(train_order1_r1).values()), Counter(train_order1_r1)

    train_loader_r1.save_state_rank()

    train_loader_r0.save_state_rank()

    train_order2_r0 = [
        text for idx, data in zip(range(55 * 10), train_loader_r0) for text in data.text
    ]
    assert len(train_order2_r0) == 28

    train_order2_r1 = [
        text for idx, data in zip(range(55 * 10), train_loader_r1) for text in data.text
    ]
    assert len(train_order2_r1) == 27

    train_loader_r0.shutdown()
    train_loader_r1.shutdown()


def test_dataloader_thread(ds1_path):
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
            ds1_path,
            worker_config=worker_config,
            batch_size=10,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            repeat=False,
            task_encoder=VerifyWorkerTaskEncoder(expected_num_workers=worker_config.num_workers),
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
            ds1_path,
            worker_config=worker_config,
            batch_size=10,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            repeat=False,
            task_encoder=VerifyWorkerTaskEncoder(expected_num_workers=worker_config.num_workers),
        ),
        prefetch_factor=2,
        worker_type=ThreadDataLoaderWorker,
        gc_collect_every_n_steps=0,
        watchdog_timeout_seconds=60,
        fail_on_timeout=True,
    ).with_restored_state_rank(state1) as train_loader:
        cmp_order2 = [text for idx, data in zip(range(55 * 10), train_loader) for text in data.text]
        assert train_order2 == cmp_order2, (train_order1, cmp_order2)
