# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for meta datasets."""

import gc
import json
import logging
import random
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import pytest
import torch
from click.testing import CliRunner

from megatron.energon import (
    Cooker,
    CrudeSample,
    DefaultTaskEncoder,
    Sample,
    WorkerConfig,
    basic_sample_keys,
    edataclass,
    get_loader,
    get_train_dataset,
    stateless,
)
from megatron.energon.tools.prepare import command as prepare_command
from tests.epath_s3_emulator import setup_s3_emulator

# Speed up tests significantly by reducing the torch status check interval for broken worker shutdown
try:
    torch.utils.data._utils.worker.MP_STATUS_CHECK_INTERVAL = 0.1
    torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 0.1
except AttributeError:
    pass


@edataclass
class TextSample(Sample):
    idx: int
    text: str


@stateless()
def cook_text(sample: CrudeSample) -> TextSample:
    return TextSample(
        **basic_sample_keys(sample), idx=sample["json"]["idx"], text=sample["json"]["txt"]
    )


class SimpleCookingTaskEncoder(DefaultTaskEncoder):
    cookers = [Cooker(cook=cook_text)]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.TemporaryDirectory()
    yield temp_dir
    gc.collect()
    temp_dir.cleanup()


@pytest.fixture
def dataset_path(temp_dir):
    """Create dataset path and setup test data."""
    random.seed(42)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    warnings.simplefilter("ignore", ResourceWarning)

    dataset_path = Path(temp_dir.name)
    dataset_path.mkdir(exist_ok=True, parents=True)

    # Create a small dummy datasets
    create_text_test_dataset(dataset_path / "ds1.jsonl", range(55), range(55))
    create_text_test_dataset(dataset_path / "ds2.jsonl", range(100, 155), range(100, 155))
    create_text_test_dataset(dataset_path / "ds3.jsonl", range(200, 255), range(55))

    mds_all_path = dataset_path / "metadataset_all.yaml"
    with open(mds_all_path, "w") as f:
        f.write(
            "\n".join(
                [
                    "__module__: megatron.energon",
                    "__class__: MetadatasetV2",
                    "splits:",
                    "  train:",
                    "    blend:",
                    "      - path: ds1.jsonl",
                    "        subflavors:",
                    "          ds: ds1",
                    "      - path: ds2.jsonl",
                    "        subflavors:",
                    "          ds: ds2",
                    "      - path: ds3.jsonl",
                    "        subflavors:",
                    "          ds: ds3",
                ]
            )
        )

    return dataset_path


def create_text_test_dataset(
    path: Path, txt_range: Iterable[int], key_range: Iterable[int], prefix: str = ""
):
    """Creates a small dummy test dataset for testing purposes."""

    # Write jsonl file
    with open(path, "w") as wf:
        for key, txt in zip(key_range, txt_range):
            # Write JSON entries to the file, one per line.
            wf.write(json.dumps({"idx": key, "txt": f"{prefix}{txt}"}) + "\n")

    from megatron.energon.flavors import CrudeJsonlDatasetFactory

    CrudeJsonlDatasetFactory.prepare_dataset(path)


def test_dataset(dataset_path):
    torch.manual_seed(42)
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=0,
        seed_offset=42,
    )

    # Train mode dataset
    train_dataset = get_train_dataset(
        dataset_path / "ds1.jsonl",
        worker_config=worker_config,
        batch_size=1,
        shuffle_buffer_size=None,
        max_samples_per_sequence=None,
        task_encoder=SimpleCookingTaskEncoder(),
    )
    print(len(train_dataset))
    assert len(train_dataset) == 55, f"Expected 55 samples, got {len(train_dataset)}"

    with get_loader(train_dataset) as train_loader1:
        train_order1 = [
            text for idx, data in zip(range(55 * 10), train_loader1) for text in data.text
        ]
        print(train_order1[:10])
        print(Counter(train_order1))
        assert len(Counter(train_order1)) == 55
        assert all(v == 10 for v in Counter(train_order1).values())


def test_metadataset_all(dataset_path):
    torch.manual_seed(42)
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=2,
        seed_offset=42,
    )

    # Train mode dataset
    train_dataset = get_train_dataset(
        dataset_path / "metadataset_all.yaml",
        worker_config=worker_config,
        batch_size=1,
        shuffle_buffer_size=None,
        max_samples_per_sequence=None,
        task_encoder=SimpleCookingTaskEncoder(),
    )
    print(len(train_dataset))
    assert len(train_dataset) == 55 * 3, f"Expected 55 * 3 samples, got {len(train_dataset)}"

    with get_loader(train_dataset) as train_loader1:
        train_order1 = [
            text for idx, data in zip(range(55 * 10), train_loader1) for text in data.text
        ]
        print(train_order1[:10])
        print(Counter(train_order1))
        assert len(Counter(train_order1)) == 55 * 3
        assert all(2 <= v <= 5 for v in Counter(train_order1).values())


def test_metadataset_multirank(dataset_path):
    torch.manual_seed(42)

    sample_counts = Counter()
    expected_lens = [19, 19, 17]

    for cur_rank in range(3):
        worker_config = WorkerConfig(
            rank=cur_rank,
            world_size=3,
            num_workers=5,
            seed_offset=42,
        )

        # Train mode dataset
        train_dataset = get_train_dataset(
            dataset_path / "ds1.jsonl",
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            task_encoder=SimpleCookingTaskEncoder(),
            repeat=False,
        )
        print(len(train_dataset))
        assert len(train_dataset) == expected_lens[cur_rank], (
            f"Expected {expected_lens[cur_rank]} samples, got {len(train_dataset)}"
        )

        with get_loader(train_dataset) as train_loader1:
            for data in train_loader1:
                sample_counts[int(data.text[0])] += 1

    for i in range(55):
        assert sample_counts[i] == 1, (
            f"Sample {i} should have been seen exactly once, but was seen {sample_counts[i]} times."
        )


def test_s3(dataset_path):
    # Create a joined dataset configuration
    mixed_mds_path = dataset_path / "metadataset_mixed.yaml"
    with open(mixed_mds_path, "w") as f:
        f.write(
            "\n".join(
                [
                    "__module__: megatron.energon",
                    "__class__: MetadatasetV2",
                    "splits:",
                    "  train:",
                    "    path: msc://s3test_jsonl_dataset/test/dataset/metadataset_all.yaml",
                ]
            )
        )

    with setup_s3_emulator(profile_name="s3test_jsonl_dataset") as emu:
        # Upload the dataset to the S3 emulator
        # EPath(dataset_path).copy(EPath("msc://s3/test/dataset"))
        emu.add_file(dataset_path, "test/dataset")

        with get_loader(
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
                virtual_epoch_length=55 * 10,
                task_encoder=SimpleCookingTaskEncoder(),
            )
        ) as train_dataset:
            data = list(enumerate(train_dataset))
            assert len(data) == 55 * 10, len(data)
            cnt = Counter(t for _, entry in data for t in entry.text)
            assert len(cnt) == 55 * 3
            assert all(2 <= v <= 5 for v in cnt.values())


def test_prepare(dataset_path):
    print("Creating new dataset")
    with open(dataset_path / "ds_prep.jsonl", "w") as f:
        for i in range(10):
            f.write(json.dumps({"idx": i, "txt": f"{i}"}) + "\n\n")

    runner = CliRunner()
    result = runner.invoke(
        prepare_command,
        [str(dataset_path / "ds_prep.jsonl")],
        catch_exceptions=False,
    )
    print(result.stdout)
    assert result.exit_code == 0, "Prepare failed, see output"
    assert "Done" in result.stdout, "Prepare failed, see output"
    assert "Found 10 samples" in result.stdout, "Prepare failed, see output"
    assert (dataset_path / "ds_prep.jsonl.idx").exists()

    torch.manual_seed(42)

    # Train mode dataset
    with get_loader(
        get_train_dataset(
            dataset_path / "ds_prep.jsonl",
            worker_config=WorkerConfig(
                rank=0,
                world_size=1,
                num_workers=0,
                seed_offset=42,
            ),
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            task_encoder=SimpleCookingTaskEncoder(),
        )
    ) as train_loader:
        assert len(train_loader) == 10, f"Expected 10 samples, got {len(train_loader)}"

        train_order1 = [text for _, data in zip(range(50), train_loader) for text in data.text]
        print(train_order1[:10])
        print(Counter(train_order1))
        assert len(Counter(train_order1)) == 10
        assert all(v == 5 for v in Counter(train_order1).values())
