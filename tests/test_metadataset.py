# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for meta datasets."""

import gc
import logging
import sys
import tempfile
import time
import unittest
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import webdataset as wds

from megatron.energon import (
    SavableDataset,
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_dataset,
    load_dataset,
)
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from megatron.energon.metadataset.loader_interface import DatasetBlendMode
from megatron.energon.wrappers.blend_dataset import BlendDataset

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


def assert_nested_equal(a: Any, b: Any, path: str = "") -> None:
    """
    Recursively checks that two nested data structures (consisting of dicts, lists, tuples,
    and other basic types) are equal. If they are not equal, prints the path of the first mismatch
    and raises an AssertionError.

    Args:
        a: First nested structure to compare.
        b: Second nested structure to compare.
        path: Internal parameter used to pass the current traversal path (do not set this manually).

    Raises:
        AssertionError: If a mismatch is found.
    """
    # Check if types differ
    if type(a) is not type(b):
        mismatch_details = f"Type mismatch at {path or '<root>'}: {type(a)} != {type(b)}"
        print(mismatch_details)
        raise AssertionError(mismatch_details)

    # If they are both dictionaries, compare each key and value
    if isinstance(a, dict):
        # Check if they have the same keys
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        if a_keys != b_keys:
            missing_in_a = b_keys - a_keys
            missing_in_b = a_keys - b_keys
            mismatch_details = (
                f"Key mismatch at {path or '<root>'}:\n"
                + "Missing in first object: "
                + ", ".join(f"[{k}]={b[k]!r}" for k in missing_in_a)
                + "\n"
                + "Missing in second object: "
                + ", ".join(f"[{k}]={a[k]!r}" for k in missing_in_b)
                + "\n"
            )
            print(mismatch_details)
            raise AssertionError(mismatch_details)
        for key in a:
            sub_path = f"{path}['{key}']" if path else f"['{key}']"
            assert_nested_equal(a[key], b[key], sub_path)

    # If they are lists (or tuples), compare elements in order
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            mismatch_details = f"Length mismatch at {path or '<root>'}: {len(a)} != {len(b)}"
            print(mismatch_details)
            raise AssertionError(mismatch_details)
        for index, (item_a, item_b) in enumerate(zip(a, b)):
            sub_path = f"{path}[{index}]" if path else f"[{index}]"
            assert_nested_equal(item_a, item_b, sub_path)

    # Otherwise, compare values directly
    else:
        if a != b:
            mismatch_details = f"Value mismatch at {path or '<root>'}: {repr(a)} != {repr(b)}"
            print(mismatch_details)
            raise AssertionError(mismatch_details)


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

        # Create a small dummy captioning dataset
        self.create_text_test_dataset(self.dataset_path / "ds1", range(55), range(55))
        self.create_text_test_dataset(self.dataset_path / "ds2", range(100, 155), range(100, 155))
        self.create_text_test_dataset(self.dataset_path / "ds3", range(200, 255), range(0, 55))

        self.mds_path = self.dataset_path / "metadataset.yaml"
        with open(self.mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Metadataset",
                        "splits:",
                        "  train:",
                        "    datasets:",
                        "      - weight: 1",
                        "        path: ds1",
                        "        subflavor: ds1",
                        "        subflavors:",
                        "          source: metadataset.yaml",
                        "          number: 43",
                        "          mds: mds",
                        "        shuffle_over_epochs_multiplier: 3",
                        "      - weight: 1",
                        "        path: ds2",
                        "        subflavor: ds2",
                        "        subflavors:",
                        "          source: metadataset.yaml",
                        "          number: 44",
                        "          mds: mds",
                        "  val:",
                        "    datasets:",
                        "      - weight: 1",
                        "        path: ds1",
                        "        split_part: train",
                        "      - weight: 1",
                        "        path: ds2",
                        "        split_part: train",
                    ]
                )
            )
        self.nested_mds_path = self.dataset_path / "nested_metadataset.yaml"
        with open(self.nested_mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "splits:",
                        "  train:",
                        "    datasets:",
                        "      - weight: 4",
                        "        path: ./metadataset.yaml",
                        "        split_part: train",
                        "        subflavor: train",
                        "        subflavors:",
                        "          source: nested_metadataset.yaml",
                        "          mds: nested_train",
                        "      - path: ./metadataset.yaml",
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

        train_subflavors = [
            subflavor["__subflavor__"]
            for idx, data in zip(range(55), train_loader1)
            for subflavor in data.__subflavors__
        ]
        print(train_subflavors[:10])
        print(Counter(train_subflavors))
        assert len(Counter(train_subflavors)) == 2
        assert all(250 <= v <= 300 for v in Counter(train_subflavors).values())

        # Train mode dataset
        train_dataset = get_train_dataset(
            self.mds_path,
            worker_config=worker_config,
            batch_size=10,
            shuffle_buffer_size=25,
            max_samples_per_sequence=25,
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

        # Val mode dataset
        val_dataset = get_val_dataset(self.mds_path, worker_config=worker_config, batch_size=10)
        print(len(val_dataset))
        assert len(val_dataset) == 11

        val_loader1 = get_loader(val_dataset)

        val_order1 = [text for data in val_loader1 for text in data.text]
        assert len(val_order1) == 110
        print(Counter(val_order1))
        assert all(v == 1 for v in Counter(val_order1).values())

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
        assert [raw_dataset.weight for raw_dataset in raw_datasets.datasets] == [0.4, 0.4, 0.1, 0.1]
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
                "__subflavor__": "train",
            },
            {
                "source": "nested_metadataset.yaml",
                "dataset.yaml": True,
                "number": 44,
                "mds": "nested_train",
                "__subflavor__": "train",
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

        # Train mode dataset
        train_dataset = get_train_dataset(
            self.nested_mds_path,
            worker_config=worker_config,
            batch_size=10,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
        )
        print(len(train_dataset))
        assert len(train_dataset) == 22

        train_loader1 = get_loader(train_dataset)

        train_order1 = [
            text for idx, data in zip(range(55 * 10), train_loader1) for text in data.text
        ]
        print(train_order1[:10])
        print(Counter(train_order1))
        assert len(Counter(train_order1)) == 110
        assert all(48 <= v <= 53 for v in Counter(train_order1).values())

        train_subflavors = [
            subflavor.get("__subflavor__")
            for idx, data in zip(range(55), train_loader1)
            for subflavor in data.__subflavors__
        ]
        cnt = Counter(train_subflavors)
        print(train_subflavors[:10])
        print(cnt)
        avg = 55 * 10 / 5
        assert len(Counter(train_subflavors)) == 2
        assert avg * 4 - 40 < cnt["train"] < avg * 4 + 40
        assert avg - 10 < cnt[None] < avg + 10

        train_subflavorss = [
            tuple(subflavor.items())
            for idx, data in zip(range(55), train_loader1)
            for subflavor in data.__subflavors__
        ]
        cnt = Counter(train_subflavorss)
        print(train_subflavorss[:10])
        print(cnt)
        assert len(Counter(train_subflavorss)) == 3
        assert (
            avg * 2 - 20
            < cnt[
                (
                    ("source", "nested_metadataset.yaml"),
                    ("dataset.yaml", True),
                    ("number", 43),
                    ("__subflavor__", "train"),
                    ("mds", "nested_train"),
                )
            ]
            < avg * 2 + 20
        )
        assert (
            avg * 2 - 20
            < cnt[
                (
                    ("source", "nested_metadataset.yaml"),
                    ("dataset.yaml", True),
                    ("number", 44),
                    ("__subflavor__", "train"),
                    ("mds", "nested_train"),
                )
            ]
            < avg * 2 + 20
        )
        assert (
            avg * 1 - 20
            < cnt[
                (
                    ("source", "nested_metadataset.yaml"),
                    ("dataset.yaml", True),
                    ("number", 42),
                    ("mds", "nested_val"),
                )
            ]
            < avg * 1 + 20
        )

        # Train mode dataset
        train_dataset = get_train_dataset(
            self.mds_path,
            worker_config=worker_config,
            batch_size=10,
            shuffle_buffer_size=25,
            max_samples_per_sequence=25,
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

        # Val mode dataset
        val_dataset = get_val_dataset(self.mds_path, worker_config=worker_config, batch_size=10)
        print(len(val_dataset))
        assert len(val_dataset) == 11

        val_loader1 = get_loader(val_dataset)

        val_order1 = [text for data in val_loader1 for text in data.text]
        assert len(val_order1) == 110
        print(Counter(val_order1))
        assert all(v == 1 for v in Counter(val_order1).values())

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
                    self.nested_mds_path,
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
                    self.mds_path,
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

        wrk_cfg = worker_config.config()
        assert wrk_cfg == {
            "rank": 0,
            "world_size": 1,
            "num_workers": 0,
            "data_parallel_group": None,
        }
        print("loader.config():")
        print(loader.config())
        print()
        reference_config = {
            "type": "SavableDataLoader",
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": True,
            "prefetch_factor": None,
            "dataset": {
                "type": "MapDataset",
                "dataset": {
                    "type": "BatchDataset",
                    "batch_size": 10,
                    "batcher": "megatron.energon.task_encoder.base.DefaultTaskEncoder.batch",
                    "batcher_stateless": True,
                    "drop_last": False,
                    "error_handler": "megatron.energon.task_encoder.base.TaskEncoder.error_handler",
                    "worker_config": wrk_cfg,
                    "dataset": {
                        "type": "MapDataset",
                        "dataset": {
                            "type": "BlendDataset",
                            "dataset_weights": [
                                (
                                    {
                                        "type": "RepeatDataset",
                                        "dataset": {
                                            "type": "MapDataset",
                                            "dataset": {
                                                "type": "WebdatasetSampleLoaderDataset",
                                                "joins": 1,
                                                "len": 55,
                                                "slice_offsets": [[0, 10, 20, 30, 40, 50, 55]],
                                                "worker_config": wrk_cfg,
                                                "shuffle_over_epochs": 6,
                                                "parallel_slice_iters": 2,
                                            },
                                            "map_fn": "megatron.energon.flavors.webdataset.base_webdataset.BaseWebdatasetFactory._load_sample_raw",
                                            "map_fn_config": {
                                                "type": "StandardWebdatasetFactory",
                                                "training": True,
                                                "_path": str(self.dataset_path / "ds1"),
                                                "shards": [
                                                    {
                                                        "name": "parts/data-0.tar",
                                                        "count": 10,
                                                        "_path": str(
                                                            self.dataset_path
                                                            / "ds1/parts/data-0.tar"
                                                        ),
                                                    },
                                                    {
                                                        "name": "parts/data-1.tar",
                                                        "count": 10,
                                                        "_path": str(
                                                            self.dataset_path
                                                            / "ds1/parts/data-1.tar"
                                                        ),
                                                    },
                                                    {
                                                        "name": "parts/data-2.tar",
                                                        "count": 10,
                                                        "_path": str(
                                                            self.dataset_path
                                                            / "ds1/parts/data-2.tar"
                                                        ),
                                                    },
                                                    {
                                                        "name": "parts/data-3.tar",
                                                        "count": 10,
                                                        "_path": str(
                                                            self.dataset_path
                                                            / "ds1/parts/data-3.tar"
                                                        ),
                                                    },
                                                    {
                                                        "name": "parts/data-4.tar",
                                                        "count": 10,
                                                        "_path": str(
                                                            self.dataset_path
                                                            / "ds1/parts/data-4.tar"
                                                        ),
                                                    },
                                                    {
                                                        "name": "parts/data-5.tar",
                                                        "count": 5,
                                                        "_path": str(
                                                            self.dataset_path
                                                            / "ds1/parts/data-5.tar"
                                                        ),
                                                    },
                                                ],
                                                "sample_excludes": [],
                                                "shuffle_over_epochs": 6,
                                                "parallel_shard_iters": 2,
                                                "max_samples_per_sequence": None,
                                                "subset": None,
                                                "subflavors": {
                                                    "source": "metadataset.yaml",
                                                    "dataset.yaml": True,
                                                    "number": 43,
                                                    "mds": "mds",
                                                    "__subflavor__": "ds1",
                                                },
                                                "sample_loader": "megatron.energon.flavors.webdataset.default_generic_webdataset.DefaultGenericWebdatasetFactory.__init__.<locals>.<lambda>",
                                                "image_decode": "torchrgb",
                                                "av_decode": "AVDecoder",
                                                "video_decode_audio": False,
                                                "guess_content": False,
                                            },
                                            "map_fn_stateless": True,
                                        },
                                        "repeats": None,
                                        "worker_config": wrk_cfg,
                                    },
                                    0.5,
                                ),
                                (
                                    {
                                        "type": "RepeatDataset",
                                        "dataset": {
                                            "type": "MapDataset",
                                            "dataset": {
                                                "type": "WebdatasetSampleLoaderDataset",
                                                "joins": 1,
                                                "len": 55,
                                                "slice_offsets": [[0, 10, 20, 30, 40, 50, 55]],
                                                "worker_config": wrk_cfg,
                                                "shuffle_over_epochs": 2,
                                                "parallel_slice_iters": 2,
                                            },
                                            "map_fn": "megatron.energon.flavors.webdataset.base_webdataset.BaseWebdatasetFactory._load_sample_raw",
                                            "map_fn_config": {
                                                "type": "StandardWebdatasetFactory",
                                                "training": True,
                                                "_path": str(self.dataset_path / "ds2"),
                                                "shards": [
                                                    {
                                                        "name": "parts/data-0.tar",
                                                        "count": 10,
                                                        "_path": str(
                                                            self.dataset_path
                                                            / "ds2/parts/data-0.tar"
                                                        ),
                                                    },
                                                    {
                                                        "name": "parts/data-1.tar",
                                                        "count": 10,
                                                        "_path": str(
                                                            self.dataset_path
                                                            / "ds2/parts/data-1.tar"
                                                        ),
                                                    },
                                                    {
                                                        "name": "parts/data-2.tar",
                                                        "count": 10,
                                                        "_path": str(
                                                            self.dataset_path
                                                            / "ds2/parts/data-2.tar"
                                                        ),
                                                    },
                                                    {
                                                        "name": "parts/data-3.tar",
                                                        "count": 10,
                                                        "_path": str(
                                                            self.dataset_path
                                                            / "ds2/parts/data-3.tar"
                                                        ),
                                                    },
                                                    {
                                                        "name": "parts/data-4.tar",
                                                        "count": 10,
                                                        "_path": str(
                                                            self.dataset_path
                                                            / "ds2/parts/data-4.tar"
                                                        ),
                                                    },
                                                    {
                                                        "name": "parts/data-5.tar",
                                                        "count": 5,
                                                        "_path": str(
                                                            self.dataset_path
                                                            / "ds2/parts/data-5.tar"
                                                        ),
                                                    },
                                                ],
                                                "sample_excludes": [],
                                                "shuffle_over_epochs": 2,
                                                "parallel_shard_iters": 2,
                                                "max_samples_per_sequence": None,
                                                "subset": None,
                                                "subflavors": {
                                                    "source": "metadataset.yaml",
                                                    "dataset.yaml": True,
                                                    "number": 44,
                                                    "mds": "mds",
                                                    "__subflavor__": "ds2",
                                                },
                                                "sample_loader": "megatron.energon.flavors.webdataset.default_generic_webdataset.DefaultGenericWebdatasetFactory.__init__.<locals>.<lambda>",
                                                "image_decode": "torchrgb",
                                                "av_decode": "AVDecoder",
                                                "video_decode_audio": False,
                                                "guess_content": False,
                                            },
                                            "map_fn_stateless": True,
                                        },
                                        "repeats": None,
                                        "worker_config": wrk_cfg,
                                    },
                                    0.5,
                                ),
                            ],
                            "worker_config": wrk_cfg,
                        },
                        "map_fn": "megatron.energon.task_encoder.base.DefaultTaskEncoder.encode_sample",
                        "map_fn_stateless": True,
                    },
                },
                "map_fn": "megatron.energon.task_encoder.base.DefaultTaskEncoder.encode_batch",
                "map_fn_stateless": True,
            },
        }
        print("Comparing dataset configs in test_save_restore_state_train.")
        assert_nested_equal(loader.config(), reference_config)

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
                    self.mds_path,
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
                self.mds_path,
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
                self.mds_path,
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
                self.mds_path,
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
                self.mds_path,
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
            get_val_dataset(self.mds_path, worker_config=worker_config, batch_size=10),
        )
        state_0 = loader.save_state_rank()
        order_1 = [data.text for idx, data in zip(range(55 * 20), loader)]
        state_1 = loader.save_state_rank()
        # print("save state done")
        order_2 = [data.text for idx, data in zip(range(55 * 20), loader)]

        loader = get_savable_loader(
            get_val_dataset(self.mds_path, worker_config=worker_config, batch_size=10),
        )
        loader.restore_state_rank(state_1)
        order_3 = [data.text for idx, data in zip(range(55 * 20), loader)]
        assert order_2 == order_3

        loader = get_savable_loader(
            get_val_dataset(self.mds_path, worker_config=worker_config, batch_size=10),
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
                    self.mds_path,
                    split_part="train",
                    worker_config=rank_config,
                    batch_size=micro_batch_size,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                )
                loader = get_loader(ds)

                subflavors = [
                    data.__subflavors__[0].get("__subflavor__")
                    for idx, data in zip(range(25), loader)
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
                    self.mds_path,
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
                self.nested_mds_path,
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
                    self.nested_mds_path,
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
                self.nested_mds_path,
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
                        self.nested_mds_path,
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


if __name__ == "__main__":
    unittest.main()
