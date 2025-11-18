# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for the dataset."""

import gc
import logging
import random
import sys
import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path

import torch
import webdataset as wds
import yaml
from click.testing import CliRunner

from megatron.energon import (
    DefaultTaskEncoder,
    TextSample,
    WorkerConfig,
    get_loader,
    get_train_dataset,
)
from megatron.energon.dataset_config import get_dataset_from_config
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from megatron.energon.loader import get_savable_loader
from megatron.energon.task_encoder.base import stateless
from megatron.energon.tools.checkpoint import command_redist

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

        # Create a small dummy captioning dataset
        self.create_text_test_dataset(self.dataset_path)

        # Create temporary directories for checkpoint files
        self.checkpoint_dir = Path(self.temp_dir.name) / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        self.redist_dir = Path(self.temp_dir.name) / "redist_checkpoints"
        self.redist_dir.mkdir(exist_ok=True, parents=True)

        print(self.dataset_path)

    def tearDown(self):
        # Remove all temporary files
        gc.collect()
        self.temp_dir.cleanup()

    @staticmethod
    def create_text_test_dataset(path: Path):
        """Creates a small dummy test dataset for testing purposes."""

        # Create num_samples unique captions
        (path / "parts").mkdir(exist_ok=True, parents=True)

        # Initialize the ShardWriter
        with wds.ShardWriter(f"{path}/parts/data-%d.tar", maxcount=100) as shard_writer:
            for idx in range(55):
                # Write individual files to shards
                shard_writer.write(
                    {
                        "__key__": f"{idx:06d}",
                        "txt": f"{idx}".encode(),
                    },
                )
                # Also create smaller shards, to verify distributions
                if idx in (1, 3, 6, 10, 20, 30, 40, 50):
                    shard_writer.next_stream()
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

        # Split with alternating train/val shards
        with open(path / MAIN_FOLDER_NAME / "split2.yaml", "w") as f:
            yaml.dump(
                {
                    "split_parts": {
                        "train": [
                            "parts/data-4.tar",
                            "parts/data-0.tar",
                            "parts/data-2.tar",
                        ],
                        "val": [
                            "parts/data-1.tar",
                            "parts/data-3.tar",
                            "parts/data-5.tar",
                        ],
                    }
                },
                f,
            )

    def test_split_parts(self):
        with open(self.dataset_path / MAIN_FOLDER_NAME / "split.yaml", "r") as f:
            print(f.read())
        with open(self.dataset_path / MAIN_FOLDER_NAME / "split2.yaml", "r") as f:
            print(f.read())

        ds = get_dataset_from_config(
            self.dataset_path,
            split_config="split2.yaml",
            split_part="train",
            worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
            training=False,
            sample_type=TextSample,
        )
        dl = get_loader(ds.build())

        all_keys = [sample.__key__ for sample in dl]
        assert all_keys == [
            "parts/data-4.tar/000011",  # Shard 4 first
            "parts/data-4.tar/000012",
            "parts/data-4.tar/000013",
            "parts/data-4.tar/000014",
            "parts/data-4.tar/000015",
            "parts/data-4.tar/000016",
            "parts/data-4.tar/000017",
            "parts/data-4.tar/000018",
            "parts/data-4.tar/000019",
            "parts/data-4.tar/000020",
            "parts/data-0.tar/000000",  # Shard 0
            "parts/data-0.tar/000001",
            "parts/data-2.tar/000004",  # Shard 2
            "parts/data-2.tar/000005",
            "parts/data-2.tar/000006",
        ]

    def test_text_dataset(self):
        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)

        ds = get_dataset_from_config(
            self.dataset_path,
            split_part="train",
            training=False,
            sample_type=TextSample,
            worker_config=worker_config,
        ).build()

        # Check len operator
        assert len(ds) == 55
        # Check if iterating returns the same
        iter1 = list(get_loader(ds))
        iter2 = list(get_loader(ds))
        assert len(iter1) == 55
        assert len(iter2) == 55
        assert all(elem1.__key__ == elem2.__key__ for elem1, elem2 in zip(iter1, iter2))
        assert all(f"{idx}" == x.text for idx, x in enumerate(get_loader(ds)))

        del ds
        gc.collect()

    def test_epoch(self):
        torch.manual_seed(42)

        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=5)

        # Without shuffle buffer, should yield everything exactly once
        ds3 = get_dataset_from_config(
            self.dataset_path,
            split_part="train",
            training=True,
            sample_type=TextSample,
            worker_config=worker_config,
        )
        loader5 = get_loader(ds3.build())
        order9 = [data.text for idx, data in zip(range(55), loader5)]
        print(order9)
        print(Counter(order9))
        assert all(v == 1 for v in Counter(order9).values())

    def test_determinism(self):
        worker_config2 = WorkerConfig(rank=0, world_size=1, num_workers=2)
        worker_config2b = WorkerConfig(rank=0, world_size=1, num_workers=2, seed_offset=43)
        worker_config4 = WorkerConfig(rank=0, world_size=1, num_workers=4)

        # This seed is used by the dataset to shuffle the data
        torch.manual_seed(42)
        ds1 = get_train_dataset(
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config2,
            batch_size=1,
            shuffle_buffer_size=42,
            max_samples_per_sequence=2,
        )
        ds1b = get_train_dataset(  # Same but different seed
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config2b,
            batch_size=1,
            shuffle_buffer_size=42,
            max_samples_per_sequence=2,
        )
        ds2 = get_train_dataset(
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config2,
            batch_size=1,
            shuffle_buffer_size=42,
            max_samples_per_sequence=2,
        )
        ds3 = get_train_dataset(
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config4,
            batch_size=1,
            shuffle_buffer_size=42,
            max_samples_per_sequence=2,
        )

        # Fork the dataset twice
        loader1 = get_loader(ds1)
        loader2 = get_loader(ds1)

        order4 = [data.text[0] for idx, data in zip(range(55 * 20), loader1)]
        order5 = [data.text[0] for idx, data in zip(range(55 * 20), loader1)]
        order6 = [data.text[0] for idx, data in zip(range(55 * 20), loader2)]
        print(order4)
        print(Counter(order4))
        # +-1 is possible due to the random shuffling (actually +-2 is possible)
        assert all(17 <= v <= 22 for v in Counter(order4).values())

        assert order4 != order5
        assert order4 == order6

        loader3 = get_loader(ds1b)
        order7 = [data.text[0] for idx, data in zip(range(55 * 20), loader3)]
        assert order6 != order7

        loader4 = get_loader(ds3)
        order8 = [data.text[0] for idx, data in zip(range(55 * 100), loader4)]
        assert order6 != order8[: len(order6)]
        print(Counter(order8))
        assert all(90 <= v <= 110 for v in Counter(order8).values())

        # Delete all locals, otherwise loaders might be kept alive
        locals().clear()
        gc.collect()

    def test_determinism_taskencoder(self):
        class TestTaskEncoder(DefaultTaskEncoder):
            @stateless(restore_seeds=True)
            def encode_sample(self, sample: TextSample) -> TextSample:
                rand_str = f"_{torch.randint(0, 1000, (1,)).item()}_{random.randint(0, 1000)}"
                return TextSample(
                    __key__=sample.__key__,
                    __restore_key__=sample.__restore_key__,
                    __subflavors__=sample.__subflavors__,
                    text=sample.text + rand_str,
                )

        for num_workers in [0, 1]:
            worker_config1 = WorkerConfig(rank=0, world_size=1, num_workers=num_workers)

            # This seed is used by the dataset to shuffle the data
            torch.manual_seed(42)
            ds1a = get_train_dataset(
                self.dataset_path,
                split_part="train",
                sample_type=TextSample,
                worker_config=worker_config1,
                batch_size=1,
                shuffle_buffer_size=42,
                max_samples_per_sequence=2,
                task_encoder=TestTaskEncoder(),
            )

            torch.manual_seed(44)
            ds1b = get_train_dataset(
                self.dataset_path,
                split_part="train",
                sample_type=TextSample,
                worker_config=worker_config1,
                batch_size=1,
                shuffle_buffer_size=42,
                max_samples_per_sequence=2,
                task_encoder=TestTaskEncoder(),
            )

            # Fork the dataset twice
            loader1a = get_loader(ds1a)
            loader1b = get_loader(ds1b)

            order1a = [data.text[0] for idx, data in zip(range(55 * 20), loader1a)]
            order1b = [data.text[0] for idx, data in zip(range(55 * 20), loader1b)]

            assert order1a == order1b

        # Delete all locals, otherwise loaders might be kept alive
        locals().clear()
        gc.collect()

    def test_determinism_taskencoder_save_restore(self):
        class TestTaskEncoder(DefaultTaskEncoder):
            @stateless(restore_seeds=True)
            def encode_sample(self, sample: TextSample) -> TextSample:
                rand_str = (
                    f"_{torch.randint(0, 1000, (1,)).item()}_{random.randint(0, 1000)}"
                    + f"_{self.current_batch_index}_{self.current_sample_index}"
                )

                return TextSample(
                    __key__=sample.__key__,
                    __restore_key__=sample.__restore_key__,
                    __subflavors__=sample.__subflavors__,
                    text=sample.text + rand_str,
                )

        for num_workers in [1, 0]:
            worker_config1 = WorkerConfig(rank=0, world_size=1, num_workers=num_workers)

            # This seed is used by the dataset to shuffle the data
            torch.manual_seed(42)
            ds1a = get_train_dataset(
                self.dataset_path,
                split_part="train",
                sample_type=TextSample,
                worker_config=worker_config1,
                batch_size=1,
                shuffle_buffer_size=42,
                max_samples_per_sequence=2,
                task_encoder=TestTaskEncoder(),
            )

            torch.manual_seed(44)
            ds1b = get_train_dataset(
                self.dataset_path,
                split_part="train",
                sample_type=TextSample,
                worker_config=worker_config1,
                batch_size=1,
                shuffle_buffer_size=42,
                max_samples_per_sequence=2,
                task_encoder=TestTaskEncoder(),
            )

            # Fork the dataset twice
            loader1a = get_savable_loader(ds1a)
            loader1b = get_savable_loader(ds1b)

            # Load 7 samples
            data_pre = [data.text[0] for idx, data in zip(range(7), loader1a)]

            # Then save state
            state = loader1a.save_state_rank()

            # Load another 20 samples
            data_post = [data.text[0] for idx, data in zip(range(20), loader1a)]

            # Restore state
            loader1b.restore_state_rank(state)

            # Load 20 samples again
            data_restored = [data.text[0] for idx, data in zip(range(20), loader1b)]

            print("Data post:", data_post)
            print("Data restored:", data_restored)

            assert data_post == data_restored

        # Delete all locals, otherwise loaders might be kept alive
        locals().clear()
        gc.collect()

    def test_restore_state(self):
        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)

        count1 = 55 * 20
        count2 = 55 * 20
        sbs = 42
        # count1 = 4
        # count2 = 2
        # sbs = None
        psi = None

        # This seed is used by the dataset to shuffle the data
        torch.manual_seed(42)

        loader = get_savable_loader(
            get_train_dataset(
                self.dataset_path,
                split_part="train",
                sample_type=TextSample,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=sbs,
                max_samples_per_sequence=2,
                parallel_shard_iters=psi,
            )
        )

        # print("save state")
        state_0 = loader.save_state_global(global_dst_rank=0)
        # print("save state done")
        order_1 = [data.text[0] for idx, data in zip(range(count1), loader)]
        assert len(order_1) == count1
        # print("save state")
        state_1 = loader.save_state_global(global_dst_rank=0)
        # print("save state done")
        order_2 = [data.text[0] for idx, data in zip(range(count2), loader)]
        assert len(order_2) == count2

        print("state0", state_0)
        print("state1", state_1)

        torch.manual_seed(213)
        loader = get_savable_loader(
            get_train_dataset(
                self.dataset_path,
                split_part="train",
                sample_type=TextSample,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=sbs,
                max_samples_per_sequence=2,
                parallel_shard_iters=psi,
            )
        )
        loader.restore_state_global(state_0, src_rank=None)
        order_45 = [data.text[0] for idx, data in zip(range(count1 + count2), loader)]
        order_4 = order_45[:count1]
        order_5 = order_45[count1:]
        # print("order1", order_1)
        # print("order2", order_2)
        # print("order4", order_4)
        assert order_1 == order_4
        # print("order5", order_5)
        assert order_2 == order_5

        torch.manual_seed(145)
        loader = get_savable_loader(
            get_train_dataset(
                self.dataset_path,
                split_part="train",
                sample_type=TextSample,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=sbs,
                max_samples_per_sequence=2,
                parallel_shard_iters=psi,
            )
        )
        # print("restore state")
        loader.restore_state_global(state_1, src_rank=None)
        # print("restore state done")
        order_3 = [data.text[0] for idx, data in zip(range(count2), loader)]
        # print("order1", order_1)
        # print("order2", order_2[:100])
        # print("order3", order_3[:100])
        assert order_2 == order_3

    def test_restore_state_dist(self):
        from multiprocessing import Manager, Process

        import torch.distributed as dist

        world_size = 3

        count1 = 55 * 20
        count2 = 55 * 20
        sbs = 42
        psi = None

        def phase1(rank: int, world_size: int, shared_dict: dict):
            worker_config = WorkerConfig(rank=rank, world_size=world_size, num_workers=0)

            # This seed is used by the dataset to shuffle the data
            torch.manual_seed(42)

            loader = get_savable_loader(
                get_train_dataset(
                    self.dataset_path,
                    split_part="train",
                    sample_type=TextSample,
                    worker_config=worker_config,
                    batch_size=1,
                    shuffle_buffer_size=sbs,
                    max_samples_per_sequence=2,
                    parallel_shard_iters=psi,
                )
            )

            state_0 = loader.save_state_global(global_dst_rank=0)
            order_1 = [data.text[0] for idx, data in zip(range(count1), loader)]
            assert len(order_1) == count1

            # print(f"Rank {rank}: order_1", order_1)

            state_1 = loader.save_state_global(global_dst_rank=0)
            order_2 = [data.text[0] for idx, data in zip(range(count2), loader)]
            assert len(order_2) == count2

            shared_dict[(rank, "order_1")] = order_1
            shared_dict[(rank, "order_2")] = order_2

            if rank == 0:
                shared_dict["state_0"] = state_0
                shared_dict["state_1"] = state_1

        def phase2(rank: int, world_size: int, shared_dict: dict):
            order_1 = shared_dict[(rank, "order_1")]
            order_2 = shared_dict[(rank, "order_2")]

            if rank == 0:
                state_0 = shared_dict["state_0"]
                state_1 = shared_dict["state_1"]
            else:
                state_0 = None
                state_1 = None

            worker_config = WorkerConfig(rank=rank, world_size=world_size, num_workers=0)

            torch.manual_seed(213)
            loader = get_savable_loader(
                get_train_dataset(
                    self.dataset_path,
                    split_part="train",
                    sample_type=TextSample,
                    worker_config=worker_config,
                    batch_size=1,
                    shuffle_buffer_size=sbs,
                    max_samples_per_sequence=2,
                    parallel_shard_iters=psi,
                )
            )
            loader.restore_state_global(state_0, src_rank=0)

            order_45 = [data.text[0] for idx, data in zip(range(count1 + count2), loader)]
            order_4 = order_45[:count1]
            order_5 = order_45[count1:]

            # print(f"Rank {rank}: order_4", order_4)

            assert order_1 == order_4
            assert order_2 == order_5

            torch.manual_seed(213)
            loader = get_savable_loader(
                get_train_dataset(
                    self.dataset_path,
                    split_part="train",
                    sample_type=TextSample,
                    worker_config=worker_config,
                    batch_size=1,
                    shuffle_buffer_size=sbs,
                    max_samples_per_sequence=2,
                    parallel_shard_iters=psi,
                )
            )
            loader.restore_state_global(state_1, src_rank=0)
            order_3 = [data.text[0] for idx, data in zip(range(count2), loader)]
            assert order_2 == order_3

        def init_process(rank, world_size, shared_dict, fn, backend="gloo"):
            """Initializes the distributed environment."""
            dist.init_process_group(
                backend=backend,
                init_method="tcp://127.0.0.1:12355",
                world_size=world_size,
                rank=rank,
            )
            fn(rank, world_size, shared_dict)
            dist.destroy_process_group()

        with Manager() as manager:
            shared_dict = manager.dict()

            # Phase 1 (save state)
            processes = []
            for rank in range(world_size):
                p = Process(target=init_process, args=(rank, world_size, shared_dict, phase1))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            # Phase 2 (restore state)
            processes = []
            for rank in range(world_size):
                p = Process(target=init_process, args=(rank, world_size, shared_dict, phase2))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

    def test_restore_state_workers(self):
        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=2)

        psi = 2
        sbs = 42
        n1 = 18
        n2 = 109
        n3 = 28
        ces = 0

        # This seed is used by the dataset to shuffle the data
        torch.manual_seed(42)
        ds = get_train_dataset(
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=sbs,
            max_samples_per_sequence=2,
            parallel_shard_iters=psi,
        )
        loader = get_savable_loader(ds, checkpoint_every_sec=ces)

        # print("save state")
        state_0 = loader.save_state_rank()
        it1 = iter(loader)
        # print("save state done")
        order_1 = [data.text[0] for idx, data in zip(range(n1), it1)]
        # print("save state")
        # time.sleep(0.5)
        state_1 = loader.save_state_rank()
        # print("save state done")
        order_2 = [data.text[0] for idx, data in zip(range(n2), it1)]
        state_2 = loader.save_state_rank()
        order_3 = [data.text[0] for idx, data in zip(range(n3), it1)]

        print("order_1", order_1)
        print("order_2", order_2)
        print("order_3", order_3)

        # print("state0", state_0)
        print("state1", state_1)
        print("state2", state_2)

        # Restoring the state of a new dataset should also yield the same
        torch.manual_seed(42)
        ds = get_train_dataset(
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=sbs,
            max_samples_per_sequence=2,
            parallel_shard_iters=psi,
        )
        loader = get_savable_loader(ds)
        loader.restore_state_rank(state_0)
        order_6 = [data.text[0] for idx, data in zip(range(n1), loader)]
        print("order1", order_1)
        print("order6", order_6)
        assert order_6 == order_1

        # Restoring the state of a new dataset should also yield the same
        torch.manual_seed(42)
        ds = get_train_dataset(
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=sbs,
            max_samples_per_sequence=2,
            parallel_shard_iters=psi,
        )
        loader = get_savable_loader(ds)
        loader.restore_state_rank(state_1)
        order_7 = [data.text[0] for idx, data in zip(range(n2), loader)]
        print("order2", order_2[:100])
        print("order7", order_7[:100])
        assert order_7 == order_2

        # Restoring the state of a new dataset should also yield the same
        torch.manual_seed(42)
        ds = get_train_dataset(
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config,
            batch_size=1,
            max_samples_per_sequence=2,
            shuffle_buffer_size=sbs,
            parallel_shard_iters=psi,
        )
        loader = get_savable_loader(ds)
        loader.restore_state_rank(state_2)
        order_8 = [data.text[0] for idx, data in zip(range(n3), loader)]
        print("order3", order_3)
        print("order8", order_8)
        assert order_8 == order_3

    def test_invariance_global_samples(self):
        # We'd like to ensure that the user can keep the same global batches
        # (deterministic pseudo random order) when changing the number of ranks (world size).

        # This can be achieved by obeying a few constraints:
        # - Global batch size must stay the same across runs
        # - Global batch size must be a multiple of (micro-batch size * world_size * num_workers)
        #   - Global batch size = micro-batch size * world_size * num_workers * gradient_accum_steps
        # - world_size * num_workers must stay the same across runs
        # Set the same torch.manual_seed(...) on each rank before constructing the dataset and the data loader

        scenarios = [
            dict(
                configs=(WorkerConfig(rank=0, world_size=1, num_workers=4),),
                micro_batch_size=2,
                global_batch_size=8,
            ),
            dict(
                configs=(
                    WorkerConfig(rank=0, world_size=2, num_workers=2),
                    WorkerConfig(rank=1, world_size=2, num_workers=2),
                ),
                micro_batch_size=2,
                global_batch_size=8,
            ),
            dict(
                configs=(
                    WorkerConfig(rank=0, world_size=4, num_workers=1),
                    WorkerConfig(rank=1, world_size=4, num_workers=1),
                    WorkerConfig(rank=2, world_size=4, num_workers=1),
                    WorkerConfig(rank=3, world_size=4, num_workers=1),
                ),
                micro_batch_size=2,
                global_batch_size=8,
            ),
            dict(
                configs=(
                    WorkerConfig(rank=0, world_size=2, num_workers=2),
                    WorkerConfig(rank=1, world_size=2, num_workers=2),
                ),
                micro_batch_size=1,  # Micro-batch 1, more accum
                global_batch_size=8,
            ),
        ]

        # Constraints to user:

        global_batches_per_scenario = []
        for scenario in scenarios:
            assert scenario["global_batch_size"] % scenario["micro_batch_size"] == 0, (
                "Global batch size must be a multiple of the micro-batch size."
            )

            world_size = len(scenario["configs"])
            gradient_accum_steps = scenario["global_batch_size"] // (
                scenario["micro_batch_size"] * world_size
            )

            batches_per_rank = []

            for rank_config in scenario["configs"]:
                torch.manual_seed(42)
                ds = get_train_dataset(
                    self.dataset_path,
                    split_part="train",
                    sample_type=TextSample,
                    worker_config=rank_config,
                    batch_size=scenario["micro_batch_size"],
                    shuffle_buffer_size=42,
                    max_samples_per_sequence=2,
                )
                loader = get_loader(ds)

                micro_batches = [
                    data.text
                    for idx, data in zip(
                        range(55 * 8 // (world_size * scenario["micro_batch_size"])), loader
                    )
                ]
                batches_per_rank.append(micro_batches)

            # Compose global batches
            global_batches_cur_rank = []
            batch_index = 0
            while batch_index < len(batches_per_rank[0]):
                global_batch = []
                for _ in range(gradient_accum_steps):
                    for rank_batches in batches_per_rank:
                        global_batch.extend(rank_batches[batch_index])
                    batch_index += 1
                    if batch_index >= len(batches_per_rank[0]):
                        # last global batch may be smaller
                        break
                global_batches_cur_rank.append(sorted(global_batch))

            global_batches_per_scenario.append(global_batches_cur_rank)

        # Check that the global batches are the same

        # Assert that all scenarios produced the same number of global batches
        assert all(
            len(global_batches) == len(global_batches_per_scenario[0])
            for global_batches in global_batches_per_scenario
        ), "Number of global batches per scenario does not match."

        for global_batches in global_batches_per_scenario:
            print("= Global batches per scenario")
            for global_batch in global_batches:
                print("  Global batch: ", global_batch)

        # Assert that all global batches are the same
        for i in range(len(global_batches_per_scenario[0])):
            for scenerio_idx, global_batches in enumerate(global_batches_per_scenario):
                assert global_batches[i] == global_batches_per_scenario[0][i], (
                    f"Global batch {i} of scenario {scenerio_idx} does not match."
                )

        # Delete all locals, otherwise loaders might be kept alive
        locals().clear()
        gc.collect()

    def test_redist(self):
        scenarios = [
            dict(
                configs=(
                    WorkerConfig(rank=0, world_size=2, num_workers=2),
                    WorkerConfig(rank=1, world_size=2, num_workers=2),
                ),
                micro_batch_size=2,
                global_batch_size=8,
            ),
            dict(
                configs=(WorkerConfig(rank=0, world_size=1, num_workers=4),),
                micro_batch_size=2,
                global_batch_size=8,
            ),
            dict(
                configs=(
                    WorkerConfig(rank=0, world_size=4, num_workers=1),
                    WorkerConfig(rank=1, world_size=4, num_workers=1),
                    WorkerConfig(rank=2, world_size=4, num_workers=1),
                    WorkerConfig(rank=3, world_size=4, num_workers=1),
                ),
                micro_batch_size=2,
                global_batch_size=8,
            ),
            dict(
                configs=(
                    WorkerConfig(rank=0, world_size=2, num_workers=2),
                    WorkerConfig(rank=1, world_size=2, num_workers=2),
                ),
                micro_batch_size=1,  # Micro-batch 1, more accum
                global_batch_size=8,
            ),
            dict(  # Same as original
                configs=(
                    WorkerConfig(rank=0, world_size=2, num_workers=2),
                    WorkerConfig(rank=1, world_size=2, num_workers=2),
                ),
                micro_batch_size=2,
                global_batch_size=8,
            ),
        ]

        # === Stage 1 first generate a saved state using scenario 0
        checkpoint_files = []

        global_batches_per_scenario = []
        scenario = scenarios[0]

        world_size = len(scenario["configs"])
        gradient_accum_steps = scenario["global_batch_size"] // (
            scenario["micro_batch_size"] * world_size
        )

        batches_per_rank = []

        for rank_config in scenario["configs"]:
            loader = get_savable_loader(
                get_train_dataset(
                    self.dataset_path,
                    split_part="train",
                    sample_type=TextSample,
                    worker_config=rank_config,
                    batch_size=scenario["micro_batch_size"],
                    shuffle_buffer_size=42,
                    max_samples_per_sequence=2,
                )
            )

            # Throw away some samples to advance the loader state
            num_pre_samples = 20
            for _ in zip(range(num_pre_samples), loader):
                pass

            # Save the state to a file
            checkpoint_file = self.checkpoint_dir / f"state_rank{rank_config.rank}.pt"
            state = loader.save_state_rank()
            torch.save(state, str(checkpoint_file))
            checkpoint_files.append(checkpoint_file)

            # Now capture the next micro-batches
            micro_batches = [
                data.text
                for idx, data in zip(
                    range(55 * 8 // (world_size * scenario["micro_batch_size"])), loader
                )
            ]
            batches_per_rank.append(micro_batches)

        # Compose global batches
        global_batches_cur_rank = []
        batch_index = 0
        while batch_index < len(batches_per_rank[0]):
            global_batch = []
            for _ in range(gradient_accum_steps):
                for rank_batches in batches_per_rank:
                    global_batch.extend(rank_batches[batch_index])
                batch_index += 1
                if batch_index >= len(batches_per_rank[0]):
                    # last global batch may be smaller
                    break
            global_batches_cur_rank.append(sorted(global_batch))

        global_batches_per_scenario.append(global_batches_cur_rank)

        # === Stage 2: Now check that the global batches are the same after redistribution

        for scenario in scenarios[1:]:
            # Redistribute the saved state
            runner = CliRunner()
            result = runner.invoke(
                command_redist,
                [
                    "--new-world-size",
                    str(len(scenario["configs"])),
                    *[str(cpt) for cpt in checkpoint_files],
                    str(self.redist_dir),
                ],
            )
            print(result.output)
            assert result.exception is None, result.exception
            assert result.exit_code == 0, "Redistribution failed"

            # Load state and check that the global batches are the same
            assert scenario["global_batch_size"] % scenario["micro_batch_size"] == 0, (
                "Global batch size must be a multiple of the micro-batch size."
            )

            world_size = len(scenario["configs"])
            gradient_accum_steps = scenario["global_batch_size"] // (
                scenario["micro_batch_size"] * world_size
            )

            batches_per_rank = []

            for rank_config in scenario["configs"]:
                loader = get_savable_loader(
                    get_train_dataset(
                        self.dataset_path,
                        split_part="train",
                        sample_type=TextSample,
                        worker_config=rank_config,
                        batch_size=scenario["micro_batch_size"],
                        shuffle_buffer_size=42,
                        max_samples_per_sequence=2,
                    )
                )

                state = torch.load(
                    str(self.redist_dir / f"state_rank{rank_config.rank}.pt"), weights_only=False
                )
                loader.restore_state_rank(state)

                micro_batches = [
                    data.text
                    for idx, data in zip(
                        range(55 * 8 // (world_size * scenario["micro_batch_size"])), loader
                    )
                ]
                batches_per_rank.append(micro_batches)

            # Compose global batches
            global_batches_cur_rank = []
            batch_index = 0
            while batch_index < len(batches_per_rank[0]):
                global_batch = []
                for _ in range(gradient_accum_steps):
                    for rank_batches in batches_per_rank:
                        global_batch.extend(rank_batches[batch_index])
                    batch_index += 1
                    if batch_index >= len(batches_per_rank[0]):
                        # last global batch may be smaller
                        break
                global_batches_cur_rank.append(sorted(global_batch))

            global_batches_per_scenario.append(global_batches_cur_rank)

        # Check that the global batches are the same

        print()

        # Assert that all scenarios produced the same global batches
        assert all(
            len(global_batches) == len(global_batches_per_scenario[0])
            for global_batches in global_batches_per_scenario
        ), "Number of global batches per scenario does not match."

        for global_batches in global_batches_per_scenario:
            print("= Global batches per scenario")
            for global_batch in global_batches:
                print("  Global batch: ", global_batch)

        # Assert that all global batches are the same
        for i in range(len(global_batches_per_scenario[0])):
            for scenerio_idx, global_batches in enumerate(global_batches_per_scenario):
                assert global_batches[i] == global_batches_per_scenario[0][i], (
                    f"Global batch {i} of scenario {scenerio_idx} does not match."
                )

        # Delete all locals, otherwise loaders might be kept alive
        locals().clear()
        gc.collect()


if __name__ == "__main__":
    unittest.main()
