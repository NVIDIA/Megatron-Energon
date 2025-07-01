# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for the dataset."""

import gc
import logging
import sys
import tempfile
import unittest
import warnings
from datetime import timedelta
from multiprocessing import Manager, Process
from pathlib import Path

import torch
import torch.distributed as dist
import webdataset as wds

from megatron.energon import (
    DefaultTaskEncoder,
    SkipSample,
    TextSample,
    WorkerConfig,
    get_savable_loader,
    get_train_dataset,
    stateless,
)
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from megatron.energon.sync_end_of_dataset import RedistributeLoader, StopFirstLoader

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

    def test_distribute_stop_first(self):
        world_size = 3

        offsets = [9, 0, 4]

        all_items = [f"{i}" for i in range(55)]
        rank_subsets = [
            set(all_items[:19]),
            set(all_items[19 : 19 + 18]),
            set(all_items[19 + 18 :]),
        ]

        def phase1(rank: int, world_size: int, shared_dict: dict):
            worker_config = WorkerConfig(rank=rank, world_size=world_size, num_workers=0)

            torch.manual_seed(42)

            epoch_reset = False
            epoch_offset = 0

            class LocalTaskEncoder(
                DefaultTaskEncoder[TextSample, TextSample, TextSample, TextSample]
            ):
                @stateless
                def encode_sample(self, sample: TextSample) -> TextSample:
                    nonlocal epoch_reset, epoch_offset
                    if epoch_reset:
                        epoch_offset = self.current_batch_index
                        epoch_reset = False
                    if self.current_batch_index >= 5 + offsets[rank] + epoch_offset:
                        print(
                            f"[r={rank}] Skip sample bi={self.current_batch_index} si={self.current_sample_index}\n",
                            end="",
                        )
                        raise SkipSample()
                    print(
                        f"[r={rank}] Return sample bi={self.current_batch_index} si={self.current_sample_index}\n",
                        end="",
                    )
                    return sample

            # First verify that the loader is working as expected
            ref_loader = get_savable_loader(
                get_train_dataset(
                    self.dataset_path,
                    split_part="train",
                    sample_type=TextSample,
                    worker_config=worker_config,
                    batch_size=1,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=2,
                    parallel_shard_iters=1,
                    repeat=False,
                    task_encoder=LocalTaskEncoder(),
                ),
                checkpoint_every_sec=0,
                checkpoint_every_min_n_samples=1,
            )

            order_all = [data.text[0] for idx, data in zip(range(100), ref_loader)]
            assert len(order_all) == 5 + offsets[rank], f"Rank {rank} has {len(order_all)} samples"
            assert all(item in rank_subsets[rank] for item in order_all), (
                f"Rank {rank} has {order_all} samples"
            )

            epoch_reset = True

            order_all = [data.text[0] for idx, data in zip(range(100), ref_loader)]
            assert len(order_all) == 5 + offsets[rank], f"Rank {rank} has {len(order_all)} samples"
            assert all(item in rank_subsets[rank] for item in order_all), (
                f"Rank {rank} has {order_all} samples"
            )

            epoch_offset = 0

            # To the actual test
            loader = StopFirstLoader(
                get_savable_loader(
                    get_train_dataset(
                        self.dataset_path,
                        split_part="train",
                        sample_type=TextSample,
                        worker_config=worker_config,
                        batch_size=1,
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=2,
                        parallel_shard_iters=1,
                        repeat=False,
                        task_encoder=LocalTaskEncoder(),
                    ),
                    checkpoint_every_sec=0,
                    checkpoint_every_min_n_samples=1,
                )
            )

            state_0 = loader.save_state_global(global_dst_rank=0)
            order_1 = [data.text[0] for idx, data in zip(range(4), loader)]
            assert len(order_1) == 4, f"Rank {rank} has {len(order_1)} samples"
            assert all(item in rank_subsets[rank] for item in order_1), (
                f"Rank {rank} has {order_1} samples"
            )

            print(f"Rank {rank} order_1: {order_1}\n", end="")

            # Exhaust the synchronized loader. It should only give a single sample.
            state_1 = loader.save_state_global(global_dst_rank=0)
            order_2 = [data.text[0] for idx, data in zip(range(10), loader)]
            assert len(order_2) == 1, f"Rank {rank} has {len(order_2)} samples"
            assert all(item in rank_subsets[rank] for item in order_2), (
                f"Rank {rank} has {order_2} samples"
            )

            print(f"Rank {rank} order_2: {order_2}\n", end="")

            # Restart iterating until exhausted.
            epoch_reset = True

            state_2 = loader.save_state_global(global_dst_rank=0)
            order_3 = [data.text[0] for idx, data in zip(range(30), loader)]
            assert len(order_3) == 5, f"Rank {rank} has {len(order_3)} samples"

            print(f"Rank {rank} order_3: {order_3}\n", end="")

            assert all(item in rank_subsets[rank] for item in order_3), (
                f"Rank {rank} has {order_3} samples"
            )

            shared_dict[(rank, "order_1")] = order_1
            shared_dict[(rank, "order_2")] = order_2
            shared_dict[(rank, "order_3")] = order_3

            if rank == 0:
                shared_dict["state_0"] = state_0
                shared_dict["state_1"] = state_1
                shared_dict["state_2"] = state_2

            print(f"Rank {rank} finished phase 1\n", end="")
            dist.barrier()

        def phase2(rank: int, world_size: int, shared_dict: dict):
            torch.manual_seed(213)

            epoch_reset = False
            epoch_offset = 0

            class LocalTaskEncoder(
                DefaultTaskEncoder[TextSample, TextSample, TextSample, TextSample]
            ):
                @stateless
                def encode_sample(self, sample: TextSample) -> TextSample:
                    nonlocal epoch_reset, epoch_offset
                    if epoch_reset:
                        epoch_offset = self.current_batch_index
                        epoch_reset = False
                    if self.current_batch_index >= 5 + offsets[rank] + epoch_offset:
                        raise SkipSample()
                    return sample

            order_1 = shared_dict[(rank, "order_1")]
            order_2 = shared_dict[(rank, "order_2")]
            order_3 = shared_dict[(rank, "order_3")]

            if rank == 0:
                state_0 = shared_dict["state_0"]
                state_1 = shared_dict["state_1"]
                state_2 = shared_dict["state_2"]
            else:
                state_0 = None
                state_1 = None
                state_2 = None

            worker_config = WorkerConfig(rank=rank, world_size=world_size, num_workers=0)

            loader = StopFirstLoader(
                get_savable_loader(
                    get_train_dataset(
                        self.dataset_path,
                        split_part="train",
                        sample_type=TextSample,
                        worker_config=worker_config,
                        batch_size=1,
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=2,
                        parallel_shard_iters=1,
                        repeat=False,
                        task_encoder=LocalTaskEncoder(),
                    ),
                    checkpoint_every_sec=0,
                    checkpoint_every_min_n_samples=1,
                )
            )
            loader.restore_state_global(state_0, src_rank=0)

            order_1_r = [data.text[0] for idx, data in zip(range(4), loader)]
            assert order_1_r == order_1

            order_2_r = [data.text[0] for idx, data in zip(range(10), loader)]
            assert order_2_r == order_2

            loader = StopFirstLoader(
                get_savable_loader(
                    get_train_dataset(
                        self.dataset_path,
                        split_part="train",
                        sample_type=TextSample,
                        worker_config=worker_config,
                        batch_size=1,
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=2,
                        parallel_shard_iters=1,
                        repeat=False,
                        task_encoder=LocalTaskEncoder(),
                    ),
                    checkpoint_every_sec=0,
                    checkpoint_every_min_n_samples=1,
                )
            )
            loader.restore_state_global(state_1, src_rank=0)

            order_2_r = [data.text[0] for idx, data in zip(range(10), loader)]
            assert order_2_r == order_2

            epoch_reset = True

            order_3_r = [data.text[0] for idx, data in zip(range(30), loader)]
            assert order_3_r == order_3

            loader = StopFirstLoader(
                get_savable_loader(
                    get_train_dataset(
                        self.dataset_path,
                        split_part="train",
                        sample_type=TextSample,
                        worker_config=worker_config,
                        batch_size=1,
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=2,
                        parallel_shard_iters=1,
                        repeat=False,
                        task_encoder=LocalTaskEncoder(),
                    ),
                    checkpoint_every_sec=0,
                    checkpoint_every_min_n_samples=1,
                )
            )
            loader.restore_state_global(state_2, src_rank=0)

            epoch_reset = True

            order_3_r = [data.text[0] for idx, data in zip(range(30), loader)]
            assert order_3_r == order_3

            dist.barrier()
            print(f"Rank {rank} finished phase 2\n", end="")

        def init_process(rank, world_size, shared_dict, fn, backend="gloo"):
            """Initializes the distributed environment."""
            dist.init_process_group(
                backend=backend,
                init_method="tcp://127.0.0.1:12355",
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=5),
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
                assert p.exitcode == 0

            # Phase 2 (restore state)
            processes = []
            for rank in range(world_size):
                p = Process(target=init_process, args=(rank, world_size, shared_dict, phase2))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
                assert p.exitcode == 0

    def test_distribute_stop_redistribute(self):
        world_size = 3

        offsets = [9, 0, 4]

        all_items = [f"{i}" for i in range(55)]
        rank_subsets = [
            set(all_items[:19]),
            set(all_items[19 : 19 + 18]),
            set(all_items[19 + 18 :]),
        ]

        def phase1(rank: int, world_size: int, shared_dict: dict):
            worker_config = WorkerConfig(rank=rank, world_size=world_size, num_workers=0)

            torch.manual_seed(42)

            epoch_reset = False
            epoch_offset = 0

            class LocalTaskEncoder(
                DefaultTaskEncoder[TextSample, TextSample, TextSample, TextSample]
            ):
                @stateless
                def encode_sample(self, sample: TextSample) -> TextSample:
                    nonlocal epoch_reset, epoch_offset
                    if epoch_reset:
                        epoch_offset = self.current_batch_index
                        epoch_reset = False
                    if self.current_batch_index >= 5 + offsets[rank] + epoch_offset:
                        # print(f"[r={rank}] Skip sample bi={self.current_batch_index} si={self.current_sample_index}\n", end="")
                        raise SkipSample()
                    # print(f"[r={rank}] Return sample bi={self.current_batch_index} si={self.current_sample_index}\n", end="")
                    return sample

            # This seed is used by the dataset to shuffle the data

            ref_loader = get_savable_loader(
                get_train_dataset(
                    self.dataset_path,
                    split_part="train",
                    sample_type=TextSample,
                    worker_config=worker_config,
                    batch_size=1,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=2,
                    parallel_shard_iters=1,
                    repeat=False,
                    task_encoder=LocalTaskEncoder(),
                ),
                checkpoint_every_sec=0,
                checkpoint_every_min_n_samples=1,
            )

            order_all = [data.text[0] for idx, data in zip(range(100), ref_loader)]
            assert len(order_all) == 5 + offsets[rank], f"Rank {rank} has {len(order_all)} samples"
            assert all(item in rank_subsets[rank] for item in order_all), (
                f"Rank {rank} has {order_all} samples"
            )

            epoch_reset = True

            order_all = [data.text[0] for idx, data in zip(range(100), ref_loader)]
            assert len(order_all) == 5 + offsets[rank], f"Rank {rank} has {len(order_all)} samples"
            assert all(item in rank_subsets[rank] for item in order_all), (
                f"Rank {rank} has {order_all} samples"
            )

            epoch_offset = 0

            loader = RedistributeLoader(
                get_savable_loader(
                    get_train_dataset(
                        self.dataset_path,
                        split_part="train",
                        sample_type=TextSample,
                        worker_config=worker_config,
                        batch_size=1,
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=2,
                        parallel_shard_iters=1,
                        repeat=False,
                        task_encoder=LocalTaskEncoder(),
                    ),
                    checkpoint_every_sec=0,
                    checkpoint_every_min_n_samples=1,
                )
            )

            state_0 = loader.save_state_global(global_dst_rank=0)
            order_1 = [data.text[0] for idx, data in zip(range(5), loader)]
            assert len(order_1) == 5, f"Rank {rank} has {len(order_1)} samples"
            assert all(item in rank_subsets[rank] for item in order_1), (
                f"Rank {rank} has {order_1} samples"
            )

            # print(f"Rank {rank}: order_1", order_1)

            state_1 = loader.save_state_global(global_dst_rank=0)
            order_2 = [data.text[0] for idx, data in zip(range(10), loader)]
            assert len(order_2) == 4, f"Rank {rank} has {len(order_2)} samples"
            # States:
            # [-1] remaining samples [9, 0, 4]
            # [0] remaining samples [7, 0, 3]
            # [1] remaining samples [6, 0, 1]
            # [2] remaining samples [4, 0, 0]
            # [3] remaining samples [1, 0, 0]
            # End, not enough samples left
            if rank == 0:
                assert all(item in rank_subsets[0] for item in order_2), (
                    f"Rank {rank} has {order_2} samples"
                )
            elif rank == 1:
                assert order_2[0] in rank_subsets[0]
                assert order_2[1] in rank_subsets[2]
                assert order_2[2] in rank_subsets[0]
                assert order_2[3] in rank_subsets[0]
            elif rank == 2:
                assert order_2[0] in rank_subsets[2]
                assert order_2[1] in rank_subsets[2]
                assert order_2[2] in rank_subsets[2]
                assert order_2[3] in rank_subsets[0]

            epoch_reset = True

            state_2 = loader.save_state_global(global_dst_rank=0)
            order_3 = [data.text[0] for idx, data in zip(range(30), loader)]
            assert len(order_3) == 9, f"Rank {rank} has {len(order_3)} samples"

            assert all(item in rank_subsets[rank] for item in order_3[:5]), (
                f"Rank {rank} has {order_3} samples"
            )
            if rank == 0:
                assert all(item in rank_subsets[0] for item in order_3[5:]), (
                    f"Rank {rank} has {order_3} samples"
                )
            elif rank == 1:
                assert order_3[5] in rank_subsets[0]
                assert order_3[6] in rank_subsets[2]
                assert order_3[7] in rank_subsets[0]
                assert order_3[8] in rank_subsets[0]
            elif rank == 2:
                assert order_3[5] in rank_subsets[2]
                assert order_3[6] in rank_subsets[2]
                assert order_3[7] in rank_subsets[2]
                assert order_3[8] in rank_subsets[0]

            shared_dict[(rank, "order_1")] = order_1
            shared_dict[(rank, "order_2")] = order_2
            shared_dict[(rank, "order_3")] = order_3

            if rank == 0:
                shared_dict["state_0"] = state_0
                shared_dict["state_1"] = state_1
                shared_dict["state_2"] = state_2

            print(f"Rank {rank} finished phase 1\n", end="")
            dist.barrier()

        def phase2(rank: int, world_size: int, shared_dict: dict):
            torch.manual_seed(213)

            epoch_reset = False
            epoch_offset = 0

            class LocalTaskEncoder(
                DefaultTaskEncoder[TextSample, TextSample, TextSample, TextSample]
            ):
                @stateless
                def encode_sample(self, sample: TextSample) -> TextSample:
                    nonlocal epoch_reset, epoch_offset
                    if epoch_reset:
                        epoch_offset = self.current_batch_index
                        epoch_reset = False
                    if self.current_batch_index >= 5 + offsets[rank] + epoch_offset:
                        raise SkipSample()
                    return sample

            order_1 = shared_dict[(rank, "order_1")]
            order_2 = shared_dict[(rank, "order_2")]
            order_3 = shared_dict[(rank, "order_3")]

            if rank == 0:
                state_0 = shared_dict["state_0"]
                state_1 = shared_dict["state_1"]
                state_2 = shared_dict["state_2"]
            else:
                state_0 = None
                state_1 = None
                state_2 = None

            worker_config = WorkerConfig(rank=rank, world_size=world_size, num_workers=0)

            loader = RedistributeLoader(
                get_savable_loader(
                    get_train_dataset(
                        self.dataset_path,
                        split_part="train",
                        sample_type=TextSample,
                        worker_config=worker_config,
                        batch_size=1,
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=2,
                        parallel_shard_iters=1,
                        repeat=False,
                        task_encoder=LocalTaskEncoder(),
                    ),
                    checkpoint_every_sec=0,
                    checkpoint_every_min_n_samples=1,
                )
            )
            loader.restore_state_global(state_0, src_rank=0)

            order_1_r = [data.text[0] for idx, data in zip(range(5), loader)]
            assert order_1_r == order_1

            order_2_r = [data.text[0] for idx, data in zip(range(10), loader)]
            assert order_2_r == order_2

            loader = RedistributeLoader(
                get_savable_loader(
                    get_train_dataset(
                        self.dataset_path,
                        split_part="train",
                        sample_type=TextSample,
                        worker_config=worker_config,
                        batch_size=1,
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=2,
                        parallel_shard_iters=1,
                        repeat=False,
                        task_encoder=LocalTaskEncoder(),
                    ),
                    checkpoint_every_sec=0,
                    checkpoint_every_min_n_samples=1,
                )
            )
            loader.restore_state_global(state_1, src_rank=0)

            order_2_r = [data.text[0] for idx, data in zip(range(10), loader)]
            assert order_2_r == order_2

            epoch_reset = True

            order_3_r = [data.text[0] for idx, data in zip(range(30), loader)]
            assert order_3_r == order_3

            loader = RedistributeLoader(
                get_savable_loader(
                    get_train_dataset(
                        self.dataset_path,
                        split_part="train",
                        sample_type=TextSample,
                        worker_config=worker_config,
                        batch_size=1,
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=2,
                        parallel_shard_iters=1,
                        repeat=False,
                        task_encoder=LocalTaskEncoder(),
                    ),
                    checkpoint_every_sec=0,
                    checkpoint_every_min_n_samples=1,
                )
            )
            loader.restore_state_global(state_2, src_rank=0)

            epoch_reset = True

            order_3_r = [data.text[0] for idx, data in zip(range(30), loader)]
            assert order_3_r == order_3

            dist.barrier()
            print(f"Rank {rank} finished phase 2\n", end="")

        def init_process(rank, world_size, shared_dict, fn, backend="gloo"):
            """Initializes the distributed environment."""
            dist.init_process_group(
                backend=backend,
                init_method="tcp://127.0.0.1:12355",
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=5),
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
                assert p.exitcode == 0

            # Phase 2 (restore state)
            processes = []
            for rank in range(world_size):
                p = Process(target=init_process, args=(rank, world_size, shared_dict, phase2))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
                assert p.exitcode == 0


if __name__ == "__main__":
    unittest.main()
