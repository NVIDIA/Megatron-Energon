# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that the Energon dataloader works with the ``forkserver`` multiprocessing
start method (instead of the default ``fork``). This exercises the full pickle path
of the dataset + worker_config to the worker processes and verifies that saving and
restoring progress still works.
"""

import gc
import tempfile
import unittest
import warnings
from pathlib import Path

import torch
import webdataset as wds

from megatron.energon import (
    DefaultTaskEncoder,
    TextSample,
    WorkerConfig,
    get_savable_loader,
    get_train_dataset,
)
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME

# Speed up tests significantly by reducing the torch status check interval for broken worker shutdown
try:
    torch.utils.data._utils.worker.MP_STATUS_CHECK_INTERVAL = 0.1
except AttributeError:
    pass


class TestForkserver(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name)
        self.dataset_path.mkdir(exist_ok=True, parents=True)
        self.create_text_test_dataset(self.dataset_path)

    def tearDown(self):
        gc.collect()
        self.temp_dir.cleanup()

    @staticmethod
    def create_text_test_dataset(path: Path):
        (path / "parts").mkdir(exist_ok=True, parents=True)
        with wds.ShardWriter(f"{path}/parts/data-%d.tar", maxcount=100) as shard_writer:
            for idx in range(55):
                shard_writer.write(
                    {
                        "__key__": f"{idx:06d}",
                        "txt": f"{idx}".encode(),
                    }
                )
                if idx in (1, 3, 6, 10, 20, 30, 40, 50):
                    shard_writer.next_stream()
        total_shards = len(list((path / "parts").glob("data-*.tar")))

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

    def _make_loader(self, num_workers, mp_ctx, seed_offset=0):
        worker_config = WorkerConfig(
            rank=0, world_size=1, num_workers=num_workers, seed_offset=seed_offset
        )
        ds = get_train_dataset(
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=42,
            max_samples_per_sequence=2,
            task_encoder=DefaultTaskEncoder(),
        )
        return get_savable_loader(ds, multiprocessing_context=mp_ctx)

    def test_forkserver_save_restore(self):
        num_workers = 2

        # First loader: consume a few samples, save state, then consume more.
        loader1a = self._make_loader(num_workers, "forkserver")
        data_pre = [data.text[0] for idx, data in zip(range(7), loader1a)]
        state = loader1a.save_state_rank()
        data_post = [data.text[0] for idx, data in zip(range(20), loader1a)]
        assert len(data_pre) == 7
        assert len(data_post) == 20

        # Second loader: restore from the saved state and verify it continues identically.
        loader1b = self._make_loader(num_workers, "forkserver")
        loader1b.restore_state_rank(state)
        data_restored = [data.text[0] for idx, data in zip(range(20), loader1b)]

        self.assertEqual(data_post, data_restored)

    def test_forkserver_matches_fork_order(self):
        # For the same seed, forkserver should yield the same sequence as fork
        # (the start method must not affect the data ordering).
        num_workers = 2

        fork_loader = self._make_loader(num_workers, "fork", seed_offset=42)
        fork_order = [data.text[0] for idx, data in zip(range(55 * 5), fork_loader)]

        forkserver_loader = self._make_loader(num_workers, "forkserver", seed_offset=42)
        forkserver_order = [
            data.text[0] for idx, data in zip(range(55 * 5), forkserver_loader)
        ]

        self.assertEqual(fork_order, forkserver_order)


if __name__ == "__main__":
    unittest.main()
