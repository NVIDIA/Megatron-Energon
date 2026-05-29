# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for bin-idx (.bin + .idx) datasets."""

import gc
import logging
import random
import struct
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch

from megatron.energon import (
    Cooker,
    CrudeSample,
    DefaultBinIdxDatasetFactory,
    DefaultTaskEncoder,
    WorkerConfig,
    basic_sample_keys,
    edataclass,
    get_loader,
    get_train_dataset,
    get_val_dataset,
    stateless,
)
from megatron.energon.dataset_config import load_config
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import Sample
from megatron.energon.metadataset.metadataset_v2 import MetadatasetV2

# Speed up tests significantly by reducing the torch status check interval for broken worker shutdown
try:
    torch.utils.data._utils.worker.MP_STATUS_CHECK_INTERVAL = 0.1
    torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 0.1
except AttributeError:
    pass


@edataclass
class BinIdxStatsSample(Sample):
    token_sum: int
    seq_len: int
    doc_idx: int
    first_token: int


@stateless()
def cook_binidx_stats(sample: CrudeSample) -> BinIdxStatsSample:
    tokens = sample["tokens"]
    assert isinstance(tokens, np.ndarray)
    return BinIdxStatsSample(
        **basic_sample_keys(sample),
        token_sum=int(tokens.sum()),
        seq_len=int(tokens.shape[0]),
        doc_idx=int(sample["__key__"]),
        first_token=int(tokens[0]),
    )


class BinIdxStatsEncoder(DefaultTaskEncoder):
    cookers = [Cooker(cook=cook_binidx_stats)]


class TestBinIdxDataset(unittest.TestCase):
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

    def tearDown(self):
        # Remove all temporary files
        gc.collect()
        self.temp_dir.cleanup()

    @staticmethod
    def create_binidx_dataset(
        directory: Path,
        *,
        bin_name: str = "data.bin",
        num_docs: int,
        doc_len: int,
        token_start: int = 1,
    ) -> Path:
        """Creates a small dummy bin-idx pair for testing purposes."""

        directory.mkdir(parents=True, exist_ok=True)
        bin_path = directory / bin_name
        idx_path = bin_path.with_suffix(".idx")

        values = np.arange(token_start, token_start + num_docs * doc_len, dtype=np.int32)
        values.tofile(str(bin_path))

        with open(idx_path, "wb") as f:
            f.write(b"MMIDIDX\x00\x00")
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<B", 4))
            f.write(struct.pack("<Q", num_docs))
            f.write(struct.pack("<Q", num_docs))
            lengths = np.full(num_docs, doc_len, dtype=np.int32)
            f.write(lengths.tobytes())
            pointers = np.arange(num_docs, dtype=np.int64) * doc_len * 4
            f.write(pointers.tobytes())
            doc_indices = np.arange(num_docs, dtype=np.int64)
            f.write(doc_indices.tobytes())

        return bin_path

    def test_get_train_dataset_iterates_expected_tokens(self):
        torch.manual_seed(42)
        bin_path = self.create_binidx_dataset(
            self.dataset_path / "train_loop",
            num_docs=5,
            doc_len=4,
            token_start=1,
        )
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        train_dataset = get_train_dataset(
            bin_path,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            task_encoder=BinIdxStatsEncoder(),
        )
        train_loader = get_loader(train_dataset)
        expected_sums = [sum(range(i * 4 + 1, (i + 1) * 4 + 1)) for i in range(5)]
        seen = []
        for _ in range(5):
            batch = next(iter(train_loader))
            seen.append(int(batch.token_sum[0]))
            self.assertEqual(int(batch.seq_len[0]), 4)
        self.assertEqual(sorted(seen), sorted(expected_sums))

    def test_get_val_dataset_full_iteration_stable_order(self):
        """Val loader over bin-idx yields every document once in index order (no shuffle)."""
        torch.manual_seed(42)
        num_docs = 9
        doc_len = 2
        token_start = 5
        bin_path = self.create_binidx_dataset(
            self.dataset_path / "val_order",
            num_docs=num_docs,
            doc_len=doc_len,
            token_start=token_start,
        )
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )
        val_dataset = get_val_dataset(
            bin_path,
            worker_config=worker_config,
            batch_size=1,
            split_part="train",
            task_encoder=BinIdxStatsEncoder(),
        )
        val_loader = get_loader(val_dataset)

        order: list[int] = []
        for batch in val_loader:
            self.assertEqual(int(batch.seq_len[0]), doc_len)
            order.append(int(batch.doc_idx[0]))
            doc_i = int(batch.doc_idx[0])
            expected_first = token_start + doc_i * doc_len
            self.assertEqual(int(batch.first_token[0]), expected_first)
            chunk = np.arange(
                expected_first,
                expected_first + doc_len,
                dtype=np.int64,
            )
            self.assertEqual(int(batch.token_sum[0]), int(chunk.sum()))

        self.assertEqual(order, list(range(num_docs)))
        self.assertEqual(len(val_loader), num_docs)

    def test_metadataset_v2_single_dataset_reference(self):
        self.create_binidx_dataset(
            self.dataset_path,
            bin_name="tokens.bin",
            num_docs=3,
            doc_len=2,
            token_start=7,
        )
        mds_path = self.dataset_path / "mds.yaml"
        with open(mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: MetadatasetV2",
                        "splits:",
                        "  train:",
                        "    path: tokens.bin",
                    ]
                )
            )

        mds = load_config(
            EPath(mds_path),
            default_type=MetadatasetV2,
            default_kwargs=dict(path=EPath(mds_path)),
        )
        mds.post_initialize()
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )
        inner = mds.get_datasets(
            training=False,
            split_part="train",
            worker_config=worker_config,
        )
        self.assertEqual(len(inner.datasets), 1)
        factory = inner.datasets[0].dataset
        self.assertIsInstance(factory, DefaultBinIdxDatasetFactory)
        self.assertEqual(len(factory), 3)


if __name__ == "__main__":
    unittest.main()
