# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for bin-idx (.bin + .idx) datasets."""

import gc
import logging
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
from click.testing import CliRunner

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
from megatron.energon.flavors.binidx.binidx_reader import BinIdxReader, IdxReader
from megatron.energon.recipe.recipe import Recipe
from megatron.energon.tools.prepare import command as prepare_command

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


@edataclass
class BinIdxKeySample(Sample):
    has_tokens: bool


@stateless()
def cook_binidx_key(sample: CrudeSample) -> BinIdxKeySample:
    return BinIdxKeySample(**basic_sample_keys(sample), has_tokens="tokens" in sample)


class BinIdxKeyEncoder(DefaultTaskEncoder):
    cookers = [Cooker(cook=cook_binidx_key, part_filter=lambda part: part != "tokens")]


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

    def test_train_loader_with_workers_close_and_part_filter(self):
        """Full train epochs read the right tokens per sample with and without workers, the
        dataset stays usable after a loader is closed, and filtering out ``tokens`` skips the
        .bin file."""
        num_docs, doc_len = 64, 1024
        bin_path = self.create_binidx_dataset(
            self.dataset_path / "train_loop", num_docs=num_docs, doc_len=doc_len
        )
        bin_file = os.path.realpath(bin_path)

        def bin_file_open():
            return any(
                os.path.realpath(f"/proc/self/fd/{fd}") == bin_file
                for fd in os.listdir("/proc/self/fd")
            )

        def check_epoch(batches):
            docs = []
            for batch in batches:
                doc_idx = int(batch.doc_idx[0])
                first_token = 1 + doc_idx * doc_len
                self.assertEqual(int(batch.first_token[0]), first_token)
                self.assertEqual(
                    int(batch.token_sum[0]), doc_len * first_token + doc_len * (doc_len - 1) // 2
                )
                docs.append(doc_idx)
            self.assertEqual(sorted(docs), list(range(num_docs)))

        for num_workers in (0, 2):
            with self.subTest(num_workers=num_workers):

                def get_dataset(task_encoder):
                    return get_train_dataset(
                        bin_path,
                        worker_config=WorkerConfig(rank=0, world_size=1, num_workers=num_workers),
                        batch_size=1,
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=None,
                        task_encoder=task_encoder,
                        repeat=False,
                    )

                dataset = get_dataset(BinIdxStatsEncoder())
                # The .bin must not be opened before forking, workers would share its offset
                self.assertFalse(bin_file_open())

                # Closing one loader does not break another loader of the same dataset
                loader = get_loader(dataset)
                batches = iter(loader)
                first_batch = next(batches)
                get_loader(dataset).close()
                check_epoch([first_batch, *batches])
                loader.close()
                self.assertFalse(bin_file_open())

                keys = []
                for batch in get_loader(get_dataset(BinIdxKeyEncoder())):
                    self.assertFalse(any(batch.has_tokens))
                    self.assertFalse(bin_file_open())
                    keys.extend(int(key) for key in batch.__key__)
                self.assertEqual(sorted(keys), list(range(num_docs)))

    def test_prepare_cli_skips_binidx(self):
        bin_path = self.create_binidx_dataset(self.dataset_path, num_docs=2, doc_len=3)
        result = CliRunner().invoke(prepare_command, [str(bin_path)], catch_exceptions=False)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("BinIdx datasets need no preparation", result.output)

    def test_reader_lists_readable_parts_and_reopens_after_close(self):
        bin_path = self.create_binidx_dataset(
            self.dataset_path / "reader_contract",
            num_docs=2,
            doc_len=3,
        )
        reader = BinIdxReader(str(bin_path))
        self.assertEqual(len(reader), 2)
        self.assertEqual(
            [part for part, _, _ in reader.list_all_sample_parts()],
            ["0.tokens", "1.tokens"],
        )
        self.assertEqual(
            [part for part, _, _ in reader.list_sample_parts("1")],
            ["1.tokens"],
        )
        for _ in range(2):
            token_bytes, source_info = reader["1.tokens"]
            self.assertEqual(source_info.file_names, ("1.tokens",))
            self.assertEqual(np.frombuffer(token_bytes, dtype=np.int32).tolist(), [4, 5, 6])
            reader.close()

    def test_reader_rejects_truncated_bin_file(self):
        bin_path = self.create_binidx_dataset(
            self.dataset_path / "truncated_bin", num_docs=2, doc_len=3
        )
        with bin_path.open("r+b") as bin_file:
            bin_file.truncate(20)
        for use_mmap in (True, False):
            with self.subTest(use_mmap=use_mmap):
                reader = BinIdxReader(EPath(bin_path))
                if not use_mmap:
                    reader._idx_reader = IdxReader(EPath(bin_path.with_suffix(".idx")))
                try:
                    self.assertEqual(
                        np.frombuffer(reader[0]["tokens"], dtype=np.int32).tolist(), [1, 2, 3]
                    )
                    for key in (1, "1", "1.tokens"):
                        with self.subTest(key=key):
                            with self.assertRaisesRegex(
                                ValueError, "sample 1 expects 12 bytes at offset 12, got 8"
                            ):
                                reader[key]
                finally:
                    reader.close()

    def test_loaded_tokens_are_writable_and_independent(self):
        bin_path = self.create_binidx_dataset(
            self.dataset_path / "writable_tokens", num_docs=2, doc_len=3
        )
        factory = DefaultBinIdxDatasetFactory(
            EPath(bin_path),
            training=False,
            worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
        )
        reader = BinIdxReader(EPath(bin_path))
        try:
            sample = factory.load_sample(reader[0])
            self.assertEqual(sample["tokens"].dtype, np.dtype(np.int32))
            sample["tokens"][0] = 99
            self.assertEqual(sample["tokens"].tolist(), [99, 2, 3])
            reloaded = factory.load_sample(reader[0])
            self.assertEqual(reloaded["tokens"].tolist(), [1, 2, 3])
        finally:
            reader.close()

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

    def test_recipe_single_dataset_reference(self):
        self.create_binidx_dataset(
            self.dataset_path,
            bin_name="tokens.bin",
            num_docs=3,
            doc_len=2,
            token_start=7,
        )
        recipe_path = self.dataset_path / "recipe.yaml"
        with open(recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    path: tokens.bin",
                    ]
                )
            )

        recipe = load_config(
            EPath(recipe_path),
            default_type=Recipe,
            default_kwargs=dict(path=EPath(recipe_path)),
        )
        recipe.post_initialize()
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )
        inner = recipe.get_datasets(
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
