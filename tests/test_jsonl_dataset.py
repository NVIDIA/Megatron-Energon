# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for recipes."""

import gc
import json
import logging
import random
import sys
import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from click.testing import CliRunner

from megatron.energon import (
    Cooker,
    CrudeSample,
    DefaultCrudeJsonlShardListDatasetFactory,
    DefaultTaskEncoder,
    Sample,
    WorkerConfig,
    basic_sample_keys,
    edataclass,
    get_loader,
    get_train_dataset,
    stateless,
)
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.common.manifest.types import ManifestSplits as WebdatasetSplits
from megatron.energon.flavors.dataset_type import EnergonDatasetType, get_dataset_type
from megatron.energon.flavors.jsonl.ijsonl import (
    CachedIJsonlOffsetReader,
    IJsonlFile,
    IJsonlIndexReader,
    IJsonlIndexWriter,
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


class TestJsonlDataset(unittest.TestCase):
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

        # Create a small dummy datasets
        self.create_text_test_dataset(self.dataset_path / "ds1.jsonl", range(55), range(55))
        self.create_text_test_dataset(
            self.dataset_path / "ds2.jsonl", range(100, 155), range(100, 155)
        )
        self.create_text_test_dataset(self.dataset_path / "ds3.jsonl", range(200, 255), range(55))

        self.mds_all_path = self.dataset_path / "recipe_all.yaml"
        with open(self.mds_all_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
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

    def tearDown(self):
        # Remove all temporary files
        gc.collect()
        self.temp_dir.cleanup()

    def test_jsonl_offset_index_round_trips_custom_metadata_suffix(self):
        metadata_path = self.dataset_path / "chunks_00000.metadata"
        metadata_path.write_bytes(
            b"[]\n"
            b'[{"img_id_in_doc":0,"pos_in_text":4}]\n'
            b'[{"img_id_in_doc":1,"pos_in_text":8},{"img_id_in_doc":0,"pos_in_text":8}]\n'
        )

        count = 0
        offset = 0
        with IJsonlIndexWriter(EPath(metadata_path), index_suffix=".metadata.idx") as writer:
            with EPath(metadata_path).open("rb") as metadata_file:
                while True:
                    line = metadata_file.readline()
                    if not line:
                        break
                    writer.append(offset)
                    offset = metadata_file.tell()
                    count += 1
            writer.append(offset)

        offset_reader = CachedIJsonlOffsetReader(
            EPath(metadata_path),
            index_suffix=".metadata.idx",
        )
        expected_lines = metadata_path.read_bytes().splitlines(keepends=True)
        try:
            with IJsonlFile(EPath(metadata_path).open("rb")) as metadata_file:
                indexed_lines = []
                for index in range(count):
                    offset, size = offset_reader.get_ijsonl_byte_offset(index)
                    indexed_lines.append(metadata_file.next(offset, size))

            assert indexed_lines == expected_lines
            assert len(offset_reader) == len(expected_lines)
            assert offset_reader.get_total_size() == metadata_path.stat().st_size
        finally:
            offset_reader.close()

        jsonl_path = self.dataset_path / "samples.jsonl"
        jsonl_path.write_bytes(b'{"idx":0}\n{"idx":1}\n')
        offset = 0
        with IJsonlIndexWriter(EPath(jsonl_path)) as writer:
            with EPath(jsonl_path).open("rb") as jsonl_file:
                while True:
                    line = jsonl_file.readline()
                    if not line:
                        break
                    writer.append(offset)
                    offset = jsonl_file.tell()
            writer.append(offset)

        assert jsonl_path.with_suffix(".jsonl.idx").is_file()
        assert IJsonlIndexReader.count_samples(EPath(jsonl_path)) == 2

    @staticmethod
    def create_text_test_dataset(
        path: Path,
        txt_range: Iterable[int],
        key_range: Iterable[int],
        prefix: str = "",
        *,
        prepare: bool = True,
    ):
        """Creates a small dummy test dataset for testing purposes."""

        # Write jsonl file
        with open(path, "w") as wf:
            for key, txt in zip(key_range, txt_range):
                # Write JSON entries to the file, one per line.
                wf.write(json.dumps({"idx": key, "txt": f"{prefix}{txt}"}) + "\n")

        if prepare:
            from megatron.energon.flavors import CrudeJsonlDatasetFactory

            CrudeJsonlDatasetFactory.prepare_dataset(path)

    def prepare_jsonl_shard_dir(self, shard_dir: Path, split_ratio: str = "1,0,0"):
        runner = CliRunner()
        result = runner.invoke(
            prepare_command,
            [str(shard_dir), "--split-ratio", split_ratio, "--non-interactive", "--no-progress"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.stdout
        assert "Done" in result.stdout, result.stdout
        return result

    def test_dataset(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Train mode dataset
        train_dataset = get_train_dataset(
            self.dataset_path / "ds1.jsonl",
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            task_encoder=SimpleCookingTaskEncoder(),
        )
        print(len(train_dataset))
        assert len(train_dataset) == 55, f"Expected 55 samples, got {len(train_dataset)}"

        train_loader1 = get_loader(train_dataset)

        train_order1 = [
            text for idx, data in zip(range(55 * 10), train_loader1) for text in data.text
        ]
        print(train_order1[:10])
        print(Counter(train_order1))
        assert len(Counter(train_order1)) == 55
        assert all(v == 10 for v in Counter(train_order1).values())

    def test_recipe_all(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=2,
            seed_offset=42,
        )

        # Train mode dataset
        train_dataset = get_train_dataset(
            self.mds_all_path,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            task_encoder=SimpleCookingTaskEncoder(),
        )
        print(len(train_dataset))
        assert len(train_dataset) == 55 * 3, f"Expected 55 * 3 samples, got {len(train_dataset)}"

        train_loader1 = get_loader(train_dataset)

        train_order1 = [
            text for idx, data in zip(range(55 * 10), train_loader1) for text in data.text
        ]
        print(train_order1[:10])
        print(Counter(train_order1))
        assert len(Counter(train_order1)) == 55 * 3
        assert all(2 <= v <= 5 for v in Counter(train_order1).values())

    def test_recipe_multirank(self):
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
                self.dataset_path / "ds1.jsonl",
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

            train_loader1 = get_loader(train_dataset)

            for data in train_loader1:
                sample_counts[int(data.text[0])] += 1

        for i in range(55):
            assert sample_counts[i] == 1, (
                f"Sample {i} should have been seen exactly once, but was seen {sample_counts[i]} times."
            )

    def test_s3(self):
        # Create a joined dataset configuration
        mixed_recipe_path = self.dataset_path / "recipe_mixed.yaml"
        with open(mixed_recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    path: msc://s3test_jsonl_dataset/test/dataset/recipe_all.yaml",
                    ]
                )
            )

        with setup_s3_emulator(profile_name="s3test_jsonl_dataset") as emu:
            # Upload the dataset to the S3 emulator
            # EPath(self.dataset_path).copy(EPath("msc://s3/test/dataset"))
            emu.add_file(self.dataset_path, "test/dataset")

            train_dataset = get_loader(
                get_train_dataset(
                    mixed_recipe_path,
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
            )

            data = list(enumerate(train_dataset))
            assert len(data) == 55 * 10, len(data)
            cnt = Counter(t for _, entry in data for t in entry.text)
            assert len(cnt) == 55 * 3
            assert all(2 <= v <= 5 for v in cnt.values())

    def test_prepare(self):
        print("Creating new dataset")
        with open(self.dataset_path / "ds_prep.jsonl", "w") as f:
            for i in range(10):
                f.write(json.dumps({"idx": i, "txt": f"{i}"}) + "\n\n")

        runner = CliRunner()
        result = runner.invoke(
            prepare_command,
            [str(self.dataset_path / "ds_prep.jsonl")],
            catch_exceptions=False,
        )
        print(result.stdout)
        assert result.exit_code == 0, "Prepare failed, see output"
        assert "Done" in result.stdout, "Prepare failed, see output"
        assert "Found 10 samples" in result.stdout, "Prepare failed, see output"
        assert (self.dataset_path / "ds_prep.jsonl.idx").exists()

        torch.manual_seed(42)

        # Train mode dataset
        train_loader = get_loader(
            get_train_dataset(
                self.dataset_path / "ds_prep.jsonl",
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
        )
        assert len(train_loader) == 10, f"Expected 10 samples, got {len(train_loader)}"

        train_order1 = [text for _, data in zip(range(50), train_loader) for text in data.text]
        print(train_order1[:10])
        print(Counter(train_order1))
        assert len(Counter(train_order1)) == 10
        assert all(v == 5 for v in Counter(train_order1).values())

    def test_prepared_jsonl_shard_directory(self):
        shard_dir = self.dataset_path / "jsonl_shards"
        shard_dir.mkdir()
        self.create_text_test_dataset(
            shard_dir / "shard_0.jsonl", range(0, 3), range(0, 3), prepare=False
        )
        self.create_text_test_dataset(
            shard_dir / "shard_1.jsonl", range(3, 7), range(3, 7), prepare=False
        )

        self.prepare_jsonl_shard_dir(shard_dir)
        assert (shard_dir / ".nv-meta" / ".info.json").exists()
        with open(shard_dir / ".nv-meta" / ".info.json") as f:
            assert "dataset_type" not in json.load(f)
        assert get_dataset_type(EPath(shard_dir)) == EnergonDatasetType.MANIFEST_DATASET
        assert (shard_dir / ".nv-meta" / "split.yaml").exists()
        assert (shard_dir / ".nv-meta" / "dataset.yaml").exists()
        assert (shard_dir / "shard_0.jsonl.idx").exists()
        assert (shard_dir / "shard_1.jsonl.idx").exists()

        dataset = get_train_dataset(
            shard_dir,
            worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=42),
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            task_encoder=SimpleCookingTaskEncoder(),
            repeat=False,
        )
        assert len(dataset) == 7
        samples = list(get_loader(dataset))
        assert sorted(idx for batch in samples for idx in batch.idx) == list(range(7))
        assert sorted(text for batch in samples for text in batch.text) == [
            str(i) for i in range(7)
        ]

    def test_prepared_jsonl_shard_directory_multirank(self):
        shard_dir = self.dataset_path / "jsonl_shards_multirank"
        shard_dir.mkdir()
        self.create_text_test_dataset(
            shard_dir / "shard_0.jsonl", range(0, 2), range(0, 2), prepare=False
        )
        self.create_text_test_dataset(
            shard_dir / "shard_1.jsonl", range(2, 5), range(2, 5), prepare=False
        )
        self.create_text_test_dataset(
            shard_dir / "shard_2.jsonl", range(5, 9), range(5, 9), prepare=False
        )
        self.prepare_jsonl_shard_dir(shard_dir)

        sample_counts = Counter()
        for cur_rank in range(3):
            dataset = get_train_dataset(
                shard_dir,
                worker_config=WorkerConfig(
                    rank=cur_rank,
                    world_size=3,
                    num_workers=0,
                    seed_offset=42,
                ),
                batch_size=1,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                task_encoder=SimpleCookingTaskEncoder(),
                repeat=False,
            )
            for batch in get_loader(dataset):
                for idx in batch.idx:
                    sample_counts[int(idx)] += 1

        assert sample_counts == Counter(range(9))

    def test_prepared_jsonl_shard_directory_split_from_recipe(self):
        shard_dir = self.dataset_path / "jsonl_shards_split"
        shard_dir.mkdir()
        self.create_text_test_dataset(
            shard_dir / "shard_0.jsonl", range(0, 2), range(0, 2), prepare=False
        )
        self.create_text_test_dataset(
            shard_dir / "shard_1.jsonl", range(10, 12), range(10, 12), prepare=False
        )
        self.create_text_test_dataset(
            shard_dir / "shard_2.jsonl", range(20, 22), range(20, 22), prepare=False
        )
        self.prepare_jsonl_shard_dir(shard_dir, split_ratio="1,1,1")

        recipe_path = self.dataset_path / "jsonl_shard_split_mds.yaml"
        recipe_path.write_text(
            "\n".join(
                [
                    "__module__: megatron.energon",
                    "__class__: Recipe",
                    "splits:",
                    "  train:",
                    "    path: jsonl_shards_split",
                    "    split_part: val",
                ]
            )
        )

        dataset = get_train_dataset(
            recipe_path,
            worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=42),
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            task_encoder=SimpleCookingTaskEncoder(),
            repeat=False,
        )
        samples = list(get_loader(dataset))
        assert [idx for batch in samples for idx in batch.idx] == [10, 11]

    def test_prepared_jsonl_shard_directory_inline_split_config(self):
        shard_dir = self.dataset_path / "jsonl_shards_inline_split"
        shard_dir.mkdir()
        self.create_text_test_dataset(
            shard_dir / "shard_0.jsonl", range(0, 2), range(0, 2), prepare=False
        )
        self.create_text_test_dataset(
            shard_dir / "shard_1.jsonl", range(10, 12), range(10, 12), prepare=False
        )
        self.prepare_jsonl_shard_dir(shard_dir)

        factory = DefaultCrudeJsonlShardListDatasetFactory(
            EPath(shard_dir),
            training=False,
            worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
            split_config=WebdatasetSplits(split_parts={"custom": ["shard_1.jsonl"]}),
            split_part="custom",
        )
        assert len(factory) == 2
        store = factory.as_file_store()
        data, _source = store["0"]
        assert json.loads(data) == {"idx": 10, "txt": "10"}
        store.close()

    def test_prepared_jsonl_shard_directory_rejects_stale_index(self):
        shard_dir = self.dataset_path / "jsonl_shards_stale"
        shard_dir.mkdir()
        self.create_text_test_dataset(
            shard_dir / "shard_0.jsonl", range(0, 2), range(0, 2), prepare=False
        )
        self.prepare_jsonl_shard_dir(shard_dir)
        with open(shard_dir / "shard_0.jsonl", "a") as f:
            f.write(json.dumps({"idx": 2, "txt": "2"}) + "\n")

        with self.assertRaises(AssertionError):
            get_train_dataset(
                shard_dir,
                worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=42),
                batch_size=1,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                task_encoder=SimpleCookingTaskEncoder(),
                repeat=False,
            )

    def test_prepared_jsonl_shard_directory_file_store(self):
        shard_dir = self.dataset_path / "jsonl_shards_filestore"
        shard_dir.mkdir()
        self.create_text_test_dataset(
            shard_dir / "shard_0.jsonl", range(0, 2), range(0, 2), prepare=False
        )
        self.create_text_test_dataset(
            shard_dir / "shard_1.jsonl", range(2, 4), range(2, 4), prepare=False
        )
        self.prepare_jsonl_shard_dir(shard_dir)

        factory = DefaultCrudeJsonlShardListDatasetFactory(
            EPath(shard_dir),
            training=False,
            worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
            split_part="train",
        )
        store = factory.as_file_store()
        data, source = store["2"]
        assert json.loads(data) == {"idx": 2, "txt": "2"}
        assert source.dataset_path == str(EPath(shard_dir))
        store.close()


if __name__ == "__main__":
    unittest.main()
