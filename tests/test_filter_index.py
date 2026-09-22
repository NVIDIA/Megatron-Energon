# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shard-preserving filter index translation."""

import gc
import json
import struct
import tempfile
import unittest
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds
from click.testing import CliRunner

from megatron.energon import (
    TextSample,
    WorkerConfig,
    build_filter_index,
    build_filter_index_from_shard_indexes,
    load_dataset,
)
from megatron.energon.dataset_config import get_dataset_from_config
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory
from megatron.energon.flavors.common.filter_index import (
    FilterIndex,
    FilterIndexWriter,
    filter_index_paths,
)
from megatron.energon.flavors.common.manifest.io import ShardListMeta
from megatron.energon.flavors.common.manifest.paths import MAIN_FOLDER_NAME
from megatron.energon.flavors.common.manifest.types import ShardInfo
from megatron.energon.flavors.common.manifest.write import write_manifest_dataset_metadata
from megatron.energon.tools.prepare import command as prepare_command


@contextmanager
def _active_worker(worker_config: WorkerConfig) -> Iterator[None]:
    worker_config.worker_activate(0)
    try:
        yield
    finally:
        worker_config.worker_deactivate()


def _collect_factory_samples(factory):
    with _active_worker(factory.worker_config):
        return list(factory.build())


def _write_jsonl(path: Path, values: range) -> None:
    with open(path, "w") as f:
        for value in values:
            f.write(json.dumps({"idx": value, "text": f"text-{value}"}) + "\n")


def _write_binidx(path: Path, *, num_docs: int, doc_len: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.arange(num_docs * doc_len, dtype=np.int32)
    values.tofile(str(path))
    with open(path.with_suffix(".idx"), "wb") as f:
        f.write(b"MMIDIDX\x00\x00")
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", 4))
        f.write(struct.pack("<Q", num_docs))
        f.write(struct.pack("<Q", num_docs))
        f.write(np.full(num_docs, doc_len, dtype=np.int32).tobytes())
        f.write((np.arange(num_docs, dtype=np.int64) * doc_len * 4).tobytes())
        f.write(np.arange(num_docs, dtype=np.int64).tobytes())


class TestFilterIndex(unittest.TestCase):
    # Set up the test fixture
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name)

    def tearDown(self):
        # Remove all temporary files
        gc.collect()
        self.temp_dir.cleanup()

    def test_filter_index_round_trip(self) -> None:
        dataset_path = EPath(self.dataset_path / "dataset")
        shards = [
            ShardInfo(name="a", path=dataset_path / "a", count=5),
            ShardInfo(name="b", path=dataset_path / "b", count=4),
        ]
        build_filter_index(
            dataset_path,
            "keep",
            (1, 3, 5, 7),
            shards=shards,
        )
        json_path, _index_path = filter_index_paths(dataset_path, "keep")
        with json_path.open("r") as f:
            assert set(json.load(f)) == {"version", "filtered_shard_counts"}

        index = FilterIndex(dataset_path, "keep")
        assert index.total_filtered == 4
        assert index[2] == 5

    def test_filter_index_global_writer(self) -> None:
        dataset_path = EPath(self.dataset_path / "dataset")
        shards = [
            ShardInfo(name="a", path=dataset_path / "a", count=5),
            ShardInfo(name="b", path=dataset_path / "b", count=4),
        ]
        with FilterIndexWriter(
            dataset_path,
            "keep",
            shards,
        ) as writer:
            writer.append(1)
            writer.append(3)
            writer.append(5)
            writer.append(7)

        index = FilterIndex(dataset_path, "keep")
        assert index[3] == 7

    def test_filter_index_global_builder(self) -> None:
        dataset_path = EPath(self.dataset_path / "dataset")
        shards = [
            ShardInfo(name="a", path=dataset_path / "a", count=5),
            ShardInfo(name="b", path=dataset_path / "b", count=4),
        ]
        build_filter_index(
            dataset_path,
            "keep",
            (1, 3, 5, 7),
            shards=shards,
        )

        index = FilterIndex(dataset_path, "keep")
        assert index.filtered_shard_counts == {"a": 2, "b": 2}
        assert index[0] == 1
        assert index[3] == 7

    def test_filter_index_metadata_loads_without_index_map(self) -> None:
        dataset_path = EPath(self.dataset_path / "dataset")
        shards = [
            ShardInfo(name="a", path=dataset_path / "a", count=5),
            ShardInfo(name="b", path=dataset_path / "b", count=4),
        ]
        build_filter_index(
            dataset_path,
            "keep",
            (1, 3, 5, 7),
            shards=shards,
        )
        _json_path, index_path = filter_index_paths(dataset_path, "keep")
        index_path.unlink()

        index = FilterIndex(dataset_path, "keep")
        assert len(index) == 4
        assert index.translate_shards(shards) == [
            ShardInfo(name="a", path=dataset_path / "a", count=2),
            ShardInfo(name="b", path=dataset_path / "b", count=2),
        ]
        with self.assertRaises(AssertionError):
            index[0]

    def test_webdataset_filter(self) -> None:
        dataset_path = self.dataset_path / "wds"
        (dataset_path / "parts").mkdir(parents=True)
        with wds.ShardWriter(str(dataset_path / "parts/data-%d.tar"), maxcount=10) as writer:
            for idx in range(6):
                writer.write({"__key__": f"{idx:06d}", "txt": f"text-{idx}".encode()})
            total_shards = writer.shard

        BaseWebdatasetFactory.prepare_dataset(
            dataset_path,
            [f"parts/data-{{0..{total_shards - 1}}}.tar"],
            split_parts_ratio=[("train", 1.0)],
        )
        with open(dataset_path / MAIN_FOLDER_NAME / "dataset.yaml", "w") as f:
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

        meta = ShardListMeta.from_config(EPath(dataset_path), split_part="train")
        kept_by_shard = {
            meta.shards[0].name: (1, 3, 5),
        }
        build_filter_index_from_shard_indexes(
            EPath(dataset_path),
            "keep",
            kept_by_shard,
        )
        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)
        factory = get_dataset_from_config(
            EPath(dataset_path),
            split_part="train",
            training=False,
            worker_config=worker_config,
            sample_type=TextSample,
            filter_name="keep",
        )

        samples = _collect_factory_samples(factory)
        assert len(factory) == 3
        assert [sample.__key__ for sample in samples] == ["000001", "000003", "000005"]
        assert [sample.__restore_key__[1] for sample in samples] == [0, 1, 2]

    def test_prepared_jsonl_shard_filter(self) -> None:
        dataset_path = self.dataset_path / "jsonl_shards"
        dataset_path.mkdir()
        _write_jsonl(dataset_path / "a.jsonl", range(0, 3))
        _write_jsonl(dataset_path / "b.jsonl", range(3, 6))

        result = CliRunner().invoke(
            prepare_command,
            [str(dataset_path), "--split-ratio", "1,0,0", "--non-interactive", "--no-progress"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.stdout

        build_filter_index(
            EPath(dataset_path),
            "keep",
            (2, 4),
        )
        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)
        factory = get_dataset_from_config(
            EPath(dataset_path),
            split_part="train",
            training=False,
            worker_config=worker_config,
            filter_name="keep",
        )

        samples = _collect_factory_samples(factory)
        assert len(factory) == 2
        assert [sample["json"]["idx"] for sample in samples] == [2, 4]
        assert [sample["__restore_key__"][1] for sample in samples] == [0, 1]

    def test_parquet_filter(self) -> None:
        dataset_path = self.dataset_path / "parquet"
        dataset_path.mkdir()
        (dataset_path / MAIN_FOLDER_NAME).mkdir()
        pq.write_table(pa.table({"idx": [0, 1, 2]}), dataset_path / "a.parquet")
        pq.write_table(pa.table({"idx": [3, 4, 5]}), dataset_path / "b.parquet")
        shards = [
            ShardInfo(name="a.parquet", path=EPath(dataset_path / "a.parquet"), count=3),
            ShardInfo(name="b.parquet", path=EPath(dataset_path / "b.parquet"), count=3),
        ]
        write_manifest_dataset_metadata(
            EPath(dataset_path),
            shards=shards,
            split_config="split.yaml",
            split_parts_ratio=[("train", 1.0)],
            dataset_definition={
                "__module__": "megatron.energon",
                "__class__": "DefaultParquetShardListDatasetFactory",
            },
        )

        build_filter_index(
            EPath(dataset_path),
            "keep",
            (2, 4),
            shards=shards,
        )
        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)
        factory = get_dataset_from_config(
            EPath(dataset_path),
            split_part="train",
            training=False,
            worker_config=worker_config,
            filter_name="keep",
        )

        samples = _collect_factory_samples(factory)
        assert len(factory) == 2
        assert [sample["idx"] for sample in samples] == [2, 4]
        assert [sample["__restore_key__"][1] for sample in samples] == [0, 1]

    def test_binidx_filter_and_recipe_passthrough(self) -> None:
        dataset_path = self.dataset_path / "binidx"
        bin_path = dataset_path / "tokens.bin"
        _write_binidx(bin_path, num_docs=6)
        build_filter_index(
            EPath(bin_path),
            "keep",
            indexes=(1, 4),
        )
        json_path, idx_path = filter_index_paths(EPath(bin_path), "keep")
        assert json_path.name == "tokens.bin.keep.json"
        assert idx_path.name == "tokens.bin.keep.idx"

        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)
        factory = get_dataset_from_config(
            EPath(bin_path),
            split_part="train",
            training=False,
            worker_config=worker_config,
            filter_name="keep",
        )
        samples = _collect_factory_samples(factory)
        assert len(factory) == 2
        assert [sample["__key__"] for sample in samples] == ["1", "4"]
        assert [sample["__restore_key__"][1] for sample in samples] == [0, 1]

        recipe_path = dataset_path / "recipe.yaml"
        with open(recipe_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    path: tokens.bin",
                        "    filter: keep",
                    ]
                )
            )
        loader = load_dataset(EPath(recipe_path))
        loaded = loader.get_datasets(
            training=False,
            split_part="train",
            worker_config=worker_config,
        )
        assert len(loaded.datasets[0].dataset) == 2
