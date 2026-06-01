# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Parquet flavor datasets and :class:`ParquetFileStore`."""

import gc
import tempfile
import unittest
import warnings
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from megatron.energon import (
    Cooker,
    CrudeSample,
    DefaultTaskEncoder,
    Sample,
    WorkerConfig,
    basic_sample_keys,
    edataclass,
    get_loader,
    get_val_dataset,
    stateless,
)
from megatron.energon.cache.file_store import ParquetFileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.parquet.dataset import DefaultParquetDatasetFactory
from megatron.energon.flavors.parquet.prepare import (
    ParquetFileEntry,
    parse_parquet_layout,
)


@edataclass
class ParquetKeepSample(Sample):
    keep_val: int


@stateless()
def cook_parquet_keep_only(sample: CrudeSample) -> ParquetKeepSample:
    assert "drop" not in sample
    return ParquetKeepSample(
        **basic_sample_keys(sample),
        keep_val=int(sample["keep"]),
    )


class ParquetCookerPartFilterEncoder(DefaultTaskEncoder):
    cookers = [
        Cooker(
            cook=cook_parquet_keep_only,
            part_filter=lambda col: col == "keep",
        )
    ]


class TestParquetLayoutParse(unittest.TestCase):
    def test_parse_parquet_layout_mapping(self):
        layout = parse_parquet_layout(
            {
                "version": 1,
                "columns": ["a", "b"],
                "files": [{"rel_path": "p.parquet", "num_rows": 3}],
                "total_rows": 3,
            }
        )
        assert layout.version == 1
        assert layout.columns == ["a", "b"]
        assert len(layout.files) == 1
        assert layout.files[0] == ParquetFileEntry(rel_path="p.parquet", num_rows=3)
        assert layout.total_rows == 3


class TestParquetFileStore(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "ds"
        self.root.mkdir(parents=True)

    def tearDown(self):
        gc.collect()
        self.temp_dir.cleanup()

    def test_file_store_getitem_per_column(self):
        t = pa.table({"id": [1, 2], "text": ["a", "b"]})
        pq.write_table(t, self.root / "part0.parquet")

        store = ParquetFileStore(EPath(self.root))
        id_val, src = store["1.id"]
        text_val, _ = store["1.text"]
        assert id_val == 2
        assert text_val == "b"
        assert "part0.parquet" in src.shard_name

    def test_file_store_part_filter_columns(self):
        t = pa.table({"keep": [1], "drop": [9]})
        pq.write_table(t, self.root / "one.parquet")

        store = ParquetFileStore(EPath(self.root), part_filter=lambda c: c == "keep")
        keep_val, _ = store["0.keep"]
        assert keep_val == 1


class TestDefaultParquetDatasetFactoryDecoders(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "ds"
        self.root.mkdir(parents=True)

    def tearDown(self):
        gc.collect()
        self.temp_dir.cleanup()

    def test_flat_columns_with_decoder_disabled(self):
        t = pa.table({"n": [1, 2], "s": ["a", "b"]})
        pq.write_table(t, self.root / "p.parquet")
        wc = WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=0)
        ds = DefaultParquetDatasetFactory(
            EPath(self.root),
            training=False,
            worker_config=wc,
            decoder=None,
            decode_map=None,
            subflavors={"src": "test"},
        )
        wc.worker_activate(0)
        try:
            sample = next(iter(ds.build()))
        finally:
            wc.worker_deactivate()
        assert sample["n"] == 1
        assert sample["s"] == "a"
        assert sample["__subflavors__"] == {"src": "test"}


class TestParquetCookerPartFilter(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "ds"
        self.root.mkdir(parents=True)

    def tearDown(self):
        gc.collect()
        self.temp_dir.cleanup()

    def test_cooker_part_filter_excludes_columns(self):
        t = pa.table({"keep": [10, 20], "drop": [999, 888]})
        pq.write_table(t, self.root / "data.parquet")
        wc = WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=0)
        val_ds = get_val_dataset(
            EPath(self.root),
            split_part="train",
            worker_config=wc,
            batch_size=1,
            task_encoder=ParquetCookerPartFilterEncoder(),
        )
        loader = get_loader(val_ds)
        seen = [int(batch.keep_val[0]) for batch in loader]
        assert sorted(seen) == [10, 20]


if __name__ == "__main__":
    unittest.main()
