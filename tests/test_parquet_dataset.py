# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Parquet flavor datasets and :class:`ParquetFileStore`."""

import gc
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

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
    get_savable_loader,
    get_train_dataset,
    get_val_dataset,
    stateless,
)
from megatron.energon.cache.file_store import ParquetFileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.common.filter_index import FilterIndexWriter
from megatron.energon.flavors.common.manifest.types import ShardInfo
from megatron.energon.flavors.common.manifest.write import write_manifest_dataset_metadata
from megatron.energon.flavors.parquet.dataset import (
    DefaultParquetDatasetFactory,
    DefaultParquetShardListDatasetFactory,
)
from megatron.energon.flavors.parquet.prepare import (
    ParquetFileEntry,
    parse_parquet_layout,
    scan_parquet_dataset,
)
from megatron.energon.flavors.parquet.reader import IParquetReader


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


@edataclass
class ParquetIdSample(Sample):
    row_id: int


@stateless()
def cook_parquet_id(sample: CrudeSample) -> ParquetIdSample:
    return ParquetIdSample(**basic_sample_keys(sample), row_id=int(sample["id"]))


class ParquetIdEncoder(DefaultTaskEncoder):
    cookers = [Cooker(cook=cook_parquet_id)]


class ParquetCookerPartFilterEncoder(DefaultTaskEncoder):
    cookers = [
        Cooker(
            cook=cook_parquet_keep_only,
            part_filter=lambda col: col == "keep",
        )
    ]


class _DecodedRows:
    """Counts the rows decoded through :class:`pyarrow.parquet.ParquetFile`."""

    rows: int = 0

    def __enter__(self) -> "_DecodedRows":
        orig_read_row_group = pq.ParquetFile.read_row_group
        orig_iter_batches = pq.ParquetFile.iter_batches

        def read_row_group(pf, *args, **kwargs):
            table = orig_read_row_group(pf, *args, **kwargs)
            self.rows += table.num_rows
            return table

        def iter_batches(pf, *args, **kwargs):
            for batch in orig_iter_batches(pf, *args, **kwargs):
                self.rows += batch.num_rows
                yield batch

        self._patches = [
            mock.patch.object(pq.ParquetFile, "read_row_group", read_row_group),
            mock.patch.object(pq.ParquetFile, "iter_batches", iter_batches),
        ]
        for patch in self._patches:
            patch.start()
        return self

    def __exit__(self, *exc) -> None:
        for patch in reversed(self._patches):
            patch.stop()


class TestParquetLayoutParse(unittest.TestCase):
    def test_parse_parquet_layout_mapping(self):
        layout = parse_parquet_layout(
            {
                "version": 1,
                "columns": ["a", "b"],
                "files": [{"rel_path": "p.parquet", "num_rows": 3, "row_group_rows": [2, 1]}],
                "total_rows": 3,
            }
        )
        assert layout.version == 1
        assert layout.columns == ["a", "b"]
        assert len(layout.files) == 1
        assert layout.files[0] == ParquetFileEntry(
            rel_path="p.parquet", num_rows=3, row_group_rows=[2, 1]
        )
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
        parquet_path = self.root / "part0.parquet"
        pq.write_table(t, parquet_path)

        store = ParquetFileStore(EPath(parquet_path))
        full_sample = store["1"]
        id_val, src = store["1.id"]
        text_val, _ = store["1.text"]
        assert full_sample["id"] == 2
        assert full_sample["text"] == "b"
        assert id_val == 2
        assert text_val == "b"
        assert "part0.parquet" in src.shard_name

    def test_file_store_part_filter_columns(self):
        t = pa.table({"keep": [1], "drop": [9]})
        parquet_path = self.root / "one.parquet"
        pq.write_table(t, parquet_path)

        store = ParquetFileStore(EPath(parquet_path), part_filter=lambda c: c == "keep")
        keep_val, _ = store["0.keep"]
        assert keep_val == 1


class TestParquetReaderIterSlice(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "ds"
        self.root.mkdir(parents=True)
        # 10 rows in row groups of 4 rows, then 7 rows in row groups of 3 rows
        pq.write_table(pa.table({"id": list(range(10))}), self.root / "a.parquet", row_group_size=4)
        pq.write_table(
            pa.table({"id": list(range(10, 17))}), self.root / "b.parquet", row_group_size=3
        )
        self.layout = scan_parquet_dataset(EPath(self.root))

    def tearDown(self):
        gc.collect()
        self.temp_dir.cleanup()

    def test_iter_slice_matches_random_access(self):
        reader = IParquetReader(EPath(self.root), self.layout, ["id"])
        try:
            # Whole row group, mid row group start and end, across files, everything
            for start, stop in [(0, 4), (5, 7), (6, 13), (9, 10), (0, 17)]:
                with self.subTest(start=start, stop=stop):
                    assert list(reader.iter_slice(start, stop)) == [
                        reader[idx] for idx in range(start, stop)
                    ]
        finally:
            reader.close()

    def test_iter_slice_decodes_each_row_once(self):
        reader = IParquetReader(EPath(self.root), self.layout, ["id"])
        try:
            with _DecodedRows() as decoded:
                samples = list(reader.iter_slice(0, 17))
        finally:
            reader.close()
        assert [sample["id"] for sample in samples] == list(range(17))
        assert decoded.rows == 17

    def test_iter_slice_yields_none_for_filtered_samples(self):
        reader = IParquetReader(
            EPath(self.root),
            self.layout,
            ["id"],
            sample_filter=lambda key: int(key) % 2 == 0,
        )
        try:
            samples = list(reader.iter_slice(2, 7))
        finally:
            reader.close()
        assert [None if sample is None else sample["id"] for sample in samples] == [
            2,
            None,
            4,
            None,
            6,
        ]


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
        parquet_path = self.root / "p.parquet"
        pq.write_table(t, parquet_path)
        wc = WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=0)
        ds = DefaultParquetDatasetFactory(
            EPath(parquet_path),
            training=False,
            worker_config=wc,
            decoder=None,
            decode_map=None,
            tags={"src": "test"},
        )
        wc.worker_activate(0)
        try:
            sample = next(iter(ds.build()))
        finally:
            wc.worker_deactivate()
        assert sample["n"] == 1
        assert sample["s"] == "a"
        assert sample["__tags__"] == {"src": "test"}


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
        parquet_path = self.root / "data.parquet"
        pq.write_table(t, parquet_path)
        wc = WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=0)
        val_ds = get_val_dataset(
            EPath(parquet_path),
            split_part="train",
            worker_config=wc,
            batch_size=1,
            task_encoder=ParquetCookerPartFilterEncoder(),
        )
        loader = get_loader(val_ds)
        seen = [int(batch.keep_val[0]) for batch in loader]
        assert sorted(seen) == [10, 20]


class TestParquetRowGroupShuffle(unittest.TestCase):
    ROWS = 24
    ROW_GROUP_SIZE = 4

    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "data.parquet"
        pq.write_table(
            pa.table({"id": list(range(self.ROWS))}),
            self.path,
            row_group_size=self.ROW_GROUP_SIZE,
        )

    def tearDown(self):
        gc.collect()
        self.temp_dir.cleanup()

    def test_row_groups_are_shuffled_and_read_sequentially(self):
        wc = WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=0)
        ds = DefaultParquetDatasetFactory(
            EPath(self.path),
            training=True,
            worker_config=wc,
            decoder=None,
            parallel_shard_iters=1,
        )
        wc.worker_activate(0)
        try:
            iterator = iter(ds.build())
            seen = [next(iterator)["id"] for _ in range(self.ROWS)]
        finally:
            wc.worker_deactivate()

        groups = [
            seen[start : start + self.ROW_GROUP_SIZE]
            for start in range(0, self.ROWS, self.ROW_GROUP_SIZE)
        ]
        for group in groups:
            assert group == list(range(group[0], group[0] + self.ROW_GROUP_SIZE))
            assert group[0] % self.ROW_GROUP_SIZE == 0
        assert sorted(seen) == list(range(self.ROWS))
        assert seen != list(range(self.ROWS))

    def test_serial_read_order(self):
        wc = WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=0)
        ds = DefaultParquetDatasetFactory(
            EPath(self.path),
            training=True,
            worker_config=wc,
            decoder=None,
            parquet_read_order="serial",
        )
        wc.worker_activate(0)
        try:
            dataset = ds.build()
            seen = [sample["id"] for _ in range(2) for sample in dataset]
        finally:
            wc.worker_deactivate()
        assert seen == list(range(self.ROWS)) * 2

    def test_epoch_decodes_each_row_once(self):
        wc = WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=0)
        ds = DefaultParquetDatasetFactory(
            EPath(self.path),
            training=True,
            worker_config=wc,
            decoder=None,
            parallel_shard_iters=3,
        )
        wc.worker_activate(0)
        try:
            with _DecodedRows() as decoded:
                seen = [sample["id"] for sample in ds.build()]
        finally:
            wc.worker_deactivate()
        assert sorted(seen) == list(range(self.ROWS))
        assert decoded.rows == self.ROWS

    def test_max_samples_per_sequence_keeps_row_groups(self):
        wc = WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=0)
        ds = DefaultParquetDatasetFactory(
            EPath(self.path),
            training=True,
            worker_config=wc,
            decoder=None,
            parallel_shard_iters=1,
            max_samples_per_sequence=1,
        )
        wc.worker_activate(0)
        try:
            with _DecodedRows() as decoded:
                seen = [sample["id"] for sample in ds.build()]
        finally:
            wc.worker_deactivate()
        for start in range(0, self.ROWS, self.ROW_GROUP_SIZE):
            group = seen[start : start + self.ROW_GROUP_SIZE]
            assert group == list(range(group[0], group[0] + self.ROW_GROUP_SIZE))
        assert decoded.rows == self.ROWS

    def test_filter_index_reads_kept_rows(self):
        kept = [idx for idx in range(self.ROWS) if idx % 3 != 0]
        with FilterIndexWriter(
            EPath(self.path),
            "keep",
            [ShardInfo(name=self.path.name, path=EPath(self.path), count=self.ROWS)],
        ) as writer:
            writer.append_globals(kept)
        wc = WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=0)
        ds = DefaultParquetDatasetFactory(
            EPath(self.path),
            training=True,
            worker_config=wc,
            decoder=None,
            parallel_shard_iters=2,
            filter_name="keep",
        )
        wc.worker_activate(0)
        try:
            with _DecodedRows() as decoded:
                samples = list(ds.build())
        finally:
            wc.worker_deactivate()
        assert sorted(sample["id"] for sample in samples) == kept
        assert sorted(sample["__restore_key__"][-1] for sample in samples) == list(range(len(kept)))
        assert decoded.rows <= self.ROWS

    def test_invalid_read_order(self):
        wc = WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=0)
        with self.assertRaises(ValueError):
            DefaultParquetDatasetFactory(
                EPath(self.path),
                training=True,
                worker_config=wc,
                parquet_read_order="random",
            )

    def test_recipe_serial_read_order(self):
        recipe_path = Path(self.temp_dir.name) / "recipe.yaml"
        recipe_path.write_text(
            "__module__: megatron.energon\n"
            "__class__: Recipe\n"
            "splits:\n"
            "  train:\n"
            "    path: data.parquet\n"
            "    parquet_read_order: serial\n"
        )
        train_ds = get_train_dataset(
            EPath(recipe_path),
            batch_size=1,
            worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            task_encoder=ParquetIdEncoder(),
        )
        iterator = iter(get_loader(train_ds))
        seen = [int(next(iterator).row_id[0]) for _ in range(self.ROWS)]
        assert seen == list(range(self.ROWS))

    def test_save_restore_with_workers(self):
        def new_dataset():
            return get_train_dataset(
                EPath(self.path),
                batch_size=1,
                worker_config=WorkerConfig(rank=0, world_size=1, num_workers=2),
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                task_encoder=ParquetIdEncoder(),
            )

        with get_savable_loader(new_dataset()) as loader:
            iterator = iter(loader)
            first = [int(next(iterator).row_id[0]) for _ in range(5)]
            state = loader.save_state_rank()
            remaining = [int(next(iterator).row_id[0]) for _ in range(self.ROWS - 5)]

        with get_savable_loader(new_dataset()) as restored_loader:
            restored_loader.restore_state_rank(state)
            restored_iterator = iter(restored_loader)
            restored = [int(next(restored_iterator).row_id[0]) for _ in range(self.ROWS - 5)]

        assert restored == remaining
        assert sorted(first + remaining) == list(range(self.ROWS))


class TestParquetManifestDataset(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "ds"
        self.root.mkdir(parents=True)
        (self.root / ".nv-meta").mkdir()

    def tearDown(self):
        gc.collect()
        self.temp_dir.cleanup()

    def test_manifest_parquet_shards(self):
        first = self.root / "a.parquet"
        second = self.root / "b.parquet"
        pq.write_table(pa.table({"idx": [0, 1]}), first)
        pq.write_table(pa.table({"idx": [2, 3, 4]}), second)
        write_manifest_dataset_metadata(
            EPath(self.root),
            shards=[
                ShardInfo(name="a.parquet", path=EPath(first), count=2),
                ShardInfo(name="b.parquet", path=EPath(second), count=3),
            ],
            split_config="split.yaml",
            split_parts_ratio=[("train", 1.0)],
            dataset_definition={
                "__module__": "megatron.energon",
                "__class__": "DefaultParquetShardListDatasetFactory",
            },
        )
        wc = WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=0)
        ds = DefaultParquetShardListDatasetFactory(
            EPath(self.root),
            training=False,
            worker_config=wc,
            decoder=None,
        )
        wc.worker_activate(0)
        try:
            seen = [sample["idx"] for sample in ds.build()]
        finally:
            wc.worker_deactivate()
        assert seen == [0, 1, 2, 3, 4]


if __name__ == "__main__":
    unittest.main()
