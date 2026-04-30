# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import unittest
from pathlib import Path

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.aggregator_pool import AggregatorPool, BaseAggregator
from megatron.energon.flavors.webdataset.indexing import DuplicateSampleKeyError, SqliteIndexWriter
from megatron.energon.flavors.webdataset.prepare import IndexSample


def produce_items(task: list[int]):
    yield from task


class BatchCollector(BaseAggregator[int, list[list[int]]]):
    def __init__(self):
        self.batches: list[list[int]] = []

    def on_item(self, items: list[int], aggregator_pool: AggregatorPool) -> None:
        self.batches.append(list(items))

    def get_final_result_data(self) -> list[list[int]]:
        return self.batches


class TestSqliteIndexWriterBatching(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sqlite_path = EPath(Path(self.temp_dir.name) / "index.sqlite")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_append_samples_batch_reports_duplicate_sample_key(self):
        with SqliteIndexWriter(self.sqlite_path) as writer:
            with self.assertRaises(DuplicateSampleKeyError) as exc:
                writer.append_samples(
                    [
                        IndexSample(
                            tar_file_id=0,
                            sample_key="first",
                            sample_index=0,
                            byte_offset=0,
                            byte_size=10,
                        ),
                        IndexSample(
                            tar_file_id=0,
                            sample_key="duplicate",
                            sample_index=1,
                            byte_offset=10,
                            byte_size=10,
                        ),
                        IndexSample(
                            tar_file_id=1,
                            sample_key="duplicate",
                            sample_index=0,
                            byte_offset=20,
                            byte_size=10,
                        ),
                    ]
                )
            assert writer.db is not None
            row_count = writer.db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

        self.assertEqual(exc.exception.sample_key, "duplicate")
        self.assertEqual(row_count, 0)

    def test_aggregator_pool_batches_worker_output(self):
        pool = AggregatorPool(
            num_workers=1,
            user_produce_data=produce_items,
            aggregator=BatchCollector(),
            batch_size=2,
        )
        pool.submit_task([1, 2, 3])
        pool.submit_task([4, 5])

        self.assertEqual(pool.process(), [[1, 2], [3], [4, 5]])


if __name__ == "__main__":
    unittest.main()
