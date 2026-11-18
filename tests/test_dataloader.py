# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for meta datasets."""

import gc
import io
import json
import logging
import subprocess
import sys
import tempfile
import unittest
import warnings
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from textwrap import dedent
from typing import Iterable

import torch
import webdataset as wds
from click.testing import CliRunner

from megatron.energon import (
    WorkerConfig,
    get_train_dataset,
)
from megatron.energon.dataloader import DataLoader, ForkDataLoaderWorker, ThreadDataLoaderWorker
from megatron.energon.dataloader.dataloader import _split_batch_restore_key
from megatron.energon.dataloader.workers.base_worker import WorkerSampleRestoreKey
from megatron.energon.flavors.common.sample_record import IndexedRestoreKey
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from megatron.energon.task_encoder.base import DefaultTaskEncoder, stateless
from megatron.energon.tools.checkpoint import command_redist
from megatron.energon.wrappers.batch_dataset import BatchGenRestoreKey, BatchRestoreKey
from megatron.energon.wrappers.epochize_dataset import EpochizeDataset
from megatron.energon.wrappers.map_dataset import MapGenRestoreKey, MapRestoreKey
from megatron.energon.wrappers.packing_dataset import PartialSampleRestoreKey


class VerifyWorkerTaskEncoder(DefaultTaskEncoder):
    expected_num_workers: int

    def __init__(self, expected_num_workers: int):
        self.expected_num_workers = expected_num_workers
        super().__init__()

    @stateless
    def encode_sample(self, sample):
        sample = super().encode_sample(sample)
        worker_info = torch.utils.data.get_worker_info()
        if self.expected_num_workers > 0:
            assert worker_info is not None
            assert worker_info.num_workers == self.expected_num_workers
        else:
            assert worker_info is None
        return sample


def create_text_test_dataset(path: Path, txt_range: Iterable[int], key_range: Iterable[int]):
    """Creates a small dummy test dataset for testing purposes."""

    # Create num_samples unique captions
    (path / "parts").mkdir(exist_ok=True, parents=True)

    # Initialize the ShardWriter
    with wds.ShardWriter(f"{path}/parts/data-%d.tar", maxcount=10) as shard_writer:
        for key, txt in zip(key_range, txt_range):
            # Write individual files to shards
            shard_writer.write(
                {
                    "__key__": f"{key:06d}",
                    "txt": f"{txt}".encode(),
                },
            )
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
                    "tags:",
                    "  source: dataset.yaml",
                    "  dataset.yaml: true",
                    "  number: 42",
                ]
            )
        )


class GeneratorTaskEncoder(DefaultTaskEncoder):
    @stateless
    def encode_sample(self, sample):
        for output in range(3):
            yield replace(sample, text=f"{sample.text}:{self.current_sample_index}:{output}")


class ThreadedEpochDataset(EpochizeDataset):
    calls: list[int]
    worker_events: list[tuple[str, int]]

    __hash__ = None
    _worker_local_fields = ("calls",)

    def __init__(self, dataset, length, worker_config):
        super().__init__(dataset, length, worker_config)
        self.calls = []
        self.worker_events = []

    def worker_init(self):
        super().worker_init()
        self.worker_events.append(("init", self.worker_config.rank_worker_id()))

    def worker_close(self):
        self.worker_events.append(("close", self.worker_config.rank_worker_id()))
        del self.calls
        super().worker_close()

    def __iter__(self):
        for sample in super().__iter__():
            worker_id = self.worker_config.rank_worker_id()
            self.calls.append(worker_id)
            if any(previous != worker_id for previous in self.calls):
                raise AssertionError("Worker-local fields are shared between threads")
            yield sample


class TestDataLoader(unittest.TestCase):
    temp_dir: tempfile.TemporaryDirectory
    dataset_path: Path
    ds1_path: Path

    def setUp(self):
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        warnings.simplefilter("ignore", ResourceWarning)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name)
        self.dataset_path.mkdir(exist_ok=True, parents=True)
        self.ds1_path = self.dataset_path / "ds1"
        self.ds1_path.mkdir(exist_ok=True, parents=True)
        create_text_test_dataset(self.ds1_path, range(55), range(55))

    def tearDown(self):
        gc.collect()
        self.temp_dir.cleanup()

    def test_dataloader_no_workers(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Train mode dataset
        with DataLoader(
            get_train_dataset(
                self.ds1_path,
                worker_config=worker_config,
                batch_size=10,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
                task_encoder=VerifyWorkerTaskEncoder(expected_num_workers=0),
            ),
        ) as train_loader:
            assert len(train_loader) == 6, len(train_loader)

            train_order1 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]
            print(train_order1[:10])
            print(Counter(train_order1))
            assert len(train_order1) == 55, len(train_order1)
            assert len(Counter(train_order1)) == 55, Counter(train_order1)
            assert all(v == 1 for v in Counter(train_order1).values()), Counter(train_order1)

            state1 = train_loader.save_state_rank()

            train_order2 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]

        with DataLoader(
            get_train_dataset(
                self.ds1_path,
                worker_config=worker_config,
                batch_size=10,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
                task_encoder=VerifyWorkerTaskEncoder(expected_num_workers=0),
            ),
        ).with_restored_state_rank(state1) as train_loader:
            cmp_order2 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]
            assert train_order2 == cmp_order2, (train_order1, cmp_order2)

    def test_context_manager_keeps_running_workers(self):
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )
        loader = DataLoader(
            get_train_dataset(
                self.ds1_path,
                worker_config=worker_config,
                batch_size=10,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
            )
        )

        loader_iter = iter(loader)
        next(loader_iter)

        with self.assertRaises(AssertionError, msg="already started"):
            with loader:
                raise Exception("should not be reached")

        loader.shutdown()

        assert loader._workers is None

        with loader:
            assert loader._workers is not None

        assert loader._workers is None

    def test_split_batch_restore_key_by_target_size(self):
        keys = tuple(IndexedRestoreKey(kind="Webdataset", index=i) for i in range(6))
        batch = BatchRestoreKey(sample_idx=12, inner=keys)
        mapped = MapRestoreKey(sample_idx=7, inner=batch)
        wrapped = WorkerSampleRestoreKey(worker_id=5, sample_idx=32, inner=mapped)

        expected = [
            replace(wrapped, inner=replace(mapped, inner=replace(batch, inner=chunk)))
            for chunk in (keys[:4], keys[4:])
        ]
        self.assertEqual(_split_batch_restore_key(wrapped, 4), expected)
        self.assertEqual(_split_batch_restore_key(wrapped, 4, drop_last=True), expected[:1])
        self.assertEqual(
            _split_batch_restore_key(replace(batch, inner=keys[:3]), 4)[0].inner, keys[:3]
        )

        generator_batch = BatchGenRestoreKey(sample_idx=12, gen_idx=3, inner=keys)
        with self.assertRaisesRegex(ValueError, "prefetched generator batches"):
            _split_batch_restore_key(replace(wrapped, inner=generator_batch), 4)

    def test_restore_smaller_micro_batches_with_prefetched_tail(self):
        wc = WorkerConfig(rank=0, world_size=1, num_workers=1)

        def loader(batch_size, prefetch_factor, drop_last=False):
            return DataLoader(
                get_train_dataset(
                    self.ds1_path,
                    worker_config=wc,
                    batch_size=batch_size,
                    batch_drop_last=drop_last,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                    repeat=False,
                ),
                prefetch_factor=prefetch_factor,
                pin_memory=None,
            )

        with loader(8, 8) as source:
            first = next(iter(source)).text
            checkpoint = source.save_state_rank()
            self.assertTrue(checkpoint.prefetched_restore_keys[0])
            source_tail = [text for batch in source for text in batch.text]
        self.assertEqual(len(first), 8)
        self.assertEqual(len(source_tail) % 4, 3)

        for drop_last in (False, True):
            with self.subTest(drop_last=drop_last):
                with loader(4, 0, drop_last).with_restored_state_rank(checkpoint) as restored:
                    batches = [list(batch.text) for batch in restored]
                stop = len(source_tail) - 3 if drop_last else len(source_tail)
                self.assertEqual(batches, [source_tail[i : i + 4] for i in range(0, stop, 4)])

        input_file = self.dataset_path / "global.pt"
        output_dir = self.dataset_path / "redistributed"
        torch.save([checkpoint], input_file)
        result = CliRunner().invoke(
            command_redist,
            [
                str(input_file),
                str(output_dir),
                "--new-world-size",
                "1",
                "--new-micro-batch-size",
                "4",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        redistributed = torch.load(output_dir / input_file.name, weights_only=False)[0]
        with loader(4, 0).with_restored_state_rank(redistributed) as restored:
            self.assertEqual(
                [list(batch.text) for batch in restored],
                [source_tail[i : i + 4] for i in range(0, len(source_tail), 4)],
            )

    def test_fork_worker_exit_fails_iteration_and_all_pending_requests(self):
        # Bound the subprocess in case failure detection regresses into a blocking queue read.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                dedent("""
                    import multiprocessing
                    import os
                    from megatron.energon import WorkerConfig
                    from megatron.energon.dataloader import DataLoader
                    from megatron.energon.dataloader.asynchronous.base import WorkerExitedError

                    exit_worker = multiprocessing.get_context("fork").Event()

                    class Dataset:
                        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=1)

                        def __iter__(self):
                            exit_worker.wait()
                            os._exit(17)
                            yield

                        def worker_init(self): pass
                        def worker_close(self): pass
                        def reset_state_deep(self): pass
                        def close(self): pass

                    with DataLoader(
                        Dataset(), prefetch_factor=0, pin_memory=None,
                        watchdog_timeout_seconds=None, gc_collect_every_n_steps=0,
                    ) as loader:
                        worker = loader._workers[0]
                        pending = [worker.prefetch_next(), worker.prefetch_next()]
                        exit_worker.set()
                        for future in pending:
                            try:
                                future.get()
                            except WorkerExitedError as error:
                                assert "wkr-0 exited with code 17" in str(error), error
                            else:
                                raise AssertionError("Expected worker-exit error")
                    assert not worker.running()

                    with DataLoader(
                        Dataset(), prefetch_factor=0, pin_memory=None,
                        watchdog_timeout_seconds=None, gc_collect_every_n_steps=0,
                    ) as loader:
                        try:
                            next(iter(loader))
                        except WorkerExitedError as error:
                            assert "wkr-0 exited with code 17" in str(error), error
                        else:
                            raise AssertionError("Expected iteration to report the worker exit")
                """),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_generator_keys_resume_and_restore_at_the_original_index(self):
        wc = WorkerConfig(rank=0, world_size=1, num_workers=0, worker_log_level=1)

        def loader():
            return DataLoader(
                get_train_dataset(
                    self.ds1_path,
                    worker_config=wc,
                    batch_size=None,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                    task_encoder=GeneratorTaskEncoder(),
                    repeat=False,
                )
            )

        log = io.StringIO()
        with redirect_stdout(log), loader() as source:
            it = iter(source)
            first = next(it)
            checkpoint = source.save_state_rank()
            expected = next(it)
            self.assertIsInstance(first.__restore_key__, WorkerSampleRestoreKey)
            self.assertIsInstance(first.__restore_key__.inner, MapGenRestoreKey)
            self.assertEqual(source.restore_sample(expected.__restore_key__).text, expected.text)
            self.assertEqual(first.pin_memory().__restore_key__, first.__restore_key__)
            self.assertNotIsInstance(first.__restore_key__, tuple)
            with self.assertRaisesRegex(TypeError, "Expected WorkerSampleRestoreKey"):
                source.restore_sample(("DataLoaderWorker", 0, 0, "Webdataset", 0))

        events = [json.loads(line) for line in log.getvalue().splitlines() if line.startswith("{")]
        for event_type in ("DataLoader.epoch_iter.yield", "SavableDataLoader.yield"):
            event = next(event for event in events if event["t"] == event_type)
            key = event["restore_key"]
            self.assertEqual(key["type"], "WorkerSampleRestoreKey")
            self.assertEqual(key["fields"]["sample_idx"], first.__restore_key__.sample_idx)
            self.assertEqual(key["fields"]["inner"]["type"], "MapGenRestoreKey")

        with io.BytesIO() as buffer:
            torch.save(checkpoint, buffer)
            buffer.seek(0)
            checkpoint = torch.load(buffer, weights_only=False)

        with (
            redirect_stdout(io.StringIO()),
            loader().with_restored_state_rank(checkpoint) as target,
        ):
            actual = next(iter(target))
            self.assertEqual(actual.text, expected.text)
            self.assertEqual(actual.__restore_key__, expected.__restore_key__)
            self.assertIsInstance(actual.__restore_key__.inner, MapGenRestoreKey)

    def test_restore_key_json_preserves_nested_keys_and_slice(self):
        leaf = IndexedRestoreKey(kind="Webdataset", index=7)
        partial = PartialSampleRestoreKey(inner=leaf, slice=slice(1, 5))
        batch = BatchGenRestoreKey(sample_idx=3, gen_idx=1, inner=(partial, None))

        fields = json.loads(json.dumps(batch.to_json_object(), allow_nan=False))["fields"]
        self.assertEqual(fields["gen_idx"], 1)
        partial_json, missing = fields["inner"]
        self.assertIsNone(missing)
        self.assertEqual(partial_json["type"], "PartialSampleRestoreKey")
        self.assertEqual(
            partial_json["fields"]["slice"],
            {"type": "slice", "start": 1, "stop": 5, "step": None},
        )
        self.assertEqual(
            partial_json["fields"]["inner"],
            {"type": "IndexedRestoreKey", "fields": {"kind": "Webdataset", "index": 7}},
        )

    def test_thread_workers_isolate_inherited_epoch_iterator_and_custom_fields(self):
        def loader(worker_type):
            wc = WorkerConfig(rank=0, world_size=1, num_workers=2)
            dataset = get_train_dataset(
                self.ds1_path,
                worker_config=wc,
                batch_size=None,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
            )
            epoch = ThreadedEpochDataset(dataset, 12, wc)
            return epoch, DataLoader(epoch, worker_type=worker_type, prefetch_factor=2)

        _, reference = loader(ForkDataLoaderWorker)
        with reference:
            expected = [[sample.text for sample in reference] for _ in range(2)]
        dataset, threaded = loader(ThreadDataLoaderWorker)
        with threaded:
            actual = [[sample.text for sample in threaded] for _ in range(2)]

        self.assertEqual(actual, expected)
        self.assertEqual(dataset.calls, [])
        self.assertCountEqual(
            dataset.worker_events, [("init", 0), ("init", 1), ("close", 0), ("close", 1)]
        )

    def test_dataloader_fork(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=2,
            seed_offset=42,
        )

        # Train mode dataset
        with DataLoader(
            get_train_dataset(
                self.ds1_path,
                worker_config=worker_config,
                batch_size=10,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
                task_encoder=VerifyWorkerTaskEncoder(expected_num_workers=2),
            ),
            prefetch_factor=2,
            worker_type=ForkDataLoaderWorker,
            gc_collect_every_n_steps=10,
            gc_freeze_at_start=True,
            watchdog_timeout_seconds=60,
            fail_on_timeout=True,
        ) as train_loader:
            assert len(train_loader) == 6, len(train_loader)

            train_order1 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]
            print(train_order1[:10])
            print(Counter(train_order1))
            assert len(train_order1) == 55, len(train_order1)
            assert len(Counter(train_order1)) == 55, Counter(train_order1)
            assert all(v == 1 for v in Counter(train_order1).values()), Counter(train_order1)

            state1 = train_loader.save_state_rank()

            train_order2 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]

            assert len(train_order1) == len(train_order2), (len(train_order1), len(train_order2))

        with DataLoader(
            get_train_dataset(
                self.ds1_path,
                worker_config=worker_config,
                batch_size=10,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
                task_encoder=VerifyWorkerTaskEncoder(
                    expected_num_workers=worker_config.num_workers
                ),
            ),
            prefetch_factor=2,
            worker_type=ForkDataLoaderWorker,
            gc_collect_every_n_steps=10,
            gc_freeze_at_start=True,
            watchdog_timeout_seconds=60,
            fail_on_timeout=True,
        ).with_restored_state_rank(state1) as train_loader:
            cmp_order2 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]
            assert train_order2 == cmp_order2, (train_order1, cmp_order2)

    def test_dataloader_fork_multi_parallel(self):
        torch.manual_seed(42)
        worker_config_r0 = WorkerConfig(
            rank=0,
            world_size=2,
            num_workers=2,
            seed_offset=42,
        )
        worker_config_r1 = WorkerConfig(
            rank=1,
            world_size=2,
            num_workers=2,
            seed_offset=42,
        )

        # Train mode dataset
        train_loader_r0 = DataLoader(
            get_train_dataset(
                self.ds1_path,
                worker_config=worker_config_r0,
                batch_size=10,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
                task_encoder=VerifyWorkerTaskEncoder(
                    expected_num_workers=worker_config_r0.num_workers
                ),
            ),
            prefetch_factor=2,
            worker_type=ForkDataLoaderWorker,
            gc_collect_every_n_steps=10,
            gc_freeze_at_start=True,
            watchdog_timeout_seconds=60,
            fail_on_timeout=True,
        )
        assert len(train_loader_r0) == 4, len(train_loader_r0)

        train_order1_r0 = [
            text for idx, data in zip(range(55 * 10), train_loader_r0) for text in data.text
        ]
        print(train_order1_r0[:10])
        print(Counter(train_order1_r0))
        assert len(train_order1_r0) == 28, len(train_order1_r0)
        assert len(Counter(train_order1_r0)) == 28, Counter(train_order1_r0)
        assert all(v == 1 for v in Counter(train_order1_r0).values()), Counter(train_order1_r0)

        train_loader_r1 = DataLoader(
            get_train_dataset(
                self.ds1_path,
                worker_config=worker_config_r1,
                batch_size=10,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
                task_encoder=VerifyWorkerTaskEncoder(
                    expected_num_workers=worker_config_r1.num_workers
                ),
            ),
            prefetch_factor=2,
            worker_type=ForkDataLoaderWorker,
            gc_collect_every_n_steps=10,
            gc_freeze_at_start=True,
            watchdog_timeout_seconds=60,
            fail_on_timeout=True,
        )
        assert len(train_loader_r1) == 4, len(train_loader_r1)

        train_order1_r1 = [
            text for idx, data in zip(range(55 * 10), train_loader_r1) for text in data.text
        ]
        print(train_order1_r1[:10])
        print(Counter(train_order1_r1))
        assert len(train_order1_r1) == 27, len(train_order1_r1)
        assert len(Counter(train_order1_r1)) == 27, Counter(train_order1_r1)
        assert all(v == 1 for v in Counter(train_order1_r1).values()), Counter(train_order1_r1)

        train_loader_r1.save_state_rank()

        train_loader_r0.save_state_rank()

        train_order2_r0 = [
            text for idx, data in zip(range(55 * 10), train_loader_r0) for text in data.text
        ]
        assert len(train_order2_r0) == 28

        train_order2_r1 = [
            text for idx, data in zip(range(55 * 10), train_loader_r1) for text in data.text
        ]
        assert len(train_order2_r1) == 27

        train_loader_r0.shutdown()
        train_loader_r1.shutdown()

    def test_dataloader_thread(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=2,
            seed_offset=42,
        )

        # Train mode dataset
        with DataLoader(
            get_train_dataset(
                self.ds1_path,
                worker_config=worker_config,
                batch_size=10,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
                task_encoder=VerifyWorkerTaskEncoder(
                    expected_num_workers=worker_config.num_workers
                ),
            ),
            prefetch_factor=2,
            worker_type=ThreadDataLoaderWorker,
            gc_collect_every_n_steps=0,
            watchdog_timeout_seconds=60,
            fail_on_timeout=True,
        ) as train_loader:
            assert len(train_loader) == 6, len(train_loader)

            train_order1 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]
            print(train_order1[:10])
            print(Counter(train_order1))
            assert len(train_order1) == 55, len(train_order1)
            assert len(Counter(train_order1)) == 55, Counter(train_order1)
            assert all(v == 1 for v in Counter(train_order1).values()), Counter(train_order1)

            state1 = train_loader.save_state_rank()

            train_order2 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]

        with DataLoader(
            get_train_dataset(
                self.ds1_path,
                worker_config=worker_config,
                batch_size=10,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
                task_encoder=VerifyWorkerTaskEncoder(
                    expected_num_workers=worker_config.num_workers
                ),
            ),
            prefetch_factor=2,
            worker_type=ThreadDataLoaderWorker,
            gc_collect_every_n_steps=0,
            watchdog_timeout_seconds=60,
            fail_on_timeout=True,
        ).with_restored_state_rank(state1) as train_loader:
            cmp_order2 = [
                text for idx, data in zip(range(55 * 10), train_loader) for text in data.text
            ]
            assert train_order2 == cmp_order2, (train_order1, cmp_order2)


if __name__ == "__main__":
    unittest.main()
