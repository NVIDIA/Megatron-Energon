# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for meta datasets."""

import gc
import io
import json
import logging
import pickle
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from collections import Counter, OrderedDict
from contextlib import redirect_stdout
from copy import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from textwrap import dedent
from typing import Iterable

import torch
import webdataset as wds
from click.testing import CliRunner

from megatron.energon import (
    WorkerConfig,
    get_savable_loader,
    get_train_dataset,
)
from megatron.energon.checkpoint.rank_state import _split_batch_restore_key
from megatron.energon.dataloader import DataLoader, ForkDataLoaderWorker, ThreadDataLoaderWorker
from megatron.energon.dataloader.future import DoneFuture
from megatron.energon.dataloader.pin_memory import PinMemoryThread
from megatron.energon.dataloader.workers.base_worker import WorkerSampleRestoreKey
from megatron.energon.errors import FatalSampleError
from megatron.energon.flavors.common.sample_record import IndexedRestoreKey
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from megatron.energon.task_encoder.base import DefaultTaskEncoder, stateless
from megatron.energon.tools.checkpoint import command_redist
from megatron.energon.wrappers.batch_dataset import BatchGenRestoreKey, BatchRestoreKey
from megatron.energon.wrappers.epochize_dataset import EpochizeDataset
from megatron.energon.wrappers.group_batch_dataset import GroupBatchDataset
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
    @stateless(restore_task_encoder_seeds=True)
    def encode_sample(self, sample):
        for output in range(3):
            yield replace(
                sample,
                text=f"{sample.text}:{self.current_sample_index}:{output}:{self.rng.random.random()}",
            )


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

    def test_redistribution_preserves_order_and_applies_target_tail_policy(self):
        def loader(batch_size, drop_last=False):
            return get_savable_loader(
                get_train_dataset(
                    self.ds1_path,
                    worker_config=WorkerConfig(rank=0, world_size=1, num_workers=2),
                    batch_size=batch_size,
                    batch_drop_last=drop_last,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                    repeat=False,
                ),
                prefetch_factor=8,
                pin_memory=False,
            )

        with loader(8) as source:
            next(iter(source))
            state = source.save_state_rank()
            self.assertEqual(state.next_worker_id, 1)
            uninterrupted = [list(batch.text) for batch in source]
        checkpoint = self.dataset_path / "state.pt"
        torch.save([state], checkpoint)
        state = torch.load(checkpoint, weights_only=False)[0]

        for size, drop_last in ((8, False), (4, True)):
            with self.subTest(batch_size=size):
                output = self.dataset_path / str(size)
                result = CliRunner().invoke(
                    command_redist,
                    [
                        str(checkpoint),
                        str(output),
                        "--new-world-size",
                        "1",
                        "--new-micro-batch-size",
                        str(size),
                    ],
                    catch_exceptions=False,
                )
                self.assertEqual(result.exit_code, 0, result.output)
                converted = torch.load(output / checkpoint.name, weights_only=False)[0]
                with loader(size, drop_last).with_restored_state_rank(state) as direct:
                    expected = [list(batch.text) for batch in direct]
                if size == 8:
                    self.assertEqual(expected, uninterrupted)
                else:
                    self.assertTrue(any(len(batch) % size for batch in uninterrupted))
                    self.assertTrue(all(len(batch) == size for batch in expected))
                with loader(size, drop_last).with_restored_state_rank(converted) as restored:
                    self.assertEqual([list(batch.text) for batch in restored], expected)

    def test_world_size_change_rejects_pending_sample_replay(self):
        for prefetch_factor, generator, error_message in (
            (2, True, "prefetch_factor=0"),
            (0, True, "in-progress generator"),
            (0, False, None),
        ):
            with self.subTest(prefetch_factor=prefetch_factor, generator=generator):
                dataset = get_train_dataset(
                    self.ds1_path,
                    worker_config=WorkerConfig(rank=0, world_size=1, num_workers=2),
                    task_encoder=GeneratorTaskEncoder() if generator else DefaultTaskEncoder(),
                    batch_size=None,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                    repeat=False,
                )
                with get_savable_loader(
                    dataset, prefetch_factor=prefetch_factor, pin_memory=False
                ) as loader:
                    next(iter(loader))
                    state = loader.save_state_rank()
                self.assertEqual(bool(any(state.prefetched_restore_keys)), bool(prefetch_factor))

                checkpoint = self.dataset_path / f"checkpoint-{prefetch_factor}-{generator}.pt"
                torch.save([state], checkpoint)
                output = self.dataset_path / f"redistributed-{prefetch_factor}-{generator}"
                result = CliRunner().invoke(
                    command_redist,
                    [str(checkpoint), str(output), "--new-world-size", "2"],
                    catch_exceptions=False,
                )
                if error_message:
                    self.assertEqual(result.exit_code, 1, result.output)
                    self.assertIn(error_message, result.output)
                    self.assertFalse(output.exists())
                else:
                    self.assertEqual(result.exit_code, 0, result.output)
                    converted = torch.load(output / checkpoint.name, weights_only=False)
                    self.assertEqual([len(rank.worker_states) for rank in converted], [1, 1])

    def test_grouped_buffers_replay_scoped_rng_after_worker_regrouping(self):
        class Encoder(DefaultTaskEncoder):
            @stateless(restore_task_encoder_seeds=True)
            def encode_sample(self, sample):
                return replace(sample, text=f"{sample.text}:{self.rng.random.random()}")

        def loader(rank, world_size, workers):
            config = WorkerConfig(rank=rank, world_size=world_size, num_workers=workers)
            encoder = Encoder()
            dataset = get_train_dataset(
                self.ds1_path,
                worker_config=config,
                task_encoder=encoder,
                batch_size=None,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
            )
            return get_savable_loader(
                GroupBatchDataset(
                    dataset,
                    worker_config=config,
                    fixed_batch_size=3,
                    sample_group_key=lambda sample: (int(sample.text.split(":")[0]) % 2, None),
                    batcher=encoder.batch,
                    batcher_stateless=True,
                ),
                prefetch_factor=0,
                pin_memory=False,
            )

        def buffered_keys(state):
            if state["__class__"] == "GroupBatchDataset":
                return [
                    key
                    for bucket in state["buckets"].values()
                    for key in bucket["samples"]["_restore_keys"]
                ]
            return [key for child in state.get("datasets", ()) for key in buffered_keys(child)]

        with loader(0, 1, 2) as source:
            iterator = iter(source)
            for _ in range(12):
                next(iterator)
                state = source.save_state_rank()
                worker = state.workers[1].state
                if buffered_keys(worker.dataset):
                    break
            else:
                self.fail("Expected buffered samples on the worker that changes rank")
            expected = [[], []]
            for batch in iterator:
                expected[batch.__restore_key__.worker_id].append(list(batch.text))

        checkpoint = self.dataset_path / "buffered.pt"
        output = self.dataset_path / "regrouped"
        torch.save([state], checkpoint)
        result = CliRunner().invoke(
            command_redist,
            [str(checkpoint), str(output), "--new-world-size", "2"],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        converted = torch.load(output / checkpoint.name, weights_only=False)
        for rank, checkpoint_state in enumerate(converted):
            with loader(rank, 2, 1).with_restored_state_rank(checkpoint_state) as restored:
                self.assertEqual([list(batch.text) for batch in restored], expected[rank])

    def test_thread_workers_surface_unsupported_process_rng(self):
        class Encoder(DefaultTaskEncoder):
            @stateless(restore_seeds=True)
            def encode_sample(self, sample):
                return sample

        dataset = get_train_dataset(
            self.ds1_path,
            worker_config=WorkerConfig(rank=0, world_size=1, num_workers=1),
            task_encoder=Encoder(),
            batch_size=None,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            repeat=False,
        )
        with get_savable_loader(dataset, worker_type="thread", pin_memory=False) as loader:
            with self.assertRaisesRegex(FatalSampleError, "Thread workers require"):
                next(iter(loader))

    def test_pin_memory_copies_frozen_slotted_dataclass_without_constructor(self):
        class Custom:
            def pin_memory(self):
                return "pinned"

        initialized = []

        @dataclass(frozen=True, slots=True)
        class Batch:
            item: object = field(init=False)

            def __post_init__(self):
                initialized.append(self)
                object.__setattr__(self, "item", Custom())

        @dataclass(frozen=True, slots=True)
        class CustomBatch(Batch):
            def pin_memory(self):
                return "custom"

        batch = Batch()
        custom = CustomBatch()
        pinning = PinMemoryThread("cpu")
        try:
            pinned = pinning(DoneFuture(batch)).get(timeout=5)
            self.assertIsNot(pinned, batch)
            self.assertEqual(pinned.item, "pinned")
            self.assertIsInstance(batch.item, Custom)
            self.assertEqual(pinning(DoneFuture(custom)).get(timeout=5), "custom")
            self.assertEqual(initialized, [batch, custom])
        finally:
            pinning.shutdown()

    def test_grouped_resize_fails_before_replaying_prefetched_batches(self):
        def loader(size, drop_last=False):
            config = WorkerConfig(rank=0, world_size=1, num_workers=1)
            dataset = get_train_dataset(
                self.ds1_path,
                worker_config=config,
                batch_size=None,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=False,
            )
            return get_savable_loader(
                GroupBatchDataset(
                    dataset,
                    worker_config=config,
                    fixed_batch_size=size,
                    sample_group_key=lambda sample: ("all", None),
                    batcher=DefaultTaskEncoder().batch,
                    batcher_stateless=True,
                    drop_last=drop_last,
                ),
                pin_memory=False,
            )

        with loader(4) as source:
            next(iter(source))
            state = source.save_state_rank()
            self.assertTrue(state.prefetched_restore_keys[0])
            expected = [list(batch.text) for batch in source]
        state = pickle.loads(pickle.dumps(state))
        for size, drop_last in ((2, False), (4, True)):
            with self.subTest(size=size, drop_last=drop_last):
                target = loader(size, drop_last)
                with self.assertRaisesRegex(ValueError, "Cannot change grouped batch"):
                    target.restore_state_rank(state)
        with loader(4).with_restored_state_rank(state) as target:
            self.assertEqual([list(batch.text) for batch in target], expected)

    def test_resize_rejects_generator_after_batching_before_replay(self):
        class Encoder(DefaultTaskEncoder):
            @stateless
            def encode_batch(self, batch):
                yield copy(batch)
                if len(batch.text) > 2:
                    yield copy(batch)

        def loader(size):
            wc = WorkerConfig(rank=0, world_size=1, num_workers=1)
            return DataLoader(
                get_train_dataset(
                    self.ds1_path,
                    worker_config=wc,
                    batch_size=size,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                    repeat=False,
                    task_encoder=Encoder(),
                ),
                prefetch_factor=3,
                pin_memory=None,
            )

        with loader(4) as source:
            it = iter(source)
            next(it)
            next(it)
            state = source.save_state_rank()
        target = loader(2)
        with self.assertRaisesRegex(ValueError, "generator batch"):
            target.restore_state_rank(state)
        self.assertIsNone(target._workers)
        self.assertIsNone(target._restore_state)

        checkpoint = self.dataset_path / "generator.pt"
        output = self.dataset_path / "out"
        torch.save([state], checkpoint)
        result = CliRunner().invoke(
            command_redist,
            [str(checkpoint), str(output), "--new-world-size", "1", "--new-micro-batch-size", "2"],
            catch_exceptions=False,
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("generator batch", result.output)
        self.assertFalse(output.exists())

    def test_in_process_restore_leaves_checkpoint_reusable(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                wc = WorkerConfig(rank=0, world_size=1, num_workers=workers, seed_offset=7)

                def loader():
                    return get_savable_loader(
                        get_train_dataset(
                            self.ds1_path,
                            worker_config=wc,
                            batch_size=4,
                            shuffle_buffer_size=5,
                            max_samples_per_sequence=7,
                            repeat=False,
                        ),
                        worker_type="thread" if workers else "main",
                        pin_memory=False,
                    )

                with loader() as source:
                    it = iter(source)
                    next(it)
                    next(it)
                    state = source.save_state_rank()
                    expected = [tuple(batch.text) for batch in it]
                snapshot = pickle.dumps(state)
                for _ in range(2):
                    with loader().with_restored_state_rank(state) as restored:
                        self.assertEqual([tuple(batch.text) for batch in restored], expected)
                    self.assertEqual(pickle.dumps(state), snapshot)

    def test_task_encoder_rng_rejects_unscoped_access(self):
        class Encoder(DefaultTaskEncoder):
            def encode_sample(self, sample):
                return replace(sample, text=f"{sample.text}/{self.rng.random.random()}")

        encoder = Encoder()
        with self.assertRaisesRegex(FatalSampleError, "restore_task_encoder_seeds=True"):
            _ = encoder.rng

        wc = WorkerConfig(rank=0, world_size=1, num_workers=2)
        dataset = get_train_dataset(
            self.ds1_path,
            worker_config=wc,
            batch_size=4,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            repeat=False,
            task_encoder=encoder,
        )
        with get_savable_loader(dataset, worker_type="thread", pin_memory=False) as loader:
            with self.assertRaisesRegex(FatalSampleError, "restore_task_encoder_seeds=True"):
                next(iter(loader))

    def test_collective_checkpoint_preflight_propagates_local_failures(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                dedent("""
                    import datetime
                    import multiprocessing
                    import sys
                    from pathlib import Path

                    import torch.distributed as dist
                    from megatron.energon import WorkerConfig
                    from megatron.energon.dataloader import DataLoader

                    def run(rank, path):
                        dist.init_process_group(
                            "gloo", init_method=path.as_uri(), rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=6),
                        )
                        try:
                            class Dataset:
                                worker_config = WorkerConfig(rank=rank, world_size=2, num_workers=0)

                            loader = DataLoader(
                                Dataset(), pin_memory=None, watchdog_timeout_seconds=None,
                                gc_collect_every_n_steps=0,
                            )
                            if rank == 0:
                                def fail_save():
                                    raise ValueError("local save failed")
                                loader.save_state_rank = fail_save
                            for operation, expected in (
                                (lambda: loader.save_state_global(0), "local save failed"),
                                (lambda: loader.restore_state_global(
                                    [] if rank == 0 else None, src_rank=0,
                                ), "rank-ordered list"),
                            ):
                                try:
                                    operation()
                                except (RuntimeError, ValueError) as error:
                                    assert expected in str(error), error
                                else:
                                    raise AssertionError("Expected preparation error on both ranks")
                            dist.barrier()
                        finally:
                            dist.destroy_process_group()

                    context = multiprocessing.get_context("fork")
                    processes = [
                        context.Process(target=run, args=(rank, Path(sys.argv[1])))
                        for rank in range(2)
                    ]
                    for process in processes:
                        process.start()
                    for process in processes:
                        process.join(10)
                    assert [process.exitcode for process in processes] == [0, 0]
                """),
                str(self.dataset_path / "checkpoint-gloo"),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

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

    def test_fork_transport_reports_serialization_failure_and_partial_reply(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                dedent("""
                    from concurrent.futures import ThreadPoolExecutor
                    from megatron.energon.dataloader.asynchronous.base import (
                        WorkerExitedError, WorkerSerializationError,
                    )
                    from megatron.energon.dataloader.asynchronous.fork import ForkAsynchronous

                    class Worker(ForkAsynchronous):
                        def __init__(self):
                            self._asynchronous_init("probe")

                        def echo(self, value):
                            return value

                        def unpicklable_result(self):
                            return lambda: None

                        def unpicklable_exception(self):
                            raise RuntimeError(lambda: None)

                        def large_reply(self):
                            return b"x" * (8 * 1024 * 1024)

                    worker, sibling, partial = Worker(), Worker(), Worker()
                    with ThreadPoolExecutor(3) as pool:
                        list(pool.map(lambda current: current.start(), (worker, sibling, partial)))
                    try:
                        with_error = (
                            lambda: worker._worker_call(worker.echo, lambda: None),
                            lambda: worker._worker_call(worker.unpicklable_result).get(2),
                            lambda: worker._worker_call(worker.unpicklable_exception).get(2),
                        )
                        for call in with_error:
                            try:
                                call()
                            except WorkerSerializationError:
                                pass
                            else:
                                raise AssertionError("Serialization failure should reach the caller")
                        assert worker._worker_call(worker.echo, 7).get(2) == 7

                        reply = partial._worker_call(partial.large_reply)
                        assert partial._result_queue._connection.poll(3)
                        partial._process.kill()
                        partial._process.join(3)
                        try:
                            reply.get(2)
                        except WorkerExitedError as error:
                            assert "exited with code -9" in str(error), error
                        else:
                            raise AssertionError("Expected failure after a partial reply")
                        assert sibling._worker_call(sibling.echo, 8).get(2) == 8
                    finally:
                        worker.shutdown()
                        sibling.shutdown()
                        partial.shutdown()
                """),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fork_shutdown_deadline_covers_parent_reply_decoding(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                dedent("""
                    import threading
                    from megatron.energon.dataloader.asynchronous.fork import ForkAsynchronous

                    entered, release = threading.Event(), threading.Event()

                    def decode():
                        entered.set()
                        release.wait()
                        return 1

                    class Reply:
                        def __reduce__(self):
                            return decode, ()

                    class Worker(ForkAsynchronous):
                        _shutdown_timeout = 0.5
                        def __init__(self): self._asynchronous_init("decode-probe")
                        def reply(self): return Reply()

                    worker = Worker()
                    worker.start()
                    pending = worker._worker_call(worker.reply)
                    try:
                        try:
                            worker.shutdown()
                        except TimeoutError:
                            assert entered.is_set()
                            assert pending.done()
                            assert worker._process is None
                        else:
                            raise AssertionError("Expected a deadline while decoding")
                    finally:
                        release.set()
                        worker.shutdown()
                    assert not worker.running()
                """),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fork_loader_rolls_back_startup_and_stops_blocked_worker(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                dedent("""
                    import multiprocessing
                    from torch.utils.data import get_worker_info
                    from megatron.energon import WorkerConfig
                    from megatron.energon.dataloader import DataLoader, ForkDataLoaderWorker

                    context = multiprocessing.get_context("fork")
                    blocked = context.Event()
                    release = context.Event()

                    class Worker(ForkDataLoaderWorker):
                        _shutdown_timeout = 0.5

                    class Dataset:
                        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=1)
                        def worker_init(self): pass
                        def worker_close(self): pass
                        def reset_state_deep(self): pass
                        def close(self): pass

                        def __iter__(self):
                            yield get_worker_info().id
                            blocked.set()
                            release.wait()

                    class FailingDataset(Dataset):
                        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=2)

                        def worker_init(self):
                            if get_worker_info().id == 1:
                                raise RuntimeError("initialization failed")

                    def loader(dataset):
                        return DataLoader(
                            dataset, worker_type=Worker, prefetch_factor=1, pin_memory=None,
                            watchdog_timeout_seconds=None, gc_collect_every_n_steps=0,
                        )

                    failed = loader(FailingDataset())
                    try:
                        next(iter(failed))
                    except RuntimeError as error:
                        assert str(error) == "initialization failed", error
                    else:
                        raise AssertionError("Expected worker initialization failure")
                    assert failed._workers is None and not multiprocessing.active_children()

                    running = loader(Dataset())
                    assert next(iter(running)) == 0 and blocked.wait(3)
                    try:
                        running.close()
                    except TimeoutError:
                        pass
                    else:
                        raise AssertionError("Expected forced shutdown to report a timeout")
                    assert running._workers is None and not multiprocessing.active_children()
                """),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_loader_joins_pinning_and_watchdog_threads(self):
        class CustomBatch:
            def pin_memory(self):
                return "pinned"

        wc = WorkerConfig(rank=0, world_size=1, num_workers=2)
        self.assertEqual(pickle.loads(pickle.dumps(wc)).config(), wc.config())
        dataset = get_train_dataset(
            self.ds1_path,
            worker_config=wc,
            batch_size=None,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            repeat=False,
        )
        pinning = PinMemoryThread("cpu")
        existing = set(threading.enumerate())
        with DataLoader(dataset, worker_type=ThreadDataLoaderWorker, pin_memory=pinning) as loader:
            self.assertEqual(pinning(DoneFuture(CustomBatch())).get(), "pinned")
            pinned = pinning(DoneFuture(OrderedDict(item=CustomBatch()))).get()
            self.assertIsInstance(pinned, OrderedDict)
            self.assertEqual(pinned, {"item": "pinned"})
            for _ in range(2):
                self.assertEqual(sum(1 for _ in loader), 55)
        self.assertFalse(pinning.running())
        self.assertEqual(set(threading.enumerate()) - existing, set())

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
            self.assertEqual(key["sample_idx"], first.__restore_key__.sample_idx)
            self.assertEqual(key["inner"]["type"], "MapGenRestoreKey")

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

        fields = json.loads(json.dumps(batch.to_json_object(), allow_nan=False))
        self.assertEqual(fields["type"], "BatchGenRestoreKey")
        self.assertEqual(fields["gen_idx"], 1)
        partial_json, missing = fields["inner"]
        self.assertIsNone(missing)
        self.assertEqual(partial_json["type"], "PartialSampleRestoreKey")
        self.assertEqual(
            partial_json["slice"],
            {"type": "slice", "start": 1, "stop": 5, "step": None},
        )
        self.assertEqual(
            partial_json["inner"],
            {"type": "IndexedRestoreKey", "kind": "Webdataset", "index": 7},
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
