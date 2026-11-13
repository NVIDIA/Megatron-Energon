# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import logging
import pickle
import sys
import tempfile
import unittest
import warnings
from contextlib import ExitStack
from pathlib import Path
from typing import List

import torch
import webdataset as wds

from megatron.energon import (
    Batch,
    Cooker,
    DefaultTaskEncoder,
    WorkerConfig,
    basic_sample_keys,
    cooker,
    get_savable_loader,
    get_train_dataset,
    skip_safe,
    stateless,
)
from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.common.manifest.paths import MAIN_FOLDER_NAME
from megatron.energon.logical_worker import LogicalWorkerAssignment
from megatron.energon.wrappers.limit_dataset import LimitDataset
from megatron.energon.wrappers.stride_dataset import _stride_needed

try:
    torch.utils.data._utils.worker.MP_STATUS_CHECK_INTERVAL = 0.1
    torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 0.1
except AttributeError:
    pass


@edataclass
class IndexTextSample(Sample):
    index: int
    text: str


@edataclass
class IndexTextBatch(Batch):
    indexes: list[int]


@cooker()
@stateless
def cook_index_sample(sample: dict) -> IndexTextSample:
    idx = int(sample["txt"])
    return IndexTextSample(
        **basic_sample_keys(sample),
        index=idx,
        text=sample["txt"],
    )


class TestWorkerConfigLogicalWorkers(unittest.TestCase):
    def test_default_logical_equals_physical(self) -> None:
        wc = WorkerConfig(rank=0, world_size=2, num_workers=4)
        assert wc.physical_worker_count() == 8
        assert wc.logical_worker_count() == 8

    def test_physical_gt_logical_assignment(self) -> None:
        wc = WorkerConfig(rank=0, world_size=1, num_workers=4, logical_workers=2)
        assert wc.physical_worker_count() == 4
        assert wc.logical_worker_count() == 2
        a0 = wc.logical_assignment_for_physical(0)
        assert a0 == LogicalWorkerAssignment(logical_global_worker_id=0, stride_offset=0, stride=2)
        a1 = wc.logical_assignment_for_physical(1)
        assert a1 == LogicalWorkerAssignment(logical_global_worker_id=0, stride_offset=1, stride=2)

    def test_logical_gt_physical_raises(self) -> None:
        with self.assertRaises(ValueError):
            WorkerConfig(rank=0, world_size=1, num_workers=2, logical_workers=4)

    def test_synchronized_logical_worker_schedule(self):
        config = WorkerConfig(rank=0, world_size=3, num_workers=2, logical_workers=2)
        self.assertEqual(
            [config.logical_assignment_for_physical(worker).stride_offset for worker in range(6)],
            [0, 2, 1, 1, 0, 2],
        )
        self.assertEqual(
            [
                [config.logical_worker_last_physical_worker(group, step) for group in range(2)]
                for step in range(5)
            ],
            [[1, 5], [2, 4], [1, 5], [2, 4], [1, 5]],
        )
        self.assertEqual(config.logical_worker_next_offset(2), 1)
        self.assertEqual(config.logical_worker_next_offset(1, 2), 0)
        self.assertEqual(config.logical_worker_next_offset(0, 2), 1)
        self.assertEqual(config.logical_worker_next_offset(2, 2), 2)
        with self.assertRaisesRegex(ValueError, "same logical group"):
            config.logical_worker_next_offset(0, 5)

    def test_invalid_multiple_raises(self) -> None:
        with self.assertRaises(ValueError):
            WorkerConfig(rank=0, world_size=1, num_workers=5, logical_workers=3)

    def test_stride_needed(self) -> None:
        wc = WorkerConfig(rank=0, world_size=1, num_workers=4, logical_workers=2)
        assert _stride_needed(wc)
        wc2 = WorkerConfig(rank=0, world_size=1, num_workers=2)
        assert not _stride_needed(wc2)

    def test_restore_uses_logical_worker_override_without_remapping(self) -> None:
        wc = WorkerConfig(rank=0, world_size=6, num_workers=0, logical_workers=2)
        wc.worker_activate(0, override_logical_global_rank=1)
        try:
            assert wc.logical_global_worker_id() == 1
        finally:
            wc.worker_deactivate()


class TrackingCrudePackingTaskEncoder(
    DefaultTaskEncoder[IndexTextSample, IndexTextSample, IndexTextBatch, IndexTextBatch]
):
    """Crude webdataset encoder with pre/post encode, packing (2), and batching (3)."""

    cookers = [Cooker(cook_index_sample, has_tags={"crude_type": "index"})]

    def __init__(self) -> None:
        super().__init__(raw_batch_type=IndexTextBatch, batch_type=IndexTextBatch)

        self.pre_indexes: List[int] = []
        self.post_indexes: List[int] = []
        self.select_buffer_indexes: List[int] = []
        self.pack_group_indexes: List[List[int]] = []
        self.batch_packed_indexes: List[int] = []

    @stateless
    def preencode_sample(self, sample: IndexTextSample) -> IndexTextSample:
        self.pre_indexes.append(sample.index)
        return sample

    @stateless
    @skip_safe
    def postencode_sample(self, sample: IndexTextSample) -> IndexTextSample:
        self.post_indexes.append(sample.index)
        return sample

    def select_samples_to_pack(self, samples: List[IndexTextSample]) -> List[List[IndexTextSample]]:
        self.select_buffer_indexes.extend([s.index for s in samples])
        return [samples[i : i + 2] for i in range(0, len(samples), 2)]

    @stateless
    @skip_safe
    def pack_selected_samples(self, pack: List[IndexTextSample]) -> IndexTextBatch:
        indexes = [s.index for s in pack]
        self.pack_group_indexes.append(indexes)
        return IndexTextBatch.from_samples(
            pack,
            indexes=indexes,
        )

    @stateless
    @skip_safe
    def batch(self, samples: List[IndexTextBatch]) -> IndexTextBatch:
        indexes = []
        for s in samples:
            indexes.extend(s.indexes)
        self.batch_packed_indexes.extend(indexes)
        return IndexTextBatch.from_samples(
            samples,
            indexes=indexes,
        )


class TrackingCrudePackingPostPackNotSkipSafeTaskEncoder(TrackingCrudePackingTaskEncoder):
    """Same pipeline, but post/pack are not marked skip-safe."""

    @stateless
    def postencode_sample(self, sample: IndexTextSample) -> IndexTextSample:
        self.post_indexes.append(sample.index)
        return sample

    @stateless
    def pack_selected_samples(self, pack: List[IndexTextSample]) -> IndexTextBatch:
        indexes = [s.index for s in pack]
        self.pack_group_indexes.append(indexes)
        return IndexTextBatch.from_samples(
            pack,
            indexes=indexes,
        )

    @stateless(skip_safe=True)
    def batch(self, samples: List[IndexTextBatch]) -> IndexTextBatch:
        indexes = []
        for s in samples:
            indexes.extend(s.indexes)
        self.batch_packed_indexes.extend(indexes)
        return IndexTextBatch.from_samples(
            samples,
            indexes=indexes,
        )


class SeededPackingTaskEncoder(TrackingCrudePackingTaskEncoder):
    @stateless(restore_seeds=True)
    def preencode_sample(self, sample: IndexTextSample) -> IndexTextSample:
        return IndexTextSample.derive_from(
            sample, index=sample.index * 100000 + int(torch.randint(100000, ())), text=sample.text
        )


def _distributed_logical_checkpoint(rank: int, dataset_path: str, rendezvous: str) -> None:
    """Exercise real collectives and migration between process and subprocess loaders."""
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        case = TestLogicalWorkerCrudeE2E()
        case.ds_path = Path(dataset_path)
        with case._checkpoint_loader(0, rank=rank, world_size=2) as reference:
            iterator = iter(reference)
            stream = [next(iterator).indexes for _ in range(18)]
        for old_workers, new_workers in ((0, 2), (2, 0)):
            with case._checkpoint_loader(
                old_workers, rank=rank, world_size=2, sparse=True
            ) as loader:
                iterator = iter(loader)
                for _ in range(max(1, old_workers) * 3):
                    next(iterator)
                state = loader.save_state_logical_global(0)
            with case._checkpoint_loader(new_workers, rank=rank, world_size=2) as loader:
                loader.restore_state_logical_global(state, src_rank=0)
                # Saving again before the first restored output must retain pending old-stride replay.
                state = loader.save_state_logical_global(0)
                loader.restore_state_logical_global(state, src_rank=0)
                iterator = iter(loader)
                consumed = max(1, old_workers) * 3
                actual = [next(iterator).indexes for _ in range(6)]
                assert actual == stream[consumed : consumed + 6], (rank, old_workers, actual)
    finally:
        torch.distributed.destroy_process_group()


class TestLogicalWorkerCrudeE2E(unittest.TestCase):
    NUM_SAMPLES = 60

    @staticmethod
    def create_index_crude_dataset(path: Path, offset: int, count: int) -> None:
        """Creates a small crude webdataset with numeric txt/pkl indices (see test_crudedataset)."""
        from megatron.energon.flavors import BaseWebdatasetFactory

        (path / "parts").mkdir(exist_ok=True, parents=True)

        with wds.ShardWriter(f"{path}/parts/data-%d.tar", maxcount=count) as shard_writer:
            for idx in range(count):
                value = idx + offset
                shard_writer.write(
                    {
                        "__key__": f"{value:06d}",
                        "txt": f"{value}".encode(),
                        "pkl": pickle.dumps({"idx": value}),
                    },
                )
            total_shards = shard_writer.shard

        BaseWebdatasetFactory.prepare_dataset(
            path,
            [f"parts/data-{{0..{total_shards - 1}}}.tar"],
            split_parts_ratio=[("train", 1.0)],
            shuffle_seed=None,
            workers=1,
            media_filter=None,
        )

        with open(path / MAIN_FOLDER_NAME / "dataset.yaml", "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: CrudeWebdataset",
                        "tags:",
                        "  crude_type: index",
                    ]
                )
            )

    def setUp(self) -> None:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        warnings.simplefilter("ignore", ResourceWarning)

        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name)
        self.dataset_path.mkdir(exist_ok=True, parents=True)

        self.create_index_crude_dataset(
            self.dataset_path / "ds",
            offset=0,
            count=self.NUM_SAMPLES,
        )

        self.ds_path = self.dataset_path / "ds"

    def tearDown(self) -> None:
        gc.collect()
        self.temp_dir.cleanup()

    def _checkpoint_loader(
        self,
        num_workers,
        *,
        logical_workers=2,
        sparse=False,
        rank=0,
        world_size=1,
        task_encoder=None,
        limit=None,
        reset_after_epoch=False,
        virtual_epoch_length=0,
    ):
        config = WorkerConfig(
            rank=rank,
            world_size=world_size,
            num_workers=num_workers,
            logical_workers=logical_workers,
        )
        dataset = get_train_dataset(
            str(self.ds_path),
            worker_config=config,
            task_encoder=task_encoder or TrackingCrudePackingTaskEncoder(),
            batch_size=3,
            packing_buffer_size=15,
            shuffle_buffer_size=None,
            max_samples_per_sequence=4,
            repeat=True,
            virtual_epoch_length=virtual_epoch_length,
        )
        if limit is not None:
            dataset = LimitDataset(
                dataset,
                length=limit,
                reset_after_epoch=reset_after_epoch,
                worker_config=config,
            )
        return get_savable_loader(
            dataset,
            checkpoint_every_sec=3600 if sparse else 0,
            checkpoint_every_min_n_samples=100 if sparse else 1,
            watchdog_timeout_seconds=None,
        )

    def test_physical_checkpoint_restores_stride_phase(self):
        for offset in range(4):
            with self.subTest(offset=offset):
                with self._checkpoint_loader(
                    0, logical_workers=1, rank=offset, world_size=4
                ) as loader:
                    iterator = iter(loader)
                    next(iterator)
                    state = pickle.loads(pickle.dumps(loader.save_state_rank()))
                    expected = [next(iterator).indexes for _ in range(3)]
                with self._checkpoint_loader(
                    0, logical_workers=1, rank=offset, world_size=4
                ) as loader:
                    loader.restore_state_rank(state)
                    iterator = iter(loader)
                    self.assertEqual([next(iterator).indexes for _ in range(3)], expected)

    def test_limit_without_reset_continues_strided_stream(self):
        for rank in range(3):
            with self.subTest(rank=rank):
                with self._checkpoint_loader(
                    0, logical_workers=1, rank=rank, world_size=3
                ) as reference:
                    iterator = iter(reference)
                    expected = [next(iterator).indexes for _ in range(12)]
                with self._checkpoint_loader(
                    0,
                    logical_workers=1,
                    rank=rank,
                    world_size=3,
                    limit=3,
                ) as loader:
                    actual = [batch.indexes for _ in range(4) for batch in loader]
                    self.assertEqual(actual, expected)

    def test_epochize_continues_strided_stream(self):
        for rank in range(3):
            with self.subTest(rank=rank):
                with self._checkpoint_loader(
                    0, logical_workers=1, rank=rank, world_size=3
                ) as reference:
                    iterator = iter(reference)
                    expected = [next(iterator).indexes for _ in range(12)]
                with self._checkpoint_loader(
                    0,
                    logical_workers=1,
                    rank=rank,
                    world_size=3,
                    virtual_epoch_length=3,
                ) as loader:
                    actual = [batch.indexes for _ in range(4) for batch in loader]
                    self.assertEqual(actual, expected)

    def test_limit_partial_iteration_resumes_strided_stream(self):
        with self._checkpoint_loader(0, logical_workers=1, rank=1, world_size=3) as reference:
            iterator = iter(reference)
            expected = [next(iterator).indexes for _ in range(6)]
        with self._checkpoint_loader(
            0,
            logical_workers=1,
            rank=1,
            world_size=3,
            limit=3,
        ) as loader:
            iterator = iter(loader)
            self.assertEqual(next(iterator).indexes, expected[0])
            iterator.close()
            self.assertEqual([batch.indexes for batch in loader], expected[1:3])
            self.assertEqual([batch.indexes for batch in loader], expected[3:6])

    def test_limit_with_reset_restarts_strided_stream(self):
        with self._checkpoint_loader(0, logical_workers=1, rank=1, world_size=3) as reference:
            iterator = iter(reference)
            expected = [next(iterator).indexes for _ in range(3)]
        with self._checkpoint_loader(
            0,
            logical_workers=1,
            rank=1,
            world_size=3,
            limit=3,
            reset_after_epoch=True,
        ) as loader:
            for _ in range(3):
                self.assertEqual([batch.indexes for batch in loader], expected)

    def test_uneven_epochs_preserve_worker_delivery_order(self):
        with self._checkpoint_loader(2, logical_workers=1) as reference:
            iterator = iter(reference)
            expected = [next(iterator).indexes for _ in range(12)]
        for boundary in ({"virtual_epoch_length": 3}, {"limit": 3}):
            with self.subTest(boundary=boundary):
                with self._checkpoint_loader(2, logical_workers=1, **boundary) as loader:
                    self.assertEqual(len(loader), 3)
                    first = [batch.indexes for batch in loader]
                    state = pickle.loads(pickle.dumps(loader.save_state_logical_rank()))
                    actual = first + [batch.indexes for _ in range(3) for batch in loader]
                    self.assertEqual(actual, expected)
                with self._checkpoint_loader(3, logical_workers=1, **boundary) as loader:
                    loader.restore_state_logical_rank(state)
                    actual = [batch.indexes for _ in range(3) for batch in loader]
                    self.assertEqual(actual, expected[3:])

    def test_checkpoint_preserves_epoch_and_limit_progress(self):
        boundaries = (
            {"virtual_epoch_length": 8},
            {"limit": 8},
            {"limit": 8, "reset_after_epoch": True},
        )
        for boundary in boundaries:
            for consumed in (0, 2, 5, 8):
                for logical in (False, True):
                    with self.subTest(boundary=boundary, consumed=consumed, logical=logical):
                        with self._checkpoint_loader(
                            2, logical_workers=1, sparse=True, **boundary
                        ) as loader:
                            iterator = iter(loader)
                            for _ in range(consumed):
                                next(iterator)
                            state = pickle.loads(
                                pickle.dumps(
                                    loader.save_state_logical_rank()
                                    if logical
                                    else loader.save_state_rank()
                                )
                            )
                            remaining = [batch.indexes for batch in iterator]
                            next_epoch = [batch.indexes for batch in loader]
                        with self._checkpoint_loader(
                            3 if logical else 2, logical_workers=1, sparse=True, **boundary
                        ) as loader:
                            if logical:
                                loader.restore_state_logical_rank(state)
                            else:
                                loader.restore_state_rank(state)
                            self.assertEqual([batch.indexes for batch in loader], remaining)
                            self.assertEqual([batch.indexes for batch in loader], next_epoch)

    def test_synchronized_ranks_keep_each_logical_stream_in_order(self):
        with self._checkpoint_loader(2) as reference:
            iterator = iter(reference)
            streams = [[], []]
            for i in range(60):
                streams[i % 2].append(next(iterator).indexes)
        groups_by_step = ((0, 0, 1), (0, 1, 1))
        offsets_by_step = ((0, 1, 0), (2, 1, 2))
        for consumed in (0, 1, 2, 3):
            with self.subTest(consumed=consumed):
                with ExitStack() as stack:
                    loaders = [
                        stack.enter_context(
                            self._checkpoint_loader(2, rank=rank, world_size=3, sparse=True)
                        )
                        for rank in range(3)
                    ]
                    iterators = [iter(loader) for loader in loaders] if consumed else []
                    for step in range(consumed):
                        for rank, iterator in enumerate(iterators):
                            self.assertEqual(
                                next(iterator).indexes,
                                streams[groups_by_step[step % 2][rank]][
                                    step // 2 * 3 + offsets_by_step[step % 2][rank]
                                ],
                            )
                    self.assertEqual(
                        [loader.get_logical_worker_group_ids(for_save=True) for loader in loaders],
                        [(), (0,), (1,)] if consumed % 2 else [(0,), (), (1,)],
                    )
                    states = [
                        state
                        for loader in loaders
                        if (state := loader.save_state_logical_rank()) is not None
                    ]
                    states = pickle.loads(pickle.dumps(states))
                    expected_positions = [
                        consumed // 2 * 3 + 2 * (consumed % 2),
                        consumed // 2 * 3 + consumed % 2,
                    ]
                    for group_state in states:
                        for group, checkpoint in group_state.worker_states.items():
                            snapshot_index = (
                                0 if checkpoint.state is None else checkpoint.state.sample_index
                            )
                            self.assertEqual(
                                snapshot_index + checkpoint.offset, expected_positions[group]
                            )
                    if not iterators:
                        iterators = [iter(loader) for loader in loaders]
                    expected = [
                        [next(iterator).indexes for iterator in iterators] for _ in range(3)
                    ]
                with ExitStack() as stack:
                    loaders = [
                        stack.enter_context(
                            self._checkpoint_loader(2, rank=rank, world_size=3, sparse=True)
                        )
                        for rank in range(3)
                    ]
                    for loader in loaders:
                        loader.restore_state_logical_rank(states)
                    # A second save before iteration must preserve both replay cadence and phase.
                    states = [
                        state
                        for loader in loaders
                        if (state := loader.save_state_logical_rank()) is not None
                    ]
                    for loader in loaders:
                        loader.restore_state_logical_rank(states)
                    iterators = [iter(loader) for loader in loaders]
                    self.assertEqual(
                        [[next(iterator).indexes for iterator in iterators] for _ in range(3)],
                        expected,
                    )

    def test_partial_logical_round_changes_physical_fanout(self):
        with self._checkpoint_loader(2) as reference:
            iterator = iter(reference)
            streams = [[], []]
            for i in range(60):
                streams[i % 2].append(next(iterator).indexes)
        for consumed in (1, 3):
            with ExitStack() as stack:
                loaders = [
                    stack.enter_context(
                        self._checkpoint_loader(2, rank=rank, world_size=3, sparse=True)
                    )
                    for rank in range(3)
                ]
                iterators = [iter(loader) for loader in loaders]
                for _ in range(consumed):
                    for iterator in iterators:
                        next(iterator)
                states = [
                    state
                    for loader in loaders
                    if (state := loader.save_state_logical_rank()) is not None
                ]
                states = pickle.loads(pickle.dumps(states))
            with ExitStack() as stack:
                loaders = [
                    stack.enter_context(
                        self._checkpoint_loader(4, rank=rank, world_size=3, sparse=True)
                    )
                    for rank in range(3)
                ]
                for loader in loaders:
                    loader.restore_state_logical_rank(states)
                states = [
                    state
                    for loader in loaders
                    if (state := loader.save_state_logical_rank()) is not None
                ]
                for loader in loaders:
                    loader.restore_state_logical_rank(states)
                iterators = [iter(loader) for loader in loaders]
                cursors = [consumed // 2 * 3 + 2, consumed // 2 * 3 + 1]
                for step in range(8):
                    for rank, iterator in enumerate(iterators):
                        group = (rank * 4 + (consumed + step) % 4) // 6
                        self.assertEqual(next(iterator).indexes, streams[group][cursors[group]])
                        cursors[group] += 1

    def test_physical_checkpoint_preserves_next_worker(self):
        with self._checkpoint_loader(4, sparse=True) as loader:
            iterator = iter(loader)
            for _ in range(3):
                next(iterator)
            state = pickle.loads(pickle.dumps(loader.save_state_rank()))
            self.assertEqual(state.next_worker_id, 3)
            expected = [next(iterator).indexes for _ in range(8)]
        with self._checkpoint_loader(4) as loader:
            loader.restore_state_rank(state)
            iterator = iter(loader)
            self.assertEqual([next(iterator).indexes for _ in range(8)], expected)

    def test_physical_save_before_logical_restore_iteration(self):
        with self._checkpoint_loader(2, logical_workers=1, sparse=True) as loader:
            iterator = iter(loader)
            for _ in range(3):
                next(iterator)
            state = pickle.loads(pickle.dumps(loader.save_state_logical_global()))
        for workers in (0, 3):
            with self.subTest(workers=workers):
                with self._checkpoint_loader(workers, logical_workers=1) as loader:
                    loader.restore_state_logical_rank(state)
                    physical_state = pickle.loads(pickle.dumps(loader.save_state_rank()))
                    iterator = iter(loader)
                    expected = [next(iterator).indexes for _ in range(6)]
                with self._checkpoint_loader(workers, logical_workers=1) as loader:
                    loader.restore_state_rank(physical_state)
                    iterator = iter(loader)
                    self.assertEqual([next(iterator).indexes for _ in range(6)], expected)

    def test_logical_checkpoint_changes_physical_fanout(self):
        with self._checkpoint_loader(2, task_encoder=SeededPackingTaskEncoder()) as reference:
            iterator = iter(reference)
            streams = [[], []]
            for i in range(60):
                streams[i % 2].append(next(iterator).indexes)
        for old_workers, new_workers in ((2, 4), (4, 2), (4, 6), (6, 4)):
            for sparse in (False, True):
                with self.subTest(old=old_workers, new=new_workers, sparse=sparse):
                    with self._checkpoint_loader(
                        old_workers, sparse=sparse, task_encoder=SeededPackingTaskEncoder()
                    ) as loader:
                        iterator = iter(loader)
                        for _ in range(old_workers * 3):
                            next(iterator)
                        state = pickle.loads(pickle.dumps(loader.save_state_logical_global(0)))
                        self.assertEqual(len(state.worker_states), 2)
                    with self._checkpoint_loader(
                        new_workers, task_encoder=SeededPackingTaskEncoder()
                    ) as loader:
                        loader.restore_state_logical_global(state)
                        iterator = iter(loader)
                        fanout = new_workers // 2
                        consumed = old_workers // 2 * 3
                        cursors = [consumed, consumed]
                        expected = []
                        for step in range(3 * new_workers):
                            logical = ((state.iteration_index + step) % new_workers) // fanout
                            expected.append(streams[logical][cursors[logical]])
                            cursors[logical] += 1
                        self.assertEqual([next(iterator).indexes for _ in expected], expected)

    def test_logical_global_checkpoint_preserves_next_worker(self):
        with self._checkpoint_loader(4, sparse=True) as loader:
            iterator = iter(loader)
            # Group zero completed a round; group one has not contributed yet.
            for _ in range(2):
                next(iterator)
            state = pickle.loads(pickle.dumps(loader.save_state_logical_global(0)))
            self.assertEqual(state.iteration_index, 2)
            expected = [next(iterator).indexes for _ in range(8)]
        with self._checkpoint_loader(4) as loader:
            loader.restore_state_logical_global(state)
            iterator = iter(loader)
            self.assertEqual([next(iterator).indexes for _ in range(8)], expected)

    def test_logical_checkpoint_restores_partial_round(self):
        with self._checkpoint_loader(4) as loader:
            iterator = iter(loader)
            next(iterator)
            state = pickle.loads(pickle.dumps(loader.save_state_logical_global(0)))
            expected = [next(iterator).indexes for _ in range(8)]
        with self._checkpoint_loader(4) as loader:
            loader.restore_state_logical_global(state)
            iterator = iter(loader)
            self.assertEqual([next(iterator).indexes for _ in range(8)], expected)

    def test_initial_logical_checkpoint(self):
        with self._checkpoint_loader(4) as loader:
            # A lazy iterator must not make a zero-output save wait for workers.
            iter(loader)
            state = pickle.loads(pickle.dumps(loader.save_state_logical_global(0)))
        with self._checkpoint_loader(2) as reference:
            iterator = iter(reference)
            expected = [next(iterator).indexes for _ in range(8)]
        with self._checkpoint_loader(2) as loader:
            loader.restore_state_logical_global(state)
            iterator = iter(loader)
            self.assertEqual([next(iterator).indexes for _ in range(8)], expected)

    def test_logical_rank_checkpoint_routes_groups(self):
        with self._checkpoint_loader(2) as reference:
            iterator = iter(reference)
            streams = [[], []]
            for i in range(40):
                streams[i % 2].append(next(iterator).indexes)
        for consumed in (0, 3):
            states = {}
            for rank in range(6):
                with self._checkpoint_loader(0, rank=rank, world_size=6) as loader:
                    self.assertEqual(loader.get_logical_worker_group_ids(), (rank // 3,))
                    self.assertEqual(
                        loader.get_logical_worker_group_ids(for_save=True),
                        (rank // 3,) if rank % 3 == 2 else (),
                    )
                    if consumed:
                        iterator = iter(loader)
                        for _ in range(consumed):
                            next(iterator)
                    state = loader.save_state_logical_rank()
                    if rank % 3 != 2:
                        self.assertIsNone(state)
                    else:
                        self.assertEqual(tuple(state.worker_states), (rank // 3,))
                        states[rank // 3] = pickle.loads(pickle.dumps(state))
            for rank in range(4):
                with self._checkpoint_loader(0, rank=rank, world_size=4) as loader:
                    (group,) = loader.get_logical_worker_group_ids()
                    loader.restore_state_logical_rank(states[group])
                    iterator = iter(loader)
                    self.assertEqual(
                        [next(iterator).indexes for _ in range(3)],
                        [streams[group][consumed * 3 + rank % 2 + step * 2] for step in range(3)],
                    )

    def test_logical_rank_checkpoint_multiple_groups_and_sparse_replay(self):
        with self._checkpoint_loader(2) as reference:
            iterator = iter(reference)
            streams = [[], []]
            for i in range(40):
                streams[i % 2].append(next(iterator).indexes)
        for consumed in (0, 12):
            with self._checkpoint_loader(4, sparse=True) as loader:
                self.assertEqual(loader.get_logical_worker_group_ids(for_save=True), (0, 1))
                if consumed:
                    iterator = iter(loader)
                    for _ in range(consumed):
                        next(iterator)
                state = pickle.loads(pickle.dumps(loader.save_state_logical_rank()))
            with self._checkpoint_loader(6, sparse=True) as loader:
                loader.restore_state_logical_rank(state)
                # Re-saving before iteration preserves the historical replay assignment.
                state = loader.save_state_logical_rank()
                serialized = pickle.dumps(state)
                self.assertNotIn(b"_pending_skip", serialized)
                self.assertNotIn(b"_replay_outputs", serialized)
                self.assertNotIn(b"LogicalWorkerRestore", serialized)
                self.assertNotIn(b"StrideDataset", serialized)
                self.assertNotIn(b"GcDataset", serialized)
                self.assertNotIn(b"WatchdogDataset", serialized)
                self.assertNotIn(b"_logical_sample_index", serialized)
                self.assertFalse(hasattr(state, "physical_workers"))
                self.assertFalse(hasattr(state, "workers_per_rank"))
                self.assertFalse(hasattr(state, "next_worker_ids"))
            with self._checkpoint_loader(2) as loader:
                loader.restore_state_logical_rank([state])
                iterator = iter(loader)
                self.assertEqual(
                    [next(iterator).indexes for _ in range(6)],
                    [
                        streams[group][consumed // 2 + step]
                        for step in range(3)
                        for group in range(2)
                    ],
                )

    def test_logical_rank_checkpoint_validates_mapping(self):
        with self._checkpoint_loader(4) as loader:
            state = loader.save_state_logical_rank()
            iterator = iter(loader)
            next(iterator)
            self.assertIsNotNone(loader.save_state_logical_rank())
        with self._checkpoint_loader(2) as loader:
            with self.assertRaisesRegex(ValueError, "Missing logical worker group"):
                loader.restore_state_logical_rank([])
            with self.assertRaisesRegex(ValueError, "Duplicate logical worker group"):
                loader.restore_state_logical_rank([state, state])

    def test_logical_checkpoint_after_fanout_change(self):
        with self._checkpoint_loader(2) as reference:
            iterator = iter(reference)
            streams = [[], []]
            for i in range(60):
                streams[i % 2].append(next(iterator).indexes)
        with self._checkpoint_loader(4, sparse=True) as loader:
            iterator = iter(loader)
            for _ in range(12):
                next(iterator)
            state = loader.save_state_logical_global(0)
        with self._checkpoint_loader(6, sparse=True) as loader:
            loader.restore_state_logical_global(state)
            iterator = iter(loader)
            for _ in range(12):
                next(iterator)
            state = pickle.loads(pickle.dumps(loader.save_state_logical_global(0)))
        with self._checkpoint_loader(2) as loader:
            loader.restore_state_logical_global(state)
            iterator = iter(loader)
            self.assertEqual(
                [next(iterator).indexes for _ in range(6)],
                [streams[logical][12 + step] for step in range(3) for logical in range(2)],
            )

    def test_logical_checkpoint_changes_rank_count(self):
        from unittest.mock import patch

        with ExitStack() as stack:
            loaders = [
                stack.enter_context(
                    self._checkpoint_loader(0, logical_workers=1, rank=rank, world_size=3)
                )
                for rank in range(3)
            ]
            for loader in loaders:
                iterator = iter(loader)
                for _ in range(2):
                    next(iterator)
            rank_states = [loader.save_state_logical_rank() for loader in loaders]

            def gather(local, output, *args, **kwargs):
                output[:] = rank_states

            with patch("torch.distributed.gather_object", side_effect=gather):
                state = loaders[0].save_state_logical_global(0)
            # The checkpoint contains the stream immediately after six logical outputs.
            last_iterator = iter(loaders[-1])
            expected_last = next(last_iterator).indexes
        with self._checkpoint_loader(0, logical_workers=1) as loader:
            loader.restore_state_logical_global(state)
            iterator = iter(loader)
            actual = [next(iterator).indexes for _ in range(3)]
            self.assertEqual(actual[-1], expected_last)

    def test_distributed_logical_checkpoint(self):
        torch.multiprocessing.spawn(
            _distributed_logical_checkpoint,
            args=(str(self.ds_path), str(self.dataset_path / "gloo-checkpoint")),
            nprocs=2,
            join=True,
        )

    def test_pre_on_all_post_and_pack_on_strided_physical_worker(self) -> None:
        """E2E: crude webdataset, real TaskEncoder pipeline.

        Models rank=0, world_size=6, num_workers=0, logical_workers=2 (6 physical, fanout 3):
        all 60 samples are loaded on rank 0 (world_size=1 for the test loader), and an
        explicit stride assignment for physical worker 0 yields indices 0, 3, 6, ...

        Pre-encode must run on every sample; post-encode, pack (2), and batch (3) only
        on the strided subset.
        """
        # Load all samples on rank 0 (no DataLoader subprocesses — stats stay in-process).
        wc = WorkerConfig(rank=0, world_size=2, num_workers=0, logical_workers=2)

        orig_te = TrackingCrudePackingTaskEncoder()

        orig_loader = get_savable_loader(
            get_train_dataset(
                str(self.ds_path),
                task_encoder=orig_te,
                worker_config=wc,
                batch_size=3,
                packing_buffer_size=self.NUM_SAMPLES // 4,
                shuffle_buffer_size=None,
                max_samples_per_sequence=4,
                repeat=False,
            )
        )

        orig_samples = list(orig_loader)
        print(f"orig_samples: {[batch.indexes for batch in orig_samples]}")

        # Models rank=0 with world_size=6, num_workers=0, logical_workers=2 (6 physical, fanout 3).
        stride1_wc = WorkerConfig(rank=0, world_size=6, num_workers=0, logical_workers=2)
        te1 = TrackingCrudePackingTaskEncoder()
        stride1_loader = get_savable_loader(
            get_train_dataset(
                str(self.ds_path),
                task_encoder=te1,
                worker_config=stride1_wc,
                batch_size=3,
                packing_buffer_size=self.NUM_SAMPLES // 4,
                shuffle_buffer_size=None,
                max_samples_per_sequence=4,
                repeat=False,
            )
        )
        stride1_samples = list(stride1_loader)
        print(f"stride1 samples: {[batch.indexes for batch in stride1_samples]}")

        # Models rank=1 with world_size=6, num_workers=0, logical_workers=2 (6 physical, fanout 3).
        stride2_wc = WorkerConfig(rank=1, world_size=6, num_workers=0, logical_workers=2)
        te2 = TrackingCrudePackingTaskEncoder()
        stride2_loader = get_savable_loader(
            get_train_dataset(
                str(self.ds_path),
                task_encoder=te2,
                worker_config=stride2_wc,
                batch_size=3,
                packing_buffer_size=self.NUM_SAMPLES // 4,
                shuffle_buffer_size=None,
                max_samples_per_sequence=4,
                repeat=False,
            )
        )
        stride2_samples = list(stride2_loader)
        print(f"stride2_samples: {[batch.indexes for batch in stride2_samples]}")

        # Models rank=2 with world_size=6, num_workers=0, logical_workers=2 (6 physical, fanout 3).
        stride3_wc = WorkerConfig(rank=2, world_size=6, num_workers=0, logical_workers=2)
        te3 = TrackingCrudePackingTaskEncoder()
        stride3_loader = get_savable_loader(
            get_train_dataset(
                str(self.ds_path),
                task_encoder=te3,
                worker_config=stride3_wc,
                batch_size=3,
                packing_buffer_size=self.NUM_SAMPLES // 4,
                shuffle_buffer_size=None,
                max_samples_per_sequence=4,
                repeat=False,
            )
        )
        stride3_samples = list(stride3_loader)
        print(f"stride3_samples: {[batch.indexes for batch in stride3_samples]}")

        # Combining the samples from ranks 0-2 should give the same samples as the original loader.
        # It should be precisely interleaved.
        strided_samples = [stride1_samples, stride2_samples, stride3_samples]
        strided_encoders = [te1, te2, te3]
        interleaved_stride_samples = [sample for group in zip(*strided_samples) for sample in group]
        print(
            f"interleaved_stride_samples: {[batch.indexes for batch in interleaved_stride_samples]}"
        )

        assert [batch.indexes for batch in interleaved_stride_samples] == [
            batch.indexes for batch in orig_samples
        ]

        for stride_offset, (stride_samples, te) in enumerate(
            zip(strided_samples, strided_encoders)
        ):
            expected_stride_samples = orig_samples[stride_offset::3]
            expected_stride_indexes = [
                index for batch in expected_stride_samples for index in batch.indexes
            ]
            print(f"stride_samples[{stride_offset}]: {[batch.indexes for batch in stride_samples]}")
            print(f"expected_stride_indexes[{stride_offset}]: {expected_stride_indexes}")
            assert [batch.indexes for batch in stride_samples] == [
                batch.indexes for batch in expected_stride_samples
            ]
            assert te.pre_indexes == orig_te.pre_indexes
            assert te.select_buffer_indexes == orig_te.select_buffer_indexes
            print(f"stride_samples[{stride_offset}]: {te.post_indexes}")
            print(f"expected_stride_indexes[{stride_offset}]: {expected_stride_indexes}")
            assert te.post_indexes == expected_stride_indexes
            assert [
                index for pack in te.pack_group_indexes for index in pack
            ] == expected_stride_indexes
            assert te.batch_packed_indexes == expected_stride_indexes

    def test_non_skip_safe_post_pack_run_on_skipped_outputs(self) -> None:
        """E2E: non-skip-safe post/pack still run for discarded physical outputs."""
        wc = WorkerConfig(rank=0, world_size=2, num_workers=0, logical_workers=2)

        orig_te = TrackingCrudePackingPostPackNotSkipSafeTaskEncoder()
        orig_loader = get_savable_loader(
            get_train_dataset(
                str(self.ds_path),
                task_encoder=orig_te,
                worker_config=wc,
                batch_size=3,
                packing_buffer_size=self.NUM_SAMPLES // 4,
                shuffle_buffer_size=None,
                max_samples_per_sequence=4,
                repeat=False,
            )
        )
        orig_samples = list(orig_loader)

        stride_wc = WorkerConfig(rank=0, world_size=6, num_workers=0, logical_workers=2)
        te = TrackingCrudePackingPostPackNotSkipSafeTaskEncoder()
        stride_loader = get_savable_loader(
            get_train_dataset(
                str(self.ds_path),
                task_encoder=te,
                worker_config=stride_wc,
                batch_size=3,
                packing_buffer_size=self.NUM_SAMPLES // 4,
                shuffle_buffer_size=None,
                max_samples_per_sequence=4,
                repeat=False,
            )
        )
        stride_samples = list(stride_loader)

        expected_stride_samples = orig_samples[::3]
        expected_all_indexes = [index for batch in orig_samples for index in batch.indexes]
        expected_stride_indexes = [
            index for batch in expected_stride_samples for index in batch.indexes
        ]

        assert [batch.indexes for batch in stride_samples] == [
            batch.indexes for batch in expected_stride_samples
        ]
        assert te.pre_indexes == orig_te.pre_indexes
        assert te.select_buffer_indexes == orig_te.select_buffer_indexes
        assert te.post_indexes == expected_all_indexes
        assert [index for pack in te.pack_group_indexes for index in pack] == expected_all_indexes
        assert te.batch_packed_indexes == expected_stride_indexes
