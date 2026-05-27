# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import logging
import pickle
import sys
import tempfile
import unittest
import warnings
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
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from megatron.energon.logical_worker import LogicalWorkerAssignment
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
        assert wc.logical_workers_per_rank() == 4

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

    def test_invalid_multiple_raises(self) -> None:
        with self.assertRaises(ValueError):
            WorkerConfig(rank=0, world_size=1, num_workers=5, logical_workers=3)

    def test_stride_needed(self) -> None:
        wc = WorkerConfig(rank=0, world_size=1, num_workers=4, logical_workers=2)
        assert _stride_needed(wc)
        wc2 = WorkerConfig(rank=0, world_size=1, num_workers=2)
        assert not _stride_needed(wc2)


class TrackingCrudePackingTaskEncoder(
    DefaultTaskEncoder[IndexTextSample, IndexTextSample, IndexTextBatch, IndexTextBatch]
):
    """Crude webdataset encoder with pre/post encode, packing (2), and batching (3)."""

    cookers = [Cooker(cook_index_sample, has_subflavors={"crude_type": "index"})]

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
                        "subflavors:",
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
