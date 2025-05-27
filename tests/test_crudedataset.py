# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for crude datasets."""

import gc
import logging
import pickle
import re
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
    TextSample,
    WorkerConfig,
    basic_sample_keys,
    get_savable_loader,
    get_train_dataset,
    stateless,
)
from megatron.energon.cache import FileStore, FileStoreCachePool, Lazy
from megatron.energon.cache.base import CachePool
from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.sample_decoder import SampleDecoder
from megatron.energon.flavors.webdataset.structs import reraise_exception
from megatron.energon.source_info import SourceInfo
from megatron.energon.task_encoder.cooking import cooker


@edataclass
class LazyTextSample(Sample):
    txt: str
    next_txt: Lazy[str]


# Speed up tests significantly by reducing the torch status check interval for broken worker shutdown
try:
    torch.utils.data._utils.worker.MP_STATUS_CHECK_INTERVAL = 0.1
    torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 0.1
except AttributeError:
    pass


@edataclass
class TextBatch(Batch):
    txts: List[str]


@stateless
def cook_text(sample: dict) -> TextSample:
    return TextSample(
        **basic_sample_keys(sample),
        text=f"<{sample['txt']}>",
    )


@stateless
def cook_other(sample: dict) -> TextSample:
    d = pickle.loads(sample["pkl"])
    return TextSample(
        **basic_sample_keys(sample),
        text=f"<{sample['txt']}|{d['idx']}>",
    )


@stateless
def cook_aux(sample: dict, pkl_source: FileStore, fs_source: FileStore) -> TextSample:
    # ds2 is offset by 100
    d = pkl_source.get(f"{int(sample['txt']) + 100:06d}.txt", sample)
    return TextSample(
        **basic_sample_keys(sample),
        text=f"<{sample['txt']}|aux|{d}>",
    )


class CookingTaskEncoder(DefaultTaskEncoder[TextSample, TextSample, TextBatch, TextBatch]):
    """A simple task encoder for captioning."""

    cookers = [
        Cooker(cook_text, has_subflavors={"crude_type": "txtpkl"}),
        Cooker(cook_other, has_subflavors={"crude_type": "otherpkl"}),
        Cooker(cook_aux, has_subflavors={"crude_type": "aux_random_access"}),
    ]

    def batch(self, samples: List[TextSample]) -> TextBatch:
        return TextBatch.from_samples(
            samples,
            txts=[sample.text for sample in samples],
        )

    def select_samples_to_pack(self, samples):
        return [[sample] for sample in samples]

    @stateless
    def pack_selected_samples(self, samples):
        return samples[0]


@stateless
def cook_aux_filesystem_reference(
    sample: dict, pkl_source: FileStore, fs_source: FileStore
) -> TextSample:
    d = fs_source.get("aux_metadataset.yaml", sample)[:25].decode()
    return TextSample(
        **basic_sample_keys(sample),
        text=f"<{sample['txt']}|aux|{d}>",
    )


class CookingTaskEncoderWithAuxFilesystemReference(CookingTaskEncoder):
    cookers = [
        Cooker(cook_aux_filesystem_reference, has_subflavors={"crude_type": "aux_random_access"}),
    ]


@stateless
@cooker(need_cache=True, need_primary=True)
def cook_aux_primary_cache(
    sample: dict, primary: FileStore, pkl_source: FileStore, fs_source: FileStore, cache: CachePool
) -> LazyTextSample:
    # ds2 is offset by 100
    d = pkl_source.get(f"{int(sample['txt']) + 100:06d}.txt", sample)
    my_lazy_next_txt = cache.get_lazy(primary, f"{(int(sample['txt']) + 1) % 55:06d}.txt")
    return LazyTextSample(
        **basic_sample_keys(sample),
        txt=f"<{sample['txt']}|aux|{d}>",
        next_txt=my_lazy_next_txt,
    )


class LazyCookingTaskEncoder(
    DefaultTaskEncoder[LazyTextSample, LazyTextSample, TextBatch, TextBatch]
):
    # Classvar is fine here.
    decoder = SampleDecoder(image_decode="pilrgb")

    cookers = [
        Cooker(cook_aux_primary_cache, has_subflavors={"crude_type": "aux_random_access"}),
    ]

    def select_samples_to_pack(self, samples: List[LazyTextSample]) -> List[List[LazyTextSample]]:
        return [[sample] for sample in samples]

    @stateless
    def pack_selected_samples(self, samples: List[LazyTextSample]) -> TextSample:
        assert len(samples) == 1, f"Expected 1 sample, got {len(samples)}"
        next_txt = samples[0].next_txt.get(samples[0])
        return TextSample.derive_from(
            samples[0],
            text=samples[0].txt + "|" + next_txt,
        )

    def batch(self, samples: List[TextSample]) -> TextBatch:
        return TextBatch.from_samples(
            samples,
            txts=[sample.text for sample in samples],
        )


class LazyCookingTaskEncoderWithPostencode(
    DefaultTaskEncoder[LazyTextSample, LazyTextSample, TextBatch, TextBatch]
):
    # Classvar is fine here.
    decoder = SampleDecoder(image_decode="pilrgb")

    cookers = [
        Cooker(cook_aux_primary_cache, has_subflavors={"crude_type": "aux_random_access"}),
    ]

    @stateless
    def postencode_sample(self, sample: LazyTextSample) -> TextSample:
        assert isinstance(sample, LazyTextSample)
        return TextSample.derive_from(
            sample,
            text=sample.txt + "|" + sample.next_txt.get(sample),
        )

    def select_samples_to_pack(self, samples: List[LazyTextSample]) -> List[List[LazyTextSample]]:
        return [[sample] for sample in samples]

    @stateless
    def pack_selected_samples(self, samples: List[TextSample]) -> TextSample:
        assert len(samples) == 1
        return samples[0]

    def batch(self, samples: List[TextSample]) -> TextBatch:
        return TextBatch.from_samples(
            samples,
            txts=[sample.text for sample in samples],
        )


class GenericCookingTaskEncoder(DefaultTaskEncoder[TextSample, TextSample, TextBatch, TextBatch]):
    """A simple task encoder for captioning."""

    cookers = [Cooker(cook_text)]

    def batch(self, samples: List[TextSample]) -> TextBatch:
        return TextBatch.from_samples(
            samples,
            txts=[sample.text for sample in samples],
        )


class TestDataset(unittest.TestCase):
    # Set up the test fixture
    def setUp(self):
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        warnings.simplefilter("ignore", ResourceWarning)

        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name)
        # self.dataset_path = Path("./test_dataset")

        self.dataset_path.mkdir(exist_ok=True, parents=True)

        (self.dataset_path / "ds1").mkdir(exist_ok=True, parents=True)
        (self.dataset_path / "ds2").mkdir(exist_ok=True, parents=True)

        # Create a small dummy captioning dataset
        self.create_crude_text_test_dataset(self.dataset_path / "ds1", 0)
        self.create_crude_text_test_dataset(self.dataset_path / "ds2", 100)

        self.mds_path = self.dataset_path / "metadataset.yaml"
        with open(self.mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Metadataset",
                        "splits:",
                        "  train:",
                        "    datasets:",
                        "      - weight: 1",
                        "        path: ds1",
                        "        subflavor: ds1",
                        "        subflavors:",
                        "          source: metadataset.yaml",
                        "          number: 43",
                        "          mds: mds",
                        "          crude_type: txtpkl",
                        "        shuffle_over_epochs_multiplier: 3",
                        "      - weight: 1",
                        "        path: ds2",
                        "        subflavor: ds2",
                        "        subflavors:",
                        "          source: metadataset.yaml",
                        "          number: 44",
                        "          mds: mds",
                        "          crude_type: otherpkl",
                        "  val:",
                        "    datasets:",
                        "      - weight: 1",
                        "        path: ds1",
                        "        split_part: train",
                        "      - weight: 1",
                        "        path: ds2",
                        "        split_part: train",
                    ]
                )
            )

        self.aux_mds_path = self.dataset_path / "aux_metadataset.yaml"
        with open(self.aux_mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: MetadatasetV2",
                        "splits:",
                        "  train:",
                        "    path: ds1",
                        "    aux:",
                        "      pkl_source: ds2",
                        "      fs_source: filesystem://.",
                        "    subflavors:",
                        "      crude_type: aux_random_access",
                    ]
                )
            )

        print(self.dataset_path)

    def tearDown(self):
        # Remove all temporary files
        gc.collect()
        self.temp_dir.cleanup()

    @staticmethod
    def create_crude_text_test_dataset(path: Path, offset: int):
        """Creates a small dummy test dataset for testing purposes."""

        # Create num_samples unique captions
        (path / "parts").mkdir(exist_ok=True, parents=True)

        # Initialize the ShardWriter
        with wds.ShardWriter(f"{path}/parts/data-%d.tar", maxcount=10) as shard_writer:
            for idx in range(55):
                # Write individual files to shards
                shard_writer.write(
                    {
                        "__key__": f"{idx + offset:06d}",
                        "txt": f"{idx + offset}".encode(),
                        "pkl": pickle.dumps({"idx": idx + offset}),
                    },
                )
            total_shards = shard_writer.shard

        from megatron.energon.flavors import BaseWebdatasetFactory

        BaseWebdatasetFactory.prepare_dataset(
            path,
            [f"parts/data-{{0..{total_shards - 1}}}.tar"],
            split_parts_ratio=[("train", 1.0)],
            shuffle_seed=None,
            workers=1,
        )

        with open(path / MAIN_FOLDER_NAME / "dataset.yaml", "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: CrudeWebdataset",
                        "subflavors:",
                        "  dataset.yaml: true",
                        "  number: 42",
                    ]
                )
            )

    def test_metadataset(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
        )

        # Train mode dataset
        torch.manual_seed(42)
        train_dataset = get_train_dataset(
            self.mds_path,
            worker_config=worker_config,
            batch_size=3,
            task_encoder=CookingTaskEncoder(),
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            handler=reraise_exception,
        )
        loader = get_savable_loader(
            train_dataset,
        )

        print(len(train_dataset))
        # assert len(train_dataset) == 11

        for idx, data in enumerate(loader):
            if idx >= len(train_dataset):
                break

            assert isinstance(data, TextBatch)

            print("Batch", idx)
            for txt, key in zip(data.txts, data.__key__):
                key_int = int(key.split("/")[-1])
                if key_int < 100:
                    assert txt == f"<{key_int}>"
                else:
                    assert txt == f"<{key_int}|{key_int}>"

                print(key, txt)

    def test_loader(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=2,
        )

        loader = get_savable_loader(
            get_train_dataset(
                self.mds_path,
                batch_size=2,
                worker_config=worker_config,
                task_encoder=CookingTaskEncoder(),
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                packing_buffer_size=2,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )
        samples = [s.__key__ for idx, s in zip(range(100), loader)]

        print(samples)

        state = loader.save_state_rank()

        samples_after = [s.__key__ for idx, s in zip(range(100, 200), loader)]
        print(samples_after)

        loader = get_savable_loader(
            get_train_dataset(
                self.mds_path,
                batch_size=2,
                worker_config=worker_config,
                task_encoder=CookingTaskEncoder(),
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                packing_buffer_size=2,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )

        loader.restore_state_rank(state)

        samples_restored = [s.__key__ for idx, s in zip(range(100, 200), loader)]
        print(samples_restored)

        assert all([a == b for a, b in zip(samples_after, samples_restored)])

    def test_aux_random_access(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=2,
        )

        print("Initializing dataset")

        loader = get_savable_loader(
            get_train_dataset(
                self.aux_mds_path,
                batch_size=2,
                worker_config=worker_config,
                task_encoder=CookingTaskEncoder(),
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                packing_buffer_size=2,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )

        print("Iterating from dataset")
        samples = [s.txts for idx, s in zip(range(100), loader)]
        for idx, txts in enumerate(samples):
            for txt in txts:
                m = re.fullmatch(r"<([0-9]*)\|aux\|([0-9]*)>", txt)
                assert m, f"Invalid aux text: {txt}"
                assert int(m.group(2)) == int(m.group(1)) + 100

        print(samples)

        state = loader.save_state_rank()

        samples_after = [s.__key__ for idx, s in zip(range(100, 200), loader)]
        print(samples_after)

        loader = get_savable_loader(
            get_train_dataset(
                self.aux_mds_path,
                batch_size=2,
                worker_config=worker_config,
                task_encoder=CookingTaskEncoder(),
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                packing_buffer_size=2,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )

        loader.restore_state_rank(state)

        samples_restored = [s.__key__ for idx, s in zip(range(100, 200), loader)]
        print(samples_restored)

        assert all([a == b for a, b in zip(samples_after, samples_restored)])

    def test_aux_random_access_with_cache(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=2,
        )

        print("Initializing dataset")

        loader = get_savable_loader(
            get_train_dataset(
                self.aux_mds_path,
                batch_size=2,
                worker_config=worker_config,
                task_encoder=LazyCookingTaskEncoder(),
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                packing_buffer_size=2,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
            cache_pool=FileStoreCachePool(
                parent_cache_dir=self.dataset_path / "cache",
                num_workers=1,
            ),
        )

        print("Iterating from dataset")
        samples = [s.txts for idx, s in zip(range(100), loader)]
        for idx, txts in enumerate(samples):
            for txt in txts:
                m = re.fullmatch(r"<([0-9]*)\|aux\|([0-9]*)>\|([0-9]*)", txt)
                assert m, f"Invalid aux text: {txt}"
                assert int(m.group(2)) == int(m.group(1)) + 100
                assert int(m.group(3)) == (int(m.group(1)) + 1) % 55

        print(samples)

        state = loader.save_state_rank()

        samples_after = [s.__key__ for idx, s in zip(range(100, 200), loader)]
        print(samples_after)

        loader = get_savable_loader(
            get_train_dataset(
                self.aux_mds_path,
                batch_size=2,
                worker_config=worker_config,
                task_encoder=CookingTaskEncoder(),
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                packing_buffer_size=2,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
            cache_pool=FileStoreCachePool(
                parent_cache_dir=self.dataset_path / "cache",
                num_workers=1,
            ),
        )

        loader.restore_state_rank(state)

        samples_restored = [s.__key__ for idx, s in zip(range(100, 200), loader)]
        print(samples_restored)

        assert all([a == b for a, b in zip(samples_after, samples_restored)])

    def test_aux_random_access_with_cache_and_postencode(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=2,
        )

        print("Initializing dataset")

        loader = get_savable_loader(
            get_train_dataset(
                self.aux_mds_path,
                batch_size=2,
                worker_config=worker_config,
                task_encoder=LazyCookingTaskEncoderWithPostencode(),
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                packing_buffer_size=2,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
            cache_pool=FileStoreCachePool(
                parent_cache_dir=self.dataset_path / "cache",
                num_workers=1,
            ),
        )

        print("Iterating from dataset")
        samples = [s.txts for idx, s in zip(range(100), loader)]
        for idx, txts in enumerate(samples):
            for txt in txts:
                m = re.fullmatch(r"<([0-9]*)\|aux\|([0-9]*)>\|([0-9]*)", txt)
                assert m, f"Invalid aux text: {txt}"
                assert int(m.group(2)) == int(m.group(1)) + 100
                assert int(m.group(3)) == (int(m.group(1)) + 1) % 55

        print(samples)

        state = loader.save_state_rank()

        samples_after = [s.__key__ for idx, s in zip(range(100, 200), loader)]
        print(samples_after)

        loader = get_savable_loader(
            get_train_dataset(
                self.aux_mds_path,
                batch_size=2,
                worker_config=worker_config,
                task_encoder=LazyCookingTaskEncoderWithPostencode(),
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                packing_buffer_size=2,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
            cache_pool=FileStoreCachePool(
                parent_cache_dir=self.dataset_path / "cache",
                num_workers=1,
            ),
        )

        loader.restore_state_rank(state)

        samples_restored = [s.__key__ for idx, s in zip(range(100, 200), loader)]
        print(samples_restored)

        assert all([a == b for a, b in zip(samples_after, samples_restored)])

        # Verify that the sources are correct
        sample_src_check = [s.__sources__ for idx, s in zip(range(1), loader)][0]
        print(sample_src_check)
        # NOTE: Auxiliary sources have string as index, not int
        assert sample_src_check == (
            # Primary source for the sample, reading all source files
            SourceInfo(
                dataset_path=EPath(self.dataset_path / "ds1"),
                index=10,
                shard_name="parts/data-1.tar",
                file_names=("000010.pkl", "000010.txt"),
            ),
            # Auxiliary source for the sample, reading from ds2
            SourceInfo(
                dataset_path=EPath(self.dataset_path / "ds2"),
                index="000110.txt",
                shard_name="parts/data-1.tar",
                file_names=("000110.txt",),
            ),
            # Auxiliary source for the sample, reading from ds1, but next sample
            SourceInfo(
                dataset_path=EPath(self.dataset_path / "ds1"),
                index="000011.txt",
                shard_name="parts/data-1.tar",
                file_names=("000011.txt",),
            ),
            SourceInfo(
                dataset_path=EPath(self.dataset_path / "ds1"),
                index=11,
                shard_name="parts/data-1.tar",
                file_names=("000011.pkl", "000011.txt"),
            ),
            SourceInfo(
                dataset_path=EPath(self.dataset_path / "ds2"),
                index="000111.txt",
                shard_name="parts/data-1.tar",
                file_names=("000111.txt",),
            ),
            SourceInfo(
                dataset_path=EPath(self.dataset_path / "ds1"),
                index="000012.txt",
                shard_name="parts/data-1.tar",
                file_names=("000012.txt",),
            ),
        )

    def test_aux_filesystem_reference(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
        )

        loader = get_savable_loader(
            get_train_dataset(
                self.aux_mds_path,
                batch_size=1,
                worker_config=worker_config,
                task_encoder=CookingTaskEncoderWithAuxFilesystemReference(),
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
            ),
        )

        sample = next(iter(loader))

        assert sample.txts[0].endswith("|aux|__module__: megatron.ener>")

    def test_nomds(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=2,
        )

        loader = get_savable_loader(
            get_train_dataset(
                self.dataset_path / "ds1",
                batch_size=2,
                worker_config=worker_config,
                task_encoder=GenericCookingTaskEncoder(),
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
            ),
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
        )
        samples = [s.__key__ for idx, s in zip(range(100), loader)]

        print(samples)
        assert len(samples) == 100


if __name__ == "__main__":
    unittest.main()
