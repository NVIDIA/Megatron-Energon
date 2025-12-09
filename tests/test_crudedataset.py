# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for crude datasets."""

import gc
import io
import logging
import pickle
import re
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import List

import pytest
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
from megatron.energon.errors import reraise_exception
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.sample_decoder import SampleDecoder
from megatron.energon.media.extractor import MediaFilterConfig, MediaFilterStrategy
from megatron.energon.media.filesystem_prepare import prepare_filesystem_dataset
from megatron.energon.media.metadata import AVMetadata, ImageMetadata
from megatron.energon.source_info import SourceInfo
from megatron.energon.task_encoder.cooking import cooker


def _noise_image_bytes(size: tuple[int, int], fmt: str, seed: int) -> bytes:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(data.astype(np.uint8), "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def _metadata_description(metadata) -> str:
    if isinstance(metadata, ImageMetadata):
        return f"IMG-{metadata.width}x{metadata.height}-{metadata.format}"
    if isinstance(metadata, AVMetadata):
        if metadata.video_fps is not None:
            return f"VIDEO-{metadata.video_width}x{metadata.video_height}@{metadata.video_fps}fps-{metadata.video_duration:0.1f}s"
        elif metadata.audio_sample_rate is not None:
            return f"AUDIO-{metadata.audio_duration:0.1f}s@{metadata.audio_sample_rate}Hz"
        else:
            return "AV-UNKNOWN"
    return "UNKNOWN"


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


@stateless
@cooker(need_primary=True)
def cook_media_metadata(sample: dict, primary: FileStore, media: FileStore) -> TextSample:
    """This cooker loads the media from the primary and auxiliary datasets and
    returns a text sample with the metadata descriptions of each."""

    # print(f"Cooking media metadata for {sample}")
    filename = sample["__sources__"][0].file_names[0]

    primary_media_metadata = primary.get_media_metadata(filename)
    aux_media_metadata = media.get_media_metadata(filename)

    return TextSample(
        **basic_sample_keys(sample),
        text="|".join(
            [
                _metadata_description(primary_media_metadata),
                _metadata_description(aux_media_metadata),
            ]
        ),
    )


class CookingTaskEncoder(DefaultTaskEncoder[TextSample, TextSample, TextBatch, TextBatch]):
    """A simple task encoder for captioning."""

    cookers = [
        Cooker(cook_text, has_subflavors={"crude_type": "txtpkl"}),
        Cooker(cook_other, has_subflavors={"crude_type": "otherpkl"}),
        Cooker(cook_aux, has_subflavors={"crude_type": "aux_random_access"}),
        Cooker(cook_media_metadata, has_subflavors={"crude_type": "media_metadata"}),
    ]

    @stateless
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

    @stateless
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

    @stateless
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


@pytest.fixture
def dataset_path():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    warnings.simplefilter("ignore", ResourceWarning)

    # Create a temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    dataset_path = Path(temp_dir.name)
    # dataset_path = Path("./test_dataset")

    dataset_path.mkdir(exist_ok=True, parents=True)

    (dataset_path / "ds1").mkdir(exist_ok=True, parents=True)
    (dataset_path / "ds2").mkdir(exist_ok=True, parents=True)

    # Create a small dummy captioning dataset
    create_crude_text_test_dataset(dataset_path / "ds1", 0)
    create_crude_text_test_dataset(dataset_path / "ds2", 100)

    mds_path = dataset_path / "metadataset.yaml"
    with open(mds_path, "w") as f:
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
                    "        subflavors:",
                    "          source: metadataset.yaml",
                    "          number: 43",
                    "          mds: mds",
                    "          crude_type: txtpkl",
                    "        shuffle_over_epochs_multiplier: 3",
                    "      - weight: 1",
                    "        path: ds2",
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

    aux_mds_path = dataset_path / "aux_metadataset.yaml"
    with open(aux_mds_path, "w") as f:
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

    multimedia_wds_path = dataset_path / "multimedia_wds"
    create_multimedia_webdataset(multimedia_wds_path)

    multimedia_fs_path = dataset_path / "multimedia_fs"
    create_multimedia_filesystem_dataset(multimedia_fs_path)

    media_mds_path = dataset_path / "media_metadataset.yaml"
    with open(media_mds_path, "w") as f:
        f.write(
            "\n".join(
                [
                    "__module__: megatron.energon",
                    "__class__: MetadatasetV2",
                    "splits:",
                    "  train:",
                    "    path: multimedia_wds",
                    "    aux:",
                    "      media: filesystem://multimedia_fs",
                    "    subflavors:",
                    "      crude_type: media_metadata",
                ]
            )
        )

    print(dataset_path)

    yield dataset_path

    # Remove all temporary files
    gc.collect()
    temp_dir.cleanup()


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
        media_filter=None,
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


def create_multimedia_webdataset(path: Path):
    path.mkdir(exist_ok=True, parents=True)
    (path / "parts").mkdir(exist_ok=True, parents=True)

    jpg_bytes = _noise_image_bytes((32, 16), "JPEG", seed=0)
    png_bytes = _noise_image_bytes((24, 24), "PNG", seed=1)
    video_bytes = Path("tests/data/sync_test.mp4").read_bytes()
    audio_bytes = Path("tests/data/test_audio.flac").read_bytes()

    with wds.ShardWriter(f"{path}/parts/data-%d.tar", maxcount=10) as shard_writer:
        shard_writer.write({"__key__": "image000", "jpg": jpg_bytes})
        shard_writer.write({"__key__": "image001", "png": png_bytes})
        shard_writer.write({"__key__": "audio001", "flac": audio_bytes})
        shard_writer.write({"__key__": "video001", "mp4": video_bytes})
        total_shards = shard_writer.shard

    from megatron.energon.flavors import BaseWebdatasetFactory

    BaseWebdatasetFactory.prepare_dataset(
        path,
        [f"parts/data-{{0..{total_shards - 1}}}.tar"],
        split_parts_ratio=[("train", 1.0)],
        shuffle_seed=None,
        workers=1,
        media_filter=MediaFilterConfig(strategy=MediaFilterStrategy.EXTENSION),
    )

    with open(path / MAIN_FOLDER_NAME / "dataset.yaml", "w") as f:
        f.write(
            "\n".join(
                [
                    "__module__: megatron.energon",
                    "__class__: CrudeWebdataset",
                    "subflavors:",
                    "  crude_type: media_metadata",
                ]
            )
        )


@staticmethod
def create_multimedia_filesystem_dataset(path: Path):
    path.mkdir(exist_ok=True, parents=True)

    (path / "image000.jpg").write_bytes(_noise_image_bytes((32, 16), "JPEG", seed=0))
    (path / "image001.png").write_bytes(_noise_image_bytes((24, 24), "PNG", seed=1))
    shutil.copyfile("tests/data/sync_test.mp4", path / "video001.mp4")
    shutil.copyfile("tests/data/test_audio.flac", path / "audio001.flac")

    prepare_filesystem_dataset(
        EPath(path), MediaFilterConfig(strategy=MediaFilterStrategy.EXTENSION), progress=False
    )


def test_metadataset(dataset_path):
    torch.manual_seed(42)
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=0,
        global_error_handler=reraise_exception,
    )

    # Train mode dataset
    torch.manual_seed(42)
    train_dataset = get_train_dataset(
        dataset_path / "metadataset.yaml",
        worker_config=worker_config,
        batch_size=3,
        task_encoder=CookingTaskEncoder(),
        shuffle_buffer_size=None,
        max_samples_per_sequence=None,
    )
    with get_savable_loader(
        train_dataset,
    ) as loader:
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


def test_loader(dataset_path):
    torch.manual_seed(42)
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=2,
    )

    with get_savable_loader(
        get_train_dataset(
            dataset_path / "metadataset.yaml",
            batch_size=2,
            worker_config=worker_config,
            task_encoder=CookingTaskEncoder(),
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            packing_buffer_size=2,
        ),
    ) as loader:
        samples = [s.__key__ for idx, s in zip(range(100), loader)]

        print(samples)

        state = loader.save_state_rank()

        samples_after = [s.__key__ for idx, s in zip(range(100, 200), loader)]
        print(samples_after)

    with get_savable_loader(
        get_train_dataset(
            dataset_path / "metadataset.yaml",
            batch_size=2,
            worker_config=worker_config,
            task_encoder=CookingTaskEncoder(),
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            packing_buffer_size=2,
        ),
    ).with_restored_state_rank(state) as loader:
        samples_restored = [s.__key__ for idx, s in zip(range(100, 200), loader)]
        print(samples_restored)

        assert all([a == b for a, b in zip(samples_after, samples_restored)])


def test_aux_random_access(dataset_path):
    torch.manual_seed(42)
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=2,
    )

    print("Initializing dataset")

    with get_savable_loader(
        get_train_dataset(
            dataset_path / "aux_metadataset.yaml",
            batch_size=2,
            worker_config=worker_config,
            task_encoder=CookingTaskEncoder(),
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            packing_buffer_size=2,
        ),
    ) as loader:
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

    with get_savable_loader(
        get_train_dataset(
            dataset_path / "aux_metadataset.yaml",
            batch_size=2,
            worker_config=worker_config,
            task_encoder=CookingTaskEncoder(),
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            packing_buffer_size=2,
        ),
    ).with_restored_state_rank(state) as loader:
        samples_restored = [s.__key__ for idx, s in zip(range(100, 200), loader)]
        print(samples_restored)

        assert all([a == b for a, b in zip(samples_after, samples_restored)])


def test_aux_random_access_with_cache(dataset_path):
    torch.manual_seed(42)
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=2,
    )

    print("Initializing dataset")

    with get_savable_loader(
        get_train_dataset(
            dataset_path / "aux_metadataset.yaml",
            batch_size=2,
            worker_config=worker_config,
            task_encoder=LazyCookingTaskEncoder(),
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            packing_buffer_size=2,
        ),
        cache_pool=FileStoreCachePool(
            parent_cache_dir=dataset_path / "cache",
            num_workers=1,
        ),
    ) as loader:
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

    with get_savable_loader(
        get_train_dataset(
            dataset_path / "aux_metadataset.yaml",
            batch_size=2,
            worker_config=worker_config,
            task_encoder=CookingTaskEncoder(),
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            packing_buffer_size=2,
        ),
        cache_pool=FileStoreCachePool(
            parent_cache_dir=dataset_path / "cache",
            num_workers=1,
        ),
    ).with_restored_state_rank(state) as loader:
        samples_restored = [s.__key__ for idx, s in zip(range(100, 200), loader)]
        print(samples_restored)

        assert all([a == b for a, b in zip(samples_after, samples_restored)])


def test_aux_random_access_with_cache_and_postencode(dataset_path):
    torch.manual_seed(42)
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=2,
    )

    print("Initializing dataset")

    with get_savable_loader(
        get_train_dataset(
            dataset_path / "aux_metadataset.yaml",
            batch_size=2,
            worker_config=worker_config,
            task_encoder=LazyCookingTaskEncoderWithPostencode(),
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            packing_buffer_size=2,
        ),
        cache_pool=FileStoreCachePool(
            parent_cache_dir=dataset_path / "cache",
            num_workers=1,
        ),
    ) as loader:
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

    with get_savable_loader(
        get_train_dataset(
            dataset_path / "aux_metadataset.yaml",
            batch_size=2,
            worker_config=worker_config,
            task_encoder=LazyCookingTaskEncoderWithPostencode(),
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            packing_buffer_size=2,
        ),
        cache_pool=FileStoreCachePool(
            parent_cache_dir=dataset_path / "cache",
            num_workers=1,
        ),
    ).with_restored_state_rank(state) as loader:
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
                dataset_path=EPath(dataset_path / "ds1"),
                index=2,
                shard_name="parts/data-0.tar",
                file_names=("000002.pkl", "000002.txt"),
            ),
            # Auxiliary source for the sample, reading from ds2
            SourceInfo(
                dataset_path=EPath(dataset_path / "ds2"),
                index="000102.txt",
                shard_name="parts/data-0.tar",
                file_names=("000102.txt",),
            ),
            # Auxiliary source for the sample, reading from ds1, but next sample
            SourceInfo(
                dataset_path=EPath(dataset_path / "ds1"),
                index="000003.txt",
                shard_name="parts/data-0.tar",
                file_names=("000003.txt",),
            ),
            SourceInfo(
                dataset_path=EPath(dataset_path / "ds1"),
                index=21,
                shard_name="parts/data-2.tar",
                file_names=("000021.pkl", "000021.txt"),
            ),
            SourceInfo(
                dataset_path=EPath(dataset_path / "ds2"),
                index="000121.txt",
                shard_name="parts/data-2.tar",
                file_names=("000121.txt",),
            ),
            SourceInfo(
                dataset_path=EPath(dataset_path / "ds1"),
                index="000022.txt",
                shard_name="parts/data-2.tar",
                file_names=("000022.txt",),
            ),
        )


def test_aux_filesystem_reference(dataset_path):
    torch.manual_seed(42)
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=0,
    )

    with get_savable_loader(
        get_train_dataset(
            dataset_path / "aux_metadataset.yaml",
            batch_size=1,
            worker_config=worker_config,
            task_encoder=CookingTaskEncoderWithAuxFilesystemReference(),
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
        ),
    ) as loader:
        sample = next(iter(loader))

        assert sample.txts[0].endswith("|aux|__module__: megatron.ener>")


def test_media_metadata_webdataset(dataset_path):
    torch.manual_seed(42)
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=0,
    )

    loader = get_savable_loader(
        get_train_dataset(
            dataset_path / "media_metadataset.yaml",
            batch_size=1,
            worker_config=worker_config,
            task_encoder=CookingTaskEncoder(),
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
        )
    )

    descriptions = []
    for _, batch in zip(range(4), loader):
        descriptions.extend(batch.txts)

    # from pprint import pprint
    # pprint(descriptions, indent=4)

    # The descriptions are like "A|B", where A is the format
    # in the WebDataset and B is the format in the auxiliary dataset.

    assert descriptions == [
        "IMG-32x16-JPEG|IMG-32x16-JPEG",
        "IMG-24x24-PNG|IMG-24x24-PNG",
        "AUDIO-10.0s@32000Hz|AUDIO-10.0s@32000Hz",
        "VIDEO-192x108@30.0fps-63.0s|VIDEO-192x108@30.0fps-63.0s",
    ]


def test_nomds(dataset_path):
    torch.manual_seed(42)
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=2,
    )

    with get_savable_loader(
        get_train_dataset(
            dataset_path / "ds1",
            batch_size=2,
            worker_config=worker_config,
            task_encoder=GenericCookingTaskEncoder(),
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
        ),
    ) as loader:
        samples = [s.__key__ for idx, s in zip(range(100), loader)]

        print(samples)
        assert len(samples) == 100
