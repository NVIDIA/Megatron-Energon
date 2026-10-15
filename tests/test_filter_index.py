# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shard-preserving filter index translation."""

import gc
import json
import tempfile
import unittest
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import webdataset as wds
import yaml

from megatron.energon import (
    TextSample,
    WorkerConfig,
    build_filter_index,
    build_filter_index_from_shard_indexes,
    get_savable_loader,
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
from megatron.energon.flavors.dataset_factory_resolver import DatasetFactoryResolver
from megatron.energon.flavors.dataset_type import EnergonDatasetType
from megatron.energon.flavors.jsonl.crude_jsonl_dataset import (
    CrudeJsonlDatasetFactory,
    DefaultCrudeJsonlDatasetFactory,
)


@contextmanager
def _active_worker(worker_config: WorkerConfig) -> Iterator[None]:
    worker_config.worker_activate(0)
    try:
        yield
    finally:
        worker_config.worker_deactivate()


def _collect_factory_samples(factory):
    with _active_worker(factory.worker_config):
        dataset = factory.build()
        try:
            return list(dataset)
        finally:
            dataset.close()


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

    def _webdataset(self, counts, *, name="dataset"):
        """Real tar/index files, with an explicitly ordered source manifest."""
        dataset_path = self.dataset_path / name
        dataset_path.mkdir()
        for name, count in counts.items():
            with wds.TarWriter(str(dataset_path / name)) as writer:
                for i in range(count):
                    writer.write({"__key__": f"{name}-{i}", "txt": f"{name}-{i}".encode()})
        BaseWebdatasetFactory.prepare_dataset(
            dataset_path,
            list(counts),
            split_parts_ratio=[("train", 1.0)],
            workers=1,
            shuffle_seed=None,
        )
        meta_path = dataset_path / MAIN_FOLDER_NAME
        (meta_path / ".info.json").write_text(json.dumps({"shard_counts": counts}))
        (meta_path / "dataset.yaml").write_text(
            "sample_type: {__module__: megatron.energon, __class__: TextSample}\n"
            "field_map: {text: txt}\n"
        )
        path = EPath(dataset_path)
        return path, [ShardInfo(name=name, path=path / name, count=n) for name, n in counts.items()]

    def _factory(self, path, split="train"):
        return get_dataset_from_config(
            path,
            split_part=split,
            training=False,
            worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
            filter_name="keep",
        )

    def _load_samples(self, path, split="train"):
        loader = get_savable_loader(self._factory(path, split).build())
        try:
            return list(loader)
        finally:
            loader.close()

    def test_filter_index_round_trip(self) -> None:
        dataset_path, shards = self._webdataset({"a": 5, "b": 4})
        build_filter_index(
            dataset_path,
            "keep",
            (1, 3, 5, 7),
            shards=shards,
        )
        json_path, _index_path = filter_index_paths(dataset_path, "keep")
        with json_path.open("r") as f:
            metadata = json.load(f)
        assert metadata["version"] == 3
        assert [shard["name"] for shard in metadata["shards"]] == ["a", "b"]
        assert [shard["sample_count"] for shard in metadata["shards"]] == [5, 4]
        assert [shard["filtered_sample_count"] for shard in metadata["shards"]] == [2, 2]
        assert [shard["byte_size"] for shard in metadata["shards"]] == [
            shard.path.size() for shard in shards
        ]

        index = FilterIndex(dataset_path, "keep")
        assert index.total_filtered == 4
        assert index[2] == 5

    def test_filter_index_global_writer(self) -> None:
        dataset_path, shards = self._webdataset({"a": 5, "b": 4})
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
        dataset_path, shards = self._webdataset({"a": 5, "b": 4})
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
        dataset_path, shards = self._webdataset({"a": 5, "b": 4})
        build_filter_index(
            dataset_path,
            "keep",
            (1, 3, 5, 7),
            shards=shards,
        )
        with patch.object(EPath, "map", side_effect=AssertionError("must remain lazy")):
            index = FilterIndex(dataset_path, "keep")
        assert len(index) == 4
        assert index.translate_shards(shards) == [
            ShardInfo(name="a", path=dataset_path / "a", count=2),
            ShardInfo(name="b", path=dataset_path / "b", count=2),
        ]
        assert index._index_map is None
        assert index[0] == 1
        index.close()

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

    def test_reordered_splits_and_restore(self):
        path, shards = self._webdataset({"z": 3, "a": 5, "m": 2})
        # Raw order deliberately differs from alphabetical order and both splits.
        build_filter_index(path, "keep", [1, 3, 7, 9])
        (self.dataset_path / "dataset" / MAIN_FOLDER_NAME / "split.yaml").write_text(
            yaml.safe_dump(
                {"split_parts": {"train": ["m", "z", "a"], "val": ["a"], "test": ["m", "z"]}}
            )
        )
        for split, expected in (
            ("train", ["m-1", "z-1", "a-0", "a-4"]),
            ("val", ["a-0", "a-4"]),
            ("test", ["m-1", "z-1"]),
        ):
            with self.subTest(split=split):
                factory = self._factory(path, split)
                assert len(factory) == len(expected)
                with _active_worker(factory.worker_config):
                    dataset = factory.build()
                    try:
                        samples = list(dataset)
                        assert [s.text for s in samples] == expected
                        assert [s.__restore_key__[1] for s in samples] == list(range(len(expected)))
                        for sample in samples:
                            assert (
                                dataset.restore_sample(sample.__restore_key__).text == sample.text
                            )
                    finally:
                        dataset.close()

    def test_matching_layout_skips_shard_search(self):
        path, shards = self._webdataset({"z": 3, "a": 5, "m": 2})
        # Keep no samples from the middle shard. The reader still includes its
        # original samples when determining the last shard's starting offset.
        build_filter_index(path, "keep", [1, 9])
        with FilterIndex(path, "keep") as index:
            view = index.for_shards(shards)
            with patch(
                "megatron.energon.flavors.common.filter_index.bisect_right",
                side_effect=AssertionError("Matching layouts must not search for a shard"),
            ):
                assert [view[i] for i in range(len(view))] == [1, 9]
                for invalid in (-1, len(view)):
                    with self.assertRaises(IndexError):
                        view[invalid]

    def test_equal_counts_do_not_make_reordered_layouts_match(self):
        path, shards = self._webdataset({"z": 3, "a": 3})
        build_filter_index(path, "keep", [0, 5])
        with FilterIndex(path, "keep") as index:
            # Matching counts alone do not imply matching shard identities/order.
            view = index.for_shards(list(reversed(shards)))
            assert [view[i] for i in range(len(view))] == [2, 3]

    def test_explicit_filter_order_and_zero_shards(self):
        path, shards = self._webdataset({"z": 3, "a": 5, "m": 2})
        # No samples from first or last source shard, and a different filter layout.
        build_filter_index(path, "keep", [6], shards=list(reversed(shards)))
        with FilterIndex(path, "keep") as index:
            assert [s.name for s in index.shards] == ["m", "a", "z"]
            assert index.filtered_shard_counts == {"m": 0, "a": 1, "z": 0}
            view = index.for_shards(shards)
            assert len(view) == 1
            assert view[0] == 7  # a[4] in reader order z,a,m
        assert [s.text for s in _collect_factory_samples(self._factory(path))] == ["a-4"]

    def test_empty_filter(self):
        path, shards = self._webdataset({"a": 3, "b": 2})
        build_filter_index(path, "keep", [])
        factory = self._factory(path)
        assert len(factory) == 0
        assert _collect_factory_samples(factory) == []
        with FilterIndex(path, "keep") as index:
            with self.assertRaises(IndexError):
                index.for_shards(shards)[0]

    def test_old_version(self):
        path, _ = self._webdataset({"a": 3})
        build_filter_index(path, "keep", [1])
        assert [sample.text for sample in self._load_samples(path)] == ["a-1"]
        json_path, _ = filter_index_paths(path, "keep")
        for metadata in (
            {"version": 1, "filtered_shard_counts": {"a": 1}},
            {
                "version": 2,
                "shards": [{"name": "a", "sample_count": 3, "filtered_sample_count": 1}],
            },
        ):
            with self.subTest(version=metadata["version"]):
                with json_path.open("w") as f:
                    json.dump(metadata, f)
                with self.assertRaisesRegex(ValueError, "Unsupported.*Rebuild"):
                    self._load_samples(path)
        build_filter_index(path, "keep", [1], overwrite=True)
        assert [sample.text for sample in self._load_samples(path)] == ["a-1"]

    def test_split_shard_missing_from_filter(self):
        path, shards = self._webdataset({"a": 3, "b": 2})
        build_filter_index(path, "keep", [1], shards=shards[:1])
        with self.assertRaisesRegex(ValueError, "not covered by filter"):
            self._factory(path)

    def test_reordered_split_checkpoint(self):
        path, _ = self._webdataset({"z": 3, "a": 5, "m": 2})
        build_filter_index(path, "keep", [1, 3, 7, 9])
        (self.dataset_path / "dataset" / MAIN_FOLDER_NAME / "split.yaml").write_text(
            yaml.safe_dump({"split_parts": {"train": ["m", "z", "a"]}})
        )
        loader = get_savable_loader(self._factory(path).build())
        try:
            iterator = iter(loader)
            assert next(iterator).text == "m-1"
            state = loader.save_state_rank()
            remaining = [s.text for s in iterator]
            assert remaining == ["z-1", "a-0", "a-4"]
        finally:
            loader.close()
        restored = get_savable_loader(self._factory(path).build())
        try:
            restored.restore_state_rank(state)
            assert [s.text for s in restored] == remaining
        finally:
            restored.close()

    def test_jsonl_provider(self):
        path = EPath(self.dataset_path / "data.jsonl")
        with path.open("w") as f:
            for i in range(5):
                f.write(json.dumps({"value": i}) + "\n")
        CrudeJsonlDatasetFactory.prepare_dataset(path)
        build_filter_index(path, "keep", [1, 4])
        factory = self._factory(path)
        assert len(factory) == 2
        with patch(
            "megatron.energon.flavors.common.filter_index.bisect_right",
            side_effect=AssertionError("A matching single-file layout must not search"),
        ):
            assert [s["json"]["value"] for s in _collect_factory_samples(factory)] == [1, 4]

    def test_filter_creation_before_dataset_configuration(self):
        for suffix in ("json", "yaml"):
            with self.subTest(format=suffix):
                path, shards = self._webdataset({"z": 3, "a": 2}, name=suffix)
                root = self.dataset_path / suffix
                meta = root / MAIN_FOLDER_NAME
                info = {"shard_counts": {"z": 3, "a": 2}}
                info_path = meta / f".info.{suffix}"
                if suffix == "yaml":
                    (meta / ".info.json").unlink()
                    info_path.write_text(yaml.safe_dump(info, sort_keys=False))
                config_path = meta / "dataset.yaml"
                config = config_path.read_text()
                config_path.unlink()
                if suffix == "yaml":
                    config_path.write_text(
                        "__module__: does.not.exist\n__class__: UnknownFactory\n"
                    )
                # Creating a filter needs only the source info and primary shards,
                # even before the loader configuration and tar indexes are ready.
                companions = {
                    root / f"{shard.name}.idx": (root / f"{shard.name}.idx").read_bytes()
                    for shard in shards
                }
                for file in companions:
                    file.unlink()
                build_filter_index(path, "keep", [1, 4])
                config_path.write_text(config)
                for file, contents in companions.items():
                    file.write_bytes(contents)
                assert [sample.text for sample in self._load_samples(path)] == ["z-1", "a-1"]

                # Reordering the info manifest doesn't change the filter's recorded layout.
                info["shard_counts"] = {"a": 2, "z": 3}
                info_path.write_text(
                    json.dumps(info) if suffix == "json" else yaml.safe_dump(info, sort_keys=False)
                )
                assert [sample.text for sample in self._load_samples(path)] == ["z-1", "a-1"]

    def test_webdataset_rejects_stale_filter_and_rebuilds(self):
        path, _ = self._webdataset({"a": 3, "b": 2})
        root = self.dataset_path / "dataset"
        meta = root / MAIN_FOLDER_NAME
        (meta / "split.yaml").write_text(yaml.safe_dump({"split_parts": {"train": ["a"]}}))
        build_filter_index(path, "keep", [1])
        assert [sample.text for sample in self._load_samples(path)] == ["a-1"]

        # A changed shard must be detected even if it contributes no kept samples
        # and is outside the selected split.
        file = root / "b"
        original = file.read_bytes()
        file.write_bytes(original + b"x")
        with self.assertRaisesRegex(ValueError, "Stale filter.*byte size"):
            self._load_samples(path)
        file.unlink()
        with self.assertRaisesRegex(ValueError, "Stale filter.*missing"):
            self._load_samples(path)
        file.write_bytes(original)

        info_path = meta / ".info.json"
        for counts in ({"a": 4, "b": 2}, {"a": 3}):
            with self.subTest(counts=counts):
                info_path.write_text(json.dumps({"shard_counts": counts}))
                with self.assertRaisesRegex(ValueError, "Stale filter.*sample count"):
                    self._load_samples(path)
        info_path.write_text(json.dumps({"shard_counts": {"a": 3, "b": 2}}))
        file.write_bytes(original + b"x")
        build_filter_index(path, "keep", [1], overwrite=True)
        assert [sample.text for sample in self._load_samples(path)] == ["a-1"]

    def test_jsonl_rejects_stale_filter_and_rebuilds(self):
        file = self.dataset_path / "source.jsonl"
        original = b'{"value": 0}        \n{"value": 1}        \n'
        file.write_bytes(original)
        path = EPath(file)
        CrudeJsonlDatasetFactory.prepare_dataset(path)
        build_filter_index(path, "keep", [1])
        assert [s["json"]["value"] for s in self._load_samples(path)] == [1]

        file.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "Stale filter.*byte size"):
            self._load_samples(path)
        # Replace the data with more samples at the same byte size, then prepare it.
        replacement = b'{"value": 0}\n{"value": 1}\n{"value": 2}\n'
        assert len(replacement) <= len(original)
        file.write_bytes(replacement.ljust(len(original), b" "))
        CrudeJsonlDatasetFactory.prepare_dataset(path)
        with self.assertRaisesRegex(ValueError, "Stale filter.*sample count"):
            self._load_samples(path)
        build_filter_index(path, "keep", [0, 2], overwrite=True)
        assert [s["json"]["value"] for s in self._load_samples(path)] == [0, 2]

        # A truncated offset index must fail both filter creation and dataset loading.
        index_path = self.dataset_path / "source.jsonl.idx"
        offsets = index_path.read_bytes()
        for truncated in (b"", offsets[:-1]):
            with self.subTest(index_size=len(truncated)):
                index_path.write_bytes(truncated)
                with self.assertRaisesRegex(ValueError, "Invalid JSONL offset index size"):
                    build_filter_index(path, "new", [0])
                with self.assertRaisesRegex(ValueError, "Invalid JSONL offset index size"):
                    self._load_samples(path)
        CrudeJsonlDatasetFactory.prepare_dataset(path)
        assert [s["json"]["value"] for s in self._load_samples(path)] == [0, 2]

    def test_custom_direct_file_provider_loads_filtered_samples(self):
        # Register another extension using the JSONL reader to exercise the full
        # provider -> source description -> factory -> loader flow.
        class RecordsProvider(DefaultCrudeJsonlDatasetFactory):
            @classmethod
            def detect_path(cls, path):
                if path.name.endswith(".records") and path.is_file():
                    return EnergonDatasetType.JSONL
                return None

        file = self.dataset_path / "source.records"
        file.write_text('{"value": 0}\n{"value": 1}\n')
        path = EPath(file)
        CrudeJsonlDatasetFactory.prepare_dataset(path)
        resolver = DatasetFactoryResolver()
        resolver.register(RecordsProvider)
        with patch(
            "megatron.energon.flavors.dataset_factory_resolver._DATASET_FACTORY_RESOLVER", resolver
        ):
            build_filter_index(path, "keep", [1])
            assert [s["json"]["value"] for s in self._load_samples(path)] == [1]
            with file.open("a") as f:
                f.write('{"value": 2}\n')
            CrudeJsonlDatasetFactory.prepare_dataset(path)
            with self.assertRaisesRegex(ValueError, "Stale filter.*sample count"):
                self._load_samples(path)
            build_filter_index(path, "keep", [0, 2], overwrite=True)
            assert [s["json"]["value"] for s in self._load_samples(path)] == [0, 2]
