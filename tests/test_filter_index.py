# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shard-preserving filter index translation."""

import gc
import json
import struct
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import webdataset as wds
import yaml

from megatron.energon import (
    FilterIndexWriter,
    WorkerConfig,
    build_filter_index,
    build_filter_index_from_shard_indexes,
    get_savable_loader,
    get_train_dataset,
    get_val_dataset,
    load_dataset,
)
from megatron.energon.dataset_config import get_dataset_from_config
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory
from megatron.energon.flavors.common.filter_index import filter_index_paths
from megatron.energon.flavors.common.manifest.paths import MAIN_FOLDER_NAME
from megatron.energon.flavors.common.manifest.types import ShardInfo
from megatron.energon.flavors.dataset_factory_resolver import register_dataset_factory_provider
from megatron.energon.flavors.dataset_type import EnergonDatasetType
from megatron.energon.flavors.jsonl.crude_jsonl_dataset import (
    CrudeJsonlDatasetFactory,
    DefaultCrudeJsonlDatasetFactory,
)


def _write_binidx(path: Path, *, num_docs: int, doc_len: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.arange(num_docs * doc_len, dtype=np.int32)
    values.tofile(str(path))
    with open(path.with_suffix(".idx"), "wb") as f:
        f.write(b"MMIDIDX\x00\x00")
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", 4))
        f.write(struct.pack("<Q", num_docs))
        f.write(struct.pack("<Q", num_docs))
        f.write(np.full(num_docs, doc_len, dtype=np.int32).tobytes())
        f.write((np.arange(num_docs, dtype=np.int64) * doc_len * 4).tobytes())
        f.write(np.arange(num_docs, dtype=np.int64).tobytes())


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

    def _load_samples(self, path, split="train"):
        factory = get_dataset_from_config(
            path,
            split_part=split,
            training=False,
            worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
            filter_name="keep",
        )

        with get_savable_loader(factory.build()) as loader:
            samples = list(loader)
            assert len(samples) == len(factory) == len(loader)
            return samples

    def test_filter_builders_load_the_same_samples(self):
        path, _ = self._webdataset({"a": 5, "b": 4})
        expected = ["a-1", "a-3", "b-0", "b-2"]
        for builder in ("global", "shard", "writer"):
            with self.subTest(builder=builder):
                if builder == "global":
                    build_filter_index(path, "keep", [1, 3, 5, 7], overwrite=True)
                elif builder == "shard":
                    build_filter_index_from_shard_indexes(
                        path, "keep", {"a": [1, 3], "b": [0, 2]}, overwrite=True
                    )
                else:
                    with FilterIndexWriter(path, "keep", overwrite=True) as writer:
                        for index in (1, 3, 5, 7):
                            writer.append(index)
                samples = self._load_samples(path)
                assert [sample.__key__ for sample in samples] == expected
                assert [sample.text for sample in samples] == expected

    def test_filtered_splits(self):
        path, _ = self._webdataset({"z": 3, "a": 5, "m": 3})
        # The first and last shards have equal sizes: their identity and order matter.
        splits = {
            "train": ["z", "a", "m"],
            "reordered": ["m", "a", "z"],
            "val": ["a"],
            "test": ["m", "z"],
        }
        with (path / MAIN_FOLDER_NAME / "split.yaml").open("w") as f:
            yaml.safe_dump({"split_parts": splits}, f)

        # Also cover zero retained samples in individual shards and in the entire filter.
        for kept in ({"z": [1], "a": [0, 4], "m": [2]}, {"a": [4]}, {}):
            build_filter_index_from_shard_indexes(path, "keep", kept, overwrite=True)
            for split, names in splits.items():
                with self.subTest(kept=kept, split=split):
                    expected = [f"{name}-{i}" for name in names for i in kept.get(name, [])]
                    dataset = get_val_dataset(
                        path,
                        split_part=split,
                        worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
                        batch_size=None,
                        filter_name="keep",
                    )
                    with get_savable_loader(dataset) as loader:
                        assert len(loader) == len(expected)
                        samples = list(loader)
                        assert [sample.__key__ for sample in samples] == expected
                        assert [sample.text for sample in samples] == expected
                        for sample in samples:
                            restored = loader.restore_sample(sample.__restore_key__)
                            assert (restored.__key__, restored.text) == (
                                sample.__key__,
                                sample.text,
                            )

    def test_filter_source_order_and_subset(self):
        path, shards = self._webdataset({"z": 3, "a": 5, "m": 2})
        with (path / MAIN_FOLDER_NAME / "split.yaml").open("w") as f:
            yaml.safe_dump({"split_parts": {"train": ["a"], "val": ["z", "a", "m"]}}, f)
        for source in (list(reversed(shards)), shards[1:2]):
            with self.subTest(source=[shard.name for shard in source]):
                build_filter_index_from_shard_indexes(
                    path, "keep", {"a": [4]}, shards=source, overwrite=True
                )
                assert [sample.text for sample in self._load_samples(path)] == ["a-4"]
                if len(source) == 1:
                    # A train-only filter cannot be used for uncovered validation shards.
                    with self.assertRaisesRegex(
                        ValueError, "missing information about the shard 'z'"
                    ):
                        self._load_samples(path, "val")
                else:
                    assert [sample.text for sample in self._load_samples(path, "val")] == ["a-4"]

    def test_filter_distribution_across_ranks_and_workers(self):
        path, _ = self._webdataset({"z": 8, "a": 5, "m": 8})
        kept = {"z": [1, 3, 5, 7], "m": [0, 2, 4, 6]}
        build_filter_index_from_shard_indexes(path, "keep", kept)
        expected = [f"{name}-{i}" for name, indexes in kept.items() for i in indexes]
        for order in (["z", "a", "m"], ["m", "a", "z"]):
            with (path / MAIN_FOLDER_NAME / "split.yaml").open("w") as f:
                yaml.safe_dump({"split_parts": {"train": order}}, f)
            with self.subTest(order=order):
                rank_keys = []
                for rank in range(2):
                    dataset = get_val_dataset(
                        path,
                        split_part="train",
                        worker_config=WorkerConfig(rank=rank, world_size=2, num_workers=2),
                        batch_size=2,
                        filter_name="keep",
                    )
                    with get_savable_loader(dataset) as loader:
                        batches = list(loader)
                        assert len(batches) == len(loader)
                        keys = [key for batch in batches for key in batch.__key__]
                        assert [text for batch in batches for text in batch.text] == keys
                        assert len(keys) == len(set(keys)) == 4
                        rank_keys.append(keys)
                assert set(rank_keys[0]).isdisjoint(rank_keys[1])
                self.assertCountEqual(rank_keys[0] + rank_keys[1], expected)

    def test_filtered_training_recipe_checkpoint(self):
        path, _ = self._webdataset({"z": 8, "a": 5, "m": 8})
        kept = {"z": [1, 3, 5, 7], "m": [0, 2, 4, 6]}
        build_filter_index_from_shard_indexes(path, "keep", kept)
        expected = [f"{name}-{i}" for name, indexes in kept.items() for i in indexes]
        recipe_path = self.dataset_path / "recipe.yaml"
        recipe_path.write_text(
            yaml.safe_dump(
                {
                    "__module__": "megatron.energon",
                    "__class__": "Recipe",
                    "splits": {"train": {"path": str(path), "filter": "keep"}},
                }
            )
        )

        for order in (["z", "a", "m"], ["m", "a", "z"]):
            with (path / MAIN_FOLDER_NAME / "split.yaml").open("w") as f:
                yaml.safe_dump({"split_parts": {"train": order}}, f)
            for num_workers in (0, 2):
                with self.subTest(order=order, num_workers=num_workers):

                    def new_loader():
                        return get_savable_loader(
                            get_train_dataset(
                                recipe_path,
                                worker_config=WorkerConfig(
                                    rank=0, world_size=1, num_workers=num_workers, seed_offset=42
                                ),
                                batch_size=2,
                                shuffle_buffer_size=3,
                                max_samples_per_sequence=2,
                                repeat=False,
                            )
                        )

                    with new_loader() as loader:
                        batches = iter(loader)
                        first = next(batches)
                        state = loader.save_state_rank()
                        remaining = [(batch.__key__, batch.text) for batch in batches]
                        all_batches = [(first.__key__, first.text), *remaining]
                        self.assertCountEqual(
                            [key for keys, _ in all_batches for key in keys], expected
                        )
                        assert all(keys == text for keys, text in all_batches)
                        assert remaining
                    with new_loader() as restored:
                        restored.restore_state_rank(state)
                        assert [(batch.__key__, batch.text) for batch in restored] == remaining

    def test_unsupported_version(self):
        path, _ = self._webdataset({"a": 3})
        build_filter_index(path, "keep", [1])
        assert [sample.text for sample in self._load_samples(path)] == ["a-1"]
        json_path, _ = filter_index_paths(path, "keep")
        for metadata in (
            {"version": 1, "filtered_shard_counts": {"a": 1}},
            {"version": 3, "shards": []},
        ):
            with self.subTest(version=metadata["version"]):
                with json_path.open("w") as f:
                    json.dump(metadata, f)
                with self.assertRaisesRegex(ValueError, "Unsupported.*Rebuild"):
                    self._load_samples(path)
        build_filter_index(path, "keep", [1], overwrite=True)
        assert [sample.text for sample in self._load_samples(path)] == ["a-1"]

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
        @register_dataset_factory_provider
        class RecordsProvider(DefaultCrudeJsonlDatasetFactory):
            @classmethod
            def detect_path(cls, path):
                if path.name.endswith(".filter-test-records") and path.is_file():
                    return EnergonDatasetType.JSONL
                return None

        file = self.dataset_path / "source.filter-test-records"
        file.write_text('{"value": 0}\n{"value": 1}\n')
        path = EPath(file)
        CrudeJsonlDatasetFactory.prepare_dataset(path)
        build_filter_index(path, "keep", [1])
        assert [s["json"]["value"] for s in self._load_samples(path)] == [1]
        with file.open("a") as f:
            f.write('{"value": 2}\n')
        CrudeJsonlDatasetFactory.prepare_dataset(path)
        with self.assertRaisesRegex(ValueError, "Stale filter.*sample count"):
            self._load_samples(path)
        build_filter_index(path, "keep", [0, 2], overwrite=True)
        assert [s["json"]["value"] for s in self._load_samples(path)] == [0, 2]

    def test_binidx_filter_and_recipe_passthrough(self):
        dataset_path = self.dataset_path / "binidx"
        bin_path = dataset_path / "tokens.bin"
        _write_binidx(bin_path, num_docs=6)
        path = EPath(bin_path)
        build_filter_index(path, "keep", [1, 4])
        json_path, idx_path = filter_index_paths(path, "keep")
        assert json_path.name == "tokens.bin.keep.json"
        assert idx_path.name == "tokens.bin.keep.idx"

        samples = self._load_samples(path)
        assert [s["__key__"] for s in samples] == ["1", "4"]
        assert [s["tokens"].tolist() for s in samples] == [[2, 3], [8, 9]]

        recipe_path = dataset_path / "recipe.yaml"
        recipe_path.write_text(
            "__module__: megatron.energon\n"
            "__class__: Recipe\n"
            "splits:\n"
            "  train:\n"
            "    path: tokens.bin\n"
            "    filter: keep\n"
        )
        loaded = load_dataset(EPath(recipe_path)).get_datasets(
            training=False,
            split_part="train",
            worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
        )
        assert len(loaded.datasets[0].dataset) == 2
