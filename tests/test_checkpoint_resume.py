# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import unittest
from pathlib import Path
from typing import Iterable

import webdataset as wds

from megatron.energon import get_train_dataset
from megatron.energon.checkpoint import (
    ResumeDatasetMetadata,
    ResumeIdentity,
    ResumeMetadata,
    ResumeSubset,
    collect_resume_metadata,
    make_resume_identity,
    migrate_dataloader_state,
)
from megatron.energon.checkpoint.resume import _find_saved_child
from megatron.energon.flavors import BaseWebdatasetFactory
from megatron.energon.flavors.base_dataset import FlexState
from megatron.energon.flavors.common.manifest.paths import MAIN_FOLDER_NAME
from megatron.energon.loader import get_savable_loader
from megatron.energon.worker import WorkerConfig


def _create_text_dataset(path: Path, values: Iterable[int]) -> None:
    (path / "parts").mkdir(exist_ok=True, parents=True)
    with wds.ShardWriter(f"{path}/parts/data-%d.tar", maxcount=10) as shard_writer:
        for value in values:
            shard_writer.write(
                {
                    "__key__": f"{value:06d}",
                    "txt": f"{value}".encode(),
                },
            )
        total_shards = shard_writer.shard

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
                    "__module__: megatron.energon",
                    "__class__: TextWebdataset",
                    "field_map:",
                    "  text: txt",
                ]
            )
        )


def _sampler_counts(state: FlexState) -> list[int]:
    counts: list[int] = []

    def visit(node: object) -> None:
        if isinstance(node, dict):
            if node.get("__class__") == "DatasetSampler":
                counts.append(node["_sample_count"])
            for child in node.get("datasets", []):
                visit(child)

    visit(state)
    return counts


def _batch_values(batch: object) -> set[int]:
    values = getattr(batch, "text")
    if isinstance(values, str):
        return {int(values)}
    return {int(value) for value in values}


def _record_seen(seen: dict[str, set[int]], batch: object) -> None:
    for value in _batch_values(batch):
        if value < 100:
            seen["a"].add(value)
        elif value < 200:
            seen["b"].add(value)
        else:
            seen["c"].add(value)


class TestCheckpointResume(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name)
        _create_text_dataset(self.dataset_path / "a", range(0, 50))
        _create_text_dataset(self.dataset_path / "b", range(100, 150))
        _create_text_dataset(self.dataset_path / "c", range(200, 250))
        self.recipe_path = self.dataset_path / "recipe.yaml"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _loader(self, recipe_path: Path):
        dataset = get_train_dataset(
            recipe_path,
            worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
            batch_size=1,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            repeat=True,
        )
        return get_savable_loader(dataset)

    @staticmethod
    def _write_recipe(path: Path, entries: list[tuple[str, float]]) -> None:
        lines = [
            "__module__: megatron.energon",
            "__class__: Recipe",
            "splits:",
            "  train:",
            "    blend:",
        ]
        for rel_path, weight in entries:
            lines.extend(
                [
                    f"      - weight: {weight}",
                    f"        path: {rel_path}",
                    "        split_part: train",
                ]
            )
        path.write_text("\n".join(lines))

    def test_recipe_blend_add_reorder_preserves_existing_leaf_progress(self):
        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 1.0)])
        old_loader = self._loader(self.recipe_path)

        seen_before_checkpoint: dict[str, set[int]] = {"a": set(), "b": set(), "c": set()}
        old_iter = iter(old_loader)
        while len(seen_before_checkpoint["a"] | seen_before_checkpoint["b"]) < 20 or not (
            seen_before_checkpoint["a"] and seen_before_checkpoint["b"]
        ):
            _record_seen(seen_before_checkpoint, next(old_iter))

        saved_state = old_loader.save_state_rank()
        saved_metadata = collect_resume_metadata(old_loader)
        assert saved_state is not None

        self._write_recipe(self.recipe_path, [("b", 1.0), ("c", 1.0), ("a", 1.0)])
        new_loader = self._loader(self.recipe_path)
        migrated_state = migrate_dataloader_state(new_loader, saved_state, saved_metadata)
        migrated_root = migrated_state.worker_states[0]

        sampler_counts = _sampler_counts(migrated_root)
        assert len(sampler_counts) == 3
        assert sampler_counts[0] > 0
        assert sampler_counts[1] == 0
        assert sampler_counts[2] > 0

        new_loader.restore_state_rank(migrated_state)
        seen_after_resume: dict[str, set[int]] = {"a": set(), "b": set(), "c": set()}
        new_iter = iter(new_loader)
        for _ in range(30):
            _record_seen(seen_after_resume, next(new_iter))

        assert seen_before_checkpoint["a"].isdisjoint(seen_after_resume["a"])
        assert seen_before_checkpoint["b"].isdisjoint(seen_after_resume["b"])

    def test_recipe_blend_remove_preserves_remaining_leaf_progress(self):
        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 1.0), ("c", 1.0)])
        old_loader = self._loader(self.recipe_path)
        old_iter = iter(old_loader)
        seen: dict[str, set[int]] = {"a": set(), "b": set(), "c": set()}
        while not all(seen.values()):
            _record_seen(seen, next(old_iter))

        saved_state = old_loader.save_state_rank()
        saved_metadata = collect_resume_metadata(old_loader)
        assert saved_state is not None

        self._write_recipe(self.recipe_path, [("c", 1.0), ("a", 1.0)])
        new_loader = self._loader(self.recipe_path)
        migrated_state = migrate_dataloader_state(new_loader, saved_state, saved_metadata)

        sampler_counts = _sampler_counts(migrated_state.worker_states[0])
        assert len(sampler_counts) == 2
        assert all(count > 0 for count in sampler_counts)

    def test_recipe_blend_duplicate_identity_stays_fresh(self):
        self._write_recipe(self.recipe_path, [("a", 1.0), ("a", 2.0)])
        old_loader = self._loader(self.recipe_path)
        old_iter = iter(old_loader)
        for _ in range(20):
            next(old_iter)

        saved_state = old_loader.save_state_rank()
        saved_metadata = collect_resume_metadata(old_loader)
        assert saved_state is not None

        self._write_recipe(self.recipe_path, [("a", 2.0), ("c", 1.0), ("a", 1.0)])
        new_loader = self._loader(self.recipe_path)
        migrated_state = migrate_dataloader_state(new_loader, saved_state, saved_metadata)

        # Neither duplicate a leaf can be matched safely after the reorder.
        assert _sampler_counts(migrated_state.worker_states[0]) == [0, 0, 0]

    def test_nested_recipe_split_override_migrates_leaf_progress(self):
        inner_recipe_path = self.dataset_path / "inner.yaml"
        outer_recipe_path = self.dataset_path / "outer.yaml"
        self._write_recipe(inner_recipe_path, [("a", 1.0), ("b", 1.0)])
        outer_recipe_path.write_text(
            "\n".join(
                [
                    "__module__: megatron.energon",
                    "__class__: Recipe",
                    "splits:",
                    "  train:",
                    "    blend:",
                    "      - path: inner.yaml",
                    "        split_part: train",
                ]
            )
        )
        old_loader = self._loader(outer_recipe_path)
        old_iter = iter(old_loader)
        for _ in range(20):
            next(old_iter)

        saved_state = old_loader.save_state_rank()
        saved_metadata = collect_resume_metadata(old_loader)
        assert saved_state is not None

        self._write_recipe(inner_recipe_path, [("b", 1.0), ("c", 1.0), ("a", 1.0)])
        new_loader = self._loader(outer_recipe_path)
        migrated_state = migrate_dataloader_state(new_loader, saved_state, saved_metadata)

        sampler_counts = _sampler_counts(migrated_state.worker_states[0])
        assert len(sampler_counts) == 3
        assert sampler_counts[0] > 0
        assert sampler_counts[1] == 0
        assert sampler_counts[2] > 0

    def test_changed_subset_does_not_reuse_leaf_progress(self):
        def write_subset_recipe(end: str) -> None:
            self.recipe_path.write_text(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Recipe",
                        "splits:",
                        "  train:",
                        "    path: a",
                        f"    subset: {{range: [0%, {end}]}}",
                    ]
                )
            )

        write_subset_recipe("50%")
        old_loader = self._loader(self.recipe_path)
        old_iter = iter(old_loader)
        for _ in range(10):
            next(old_iter)

        saved_state = old_loader.save_state_rank()
        saved_metadata = collect_resume_metadata(old_loader)
        assert saved_state is not None

        write_subset_recipe("75%")
        new_loader = self._loader(self.recipe_path)
        migrated_state = migrate_dataloader_state(new_loader, saved_state, saved_metadata)

        assert _sampler_counts(migrated_state.worker_states[0]) == [0]

    def test_subset_round_trip_is_typed_and_hashable(self):
        identity = ResumeIdentity.from_dict(
            {
                "path": "/dataset",
                "split_part": "train",
                "subset": {
                    "range": [0.25, 0.75],
                    "absolute_range": None,
                },
            }
        )

        assert identity.subset == ResumeSubset(range=(0.25, 0.75))
        assert identity.to_dict()["subset"] == {
            "range": (0.25, 0.75),
            "absolute_range": None,
        }

        restored_identity = ResumeIdentity.from_dict(identity.to_dict())
        assert restored_identity == identity
        assert {identity, restored_identity} == {identity}

    def test_make_identity_normalizes_absolute_subset(self):
        identity = make_resume_identity(
            path="/dataset",
            split_part="train",
            subset=ResumeSubset(
                range=(0.0, 1.0),
                absolute_range=(100, None),
            ),
        )

        assert identity.subset == ResumeSubset(
            range=(0.0, 1.0),
            absolute_range=(100, None),
        )

    def test_ambiguous_exact_identity_does_not_pick_by_position(self):
        identity = make_resume_identity(path="/dataset", split_part="train")
        child = ResumeDatasetMetadata(
            type="DatasetSampler",
            config={},
            identities=(identity,),
        )

        assert _find_saved_child(child, [child, child], set()) is None

    def test_unsupported_resume_metadata_version_is_rejected(self):
        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)
        loader = get_savable_loader(
            get_train_dataset(
                self.dataset_path / "a",
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=True,
            )
        )
        saved_state = loader.save_state_rank()
        saved_metadata = collect_resume_metadata(loader)
        assert saved_state is not None

        with self.assertRaisesRegex(ValueError, "Unsupported resume metadata version 2"):
            migrate_dataloader_state(
                loader,
                saved_state,
                ResumeMetadata(version=2, root=saved_metadata.root),
            )


if __name__ == "__main__":
    unittest.main()
