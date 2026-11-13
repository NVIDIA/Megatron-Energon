# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Iterable

import webdataset as wds

from megatron.energon import DefaultTaskEncoder, get_train_dataset
from megatron.energon.checkpoint import (
    ResumeMetadata,
    collect_resume_metadata,
    migrate_dataloader_state,
)
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


def _blend_exhausted(state: FlexState) -> list[bool]:
    """Returns the exhausted flags of the outermost blend in the state."""
    if state.get("__class__") == "BlendDataset":
        return state["exhausted"]
    for child in state.get("datasets", []):
        exhausted = _blend_exhausted(child)
        if exhausted:
            return exhausted
    return []


class GroupingTaskEncoder(DefaultTaskEncoder):
    """Batches odd and even samples separately."""

    def batch_group_criterion(self, sample):
        return int(sample.text) % 2, None


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

    def _loader(
        self,
        recipe_path: Path,
        *,
        world_size: int = 1,
        num_workers: int = 0,
        batch_size: int = 1,
        repeat: bool = True,
        task_encoder: DefaultTaskEncoder | None = None,
    ):
        dataset = get_train_dataset(
            recipe_path,
            worker_config=WorkerConfig(rank=0, world_size=world_size, num_workers=num_workers),
            batch_size=batch_size,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            repeat=repeat,
            **({} if task_encoder is None else {"task_encoder": task_encoder}),
        )
        return get_savable_loader(dataset)

    def _checkpoint(self, num_batches: int, **loader_kwargs):
        """Iterates a loader of the current recipe and returns its state, metadata and samples."""
        loader = self._loader(self.recipe_path, **loader_kwargs)
        loader_iter = iter(loader)
        seen: set[int] = set()
        for _ in range(num_batches):
            seen |= _batch_values(next(loader_iter))
        saved_state = loader.save_state_rank()
        assert saved_state is not None
        return saved_state, collect_resume_metadata(loader), seen

    @staticmethod
    def _write_recipe(
        path: Path,
        entries: list[tuple[str, float]],
        *,
        blend: str = "blend",
        weight_key: str = "weight",
    ) -> None:
        lines = [
            "__module__: megatron.energon",
            "__class__: Recipe",
            "splits:",
            "  train:",
            f"    {blend}:",
        ]
        for rel_path, weight in entries:
            lines.extend(
                [
                    f"      - {weight_key}: {weight}",
                    f"        path: {rel_path}",
                    "        split_part: train",
                ]
            )
        path.write_text("\n".join(lines))

    def test_recipe_blend_add_preserves_existing_leaf_progress(self):
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

        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 1.0), ("c", 1.0)])
        new_loader = self._loader(self.recipe_path)
        migrated_state = migrate_dataloader_state(new_loader, saved_state, saved_metadata)
        migrated_root = migrated_state.worker_states[0]

        sampler_counts = _sampler_counts(migrated_root)
        assert len(sampler_counts) == 3
        assert sampler_counts[0] > 0
        assert sampler_counts[1] > 0
        assert sampler_counts[2] == 0

        new_loader.restore_state_rank(migrated_state)
        seen_after_resume: dict[str, set[int]] = {"a": set(), "b": set(), "c": set()}
        new_iter = iter(new_loader)
        for _ in range(30):
            _record_seen(seen_after_resume, next(new_iter))

        assert seen_before_checkpoint["a"].isdisjoint(seen_after_resume["a"])
        assert seen_before_checkpoint["b"].isdisjoint(seen_after_resume["b"])

    def test_recipe_blend_zero_weight_preserves_remaining_leaf_progress(self):
        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 1.0), ("c", 1.0)])
        old_loader = self._loader(self.recipe_path)
        old_iter = iter(old_loader)
        seen: dict[str, set[int]] = {"a": set(), "b": set(), "c": set()}
        while not all(seen.values()):
            _record_seen(seen, next(old_iter))

        saved_state = old_loader.save_state_rank()
        saved_metadata = collect_resume_metadata(old_loader)
        assert saved_state is not None
        saved_counts = _sampler_counts(saved_state.worker_states[0])

        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 0.0), ("c", 1.0)])
        new_loader = self._loader(self.recipe_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            migrated_state = migrate_dataloader_state(new_loader, saved_state, saved_metadata)

        assert _sampler_counts(migrated_state.worker_states[0]) == [
            saved_counts[0],
            saved_counts[2],
        ]

        new_loader.restore_state_rank(migrated_state)
        seen_after: dict[str, set[int]] = {"a": set(), "b": set(), "c": set()}
        new_iter = iter(new_loader)
        for _ in range(30):
            _record_seen(seen_after, next(new_iter))

        # The dataset with the zero weight is not used anymore, the others continue.
        assert not seen_after["b"]
        assert seen["a"].isdisjoint(seen_after["a"])
        assert seen["c"].isdisjoint(seen_after["c"])

    def test_recipe_blend_duplicate_identity_stays_fresh(self):
        self._write_recipe(self.recipe_path, [("a", 1.0), ("a", 2.0)])
        old_loader = self._loader(self.recipe_path)
        old_iter = iter(old_loader)
        for _ in range(20):
            next(old_iter)

        saved_state = old_loader.save_state_rank()
        saved_metadata = collect_resume_metadata(old_loader)
        assert saved_state is not None

        self._write_recipe(self.recipe_path, [("a", 1.0), ("a", 2.0), ("c", 1.0)])
        new_loader = self._loader(self.recipe_path)
        with self.assertWarnsRegex(UserWarning, "could not reuse the saved progress of 1 dataset"):
            migrated_state = migrate_dataloader_state(new_loader, saved_state, saved_metadata)

        # The two a leaves cannot be told apart, so none of them is matched.
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

        self._write_recipe(inner_recipe_path, [("a", 1.0), ("b", 1.0), ("c", 1.0)])
        new_loader = self._loader(outer_recipe_path)
        migrated_state = migrate_dataloader_state(new_loader, saved_state, saved_metadata)

        sampler_counts = _sampler_counts(migrated_state.worker_states[0])
        assert len(sampler_counts) == 3
        assert sampler_counts[0] > 0
        assert sampler_counts[1] > 0
        assert sampler_counts[2] == 0

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

    def test_grouped_batching_restores_after_recipe_change(self):
        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 1.0)])
        saved_state, saved_metadata, seen_before = self._checkpoint(
            20, batch_size=2, task_encoder=GroupingTaskEncoder()
        )

        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 1.0), ("c", 1.0)])
        new_loader = self._loader(
            self.recipe_path, batch_size=2, task_encoder=GroupingTaskEncoder()
        )
        new_loader.restore_state_rank(
            migrate_dataloader_state(new_loader, saved_state, saved_metadata)
        )
        new_iter = iter(new_loader)
        seen_after: set[int] = set()
        for _ in range(10):
            seen_after |= _batch_values(next(new_iter))

        assert seen_after
        assert seen_before.isdisjoint(seen_after)

    def test_single_dataset_to_blend_preserves_progress(self):
        self._write_recipe(self.recipe_path, [("a", 1.0)])
        saved_state, saved_metadata, seen_before = self._checkpoint(20)

        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 1.0)])
        new_loader = self._loader(self.recipe_path)
        migrated_state = migrate_dataloader_state(new_loader, saved_state, saved_metadata)
        migrated_root = migrated_state.worker_states[0]

        assert _sampler_counts(migrated_root) == [20, 0]
        # The sample index of the loader continues, as it does with worker processes.
        assert migrated_root["_sample_index"] == saved_state.worker_states[0]["_sample_index"]

        new_loader.restore_state_rank(migrated_state)
        new_iter = iter(new_loader)
        seen_after: set[int] = set()
        for _ in range(30):
            seen_after |= _batch_values(next(new_iter))
        assert seen_before.isdisjoint(seen_after)

    def test_blend_to_single_used_dataset_preserves_progress(self):
        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 1.0)])
        saved_state, saved_metadata, _ = self._checkpoint(30)
        saved_counts = _sampler_counts(saved_state.worker_states[0])
        assert saved_counts[0] > 0

        # Only one dataset is left with a non-zero weight, so no blend is built anymore.
        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 0.0)])
        migrated_state = migrate_dataloader_state(
            self._loader(self.recipe_path), saved_state, saved_metadata
        )

        assert _sampler_counts(migrated_state.worker_states[0]) == saved_counts[:1]

    def test_replaced_dataset_does_not_reuse_progress(self):
        # The datasets b and c have the same size and shards, but are different datasets.
        self._write_recipe(self.recipe_path, [("b", 1.0)])
        saved_state, saved_metadata, _ = self._checkpoint(20)

        self._write_recipe(self.recipe_path, [("c", 1.0)])
        migrated_state = migrate_dataloader_state(
            self._loader(self.recipe_path), saved_state, saved_metadata
        )

        assert _sampler_counts(migrated_state.worker_states[0]) == [0]

    def test_recipe_blend_duplicate_in_new_recipe_stays_fresh(self):
        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 1.0)])
        saved_state, saved_metadata, _ = self._checkpoint(30)
        saved_counts = _sampler_counts(saved_state.worker_states[0])

        self._write_recipe(self.recipe_path, [("a", 1.0), ("a", 2.0), ("b", 1.0)])
        new_loader = self._loader(self.recipe_path)
        with self.assertWarnsRegex(UserWarning, "could not reuse the saved progress of 1 dataset"):
            migrated_state = migrate_dataloader_state(new_loader, saved_state, saved_metadata)

        # Neither of the two a leaves may take the progress of the single saved a leaf.
        assert _sampler_counts(migrated_state.worker_states[0]) == [0, 0, saved_counts[1]]

    def test_epochized_blend_does_not_replay_finished_dataset(self):
        _create_text_dataset(self.dataset_path / "s", range(300, 305))
        epochized = dict(blend="blend_epochized", weight_key="repetitions")
        self._write_recipe(self.recipe_path, [("s", 1), ("b", 1)], **epochized)
        old_loader = self._loader(self.recipe_path, repeat=False)

        # Checkpoint within the epoch, once one of the datasets is finished.
        seen_before: list[int] = []
        for batch in old_loader:
            seen_before.extend(_batch_values(batch))
            saved_state = old_loader.save_state_rank()
            assert saved_state is not None
            if any(_blend_exhausted(saved_state.worker_states[0])):
                break
        else:
            self.fail("No dataset finished before the end of the epoch")
        saved_metadata = collect_resume_metadata(old_loader)

        self._write_recipe(self.recipe_path, [("s", 1), ("b", 1), ("c", 1)], **epochized)
        new_loader = self._loader(self.recipe_path, repeat=False)
        new_loader.restore_state_rank(
            migrate_dataloader_state(new_loader, saved_state, saved_metadata)
        )
        seen_after = [value for batch in new_loader for value in _batch_values(batch)]

        # The epoch contains every sample exactly once, including the added dataset.
        assert sorted(seen_before + seen_after) == [
            *range(100, 150),
            *range(200, 250),
            *range(300, 305),
        ]

    def test_zero_weight_keeps_worker_assignment_unlike_deleting(self):
        # With two ranks, the samples of these datasets are not split evenly, so the assignment of
        # their samples to workers depends on the datasets in front of them.
        _create_text_dataset(self.dataset_path / "d", range(300, 355))
        _create_text_dataset(self.dataset_path / "e", range(400, 455))
        self._write_recipe(self.recipe_path, [("d", 1.0), ("e", 1.0)])
        saved_state, saved_metadata, _ = self._checkpoint(30, world_size=2)
        saved_counts = _sampler_counts(saved_state.worker_states[0])
        assert all(count > 0 for count in saved_counts)

        def migrated_counts(entries: list[tuple[str, float]]) -> list[int]:
            self._write_recipe(self.recipe_path, entries)
            migrated_state = migrate_dataloader_state(
                self._loader(self.recipe_path, world_size=2), saved_state, saved_metadata
            )
            return _sampler_counts(migrated_state.worker_states[0])

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Appending a dataset keeps the assignment of the datasets in front of it.
            assert migrated_counts([("d", 1.0), ("e", 1.0), ("c", 1.0)]) == [*saved_counts, 0]
            # A dataset with a zero weight keeps the assignment of the datasets behind it.
            assert migrated_counts([("d", 0.0), ("e", 1.0)]) == saved_counts[1:]

        # Deleting a dataset changes the assignment of the datasets behind it. This is not
        # supported and reported instead of silently restarting.
        with self.assertWarnsRegex(UserWarning, "could not reuse the saved progress of 1 dataset"):
            assert migrated_counts([("e", 1.0)]) == [0]

    def test_migration_with_worker_processes(self):
        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 1.0)])
        old_loader = self._loader(self.recipe_path, num_workers=2)
        try:
            old_iter = iter(old_loader)
            for _ in range(24):
                next(old_iter)
            saved_state = old_loader.save_state_rank()
            saved_metadata = collect_resume_metadata(old_loader)
            assert saved_state is not None
        finally:
            old_loader.close()

        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 1.0), ("c", 1.0)])
        new_loader = self._loader(self.recipe_path, num_workers=2)
        try:
            migrated_state = migrate_dataloader_state(new_loader, saved_state, saved_metadata)
            assert len(migrated_state.worker_states) == 2
            for saved_worker, migrated_worker in zip(
                saved_state.worker_states, migrated_state.worker_states
            ):
                saved_a, saved_b = _sampler_counts(saved_worker.state.dataset_state)
                assert _sampler_counts(migrated_worker.state.dataset_state) == [saved_a, saved_b, 0]
                assert migrated_worker.state.sample_index == saved_worker.state.sample_index
                assert migrated_worker.offset == saved_worker.offset

            new_loader.restore_state_rank(migrated_state)
            new_iter = iter(new_loader)
            seen_after: dict[str, set[int]] = {"a": set(), "b": set(), "c": set()}
            for _ in range(30):
                _record_seen(seen_after, next(new_iter))
            assert all(seen_after.values())
        finally:
            new_loader.close()

    def test_json_metadata_round_trip_keeps_exact_state(self):
        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 1.0)])
        saved_state, saved_metadata, _ = self._checkpoint(20)

        json_metadata = json.loads(json.dumps(saved_metadata.to_dict()))
        migrated_state = migrate_dataloader_state(
            self._loader(self.recipe_path), saved_state, json_metadata
        )

        # The recipe is unchanged, so the whole state is reused, not only the leaf progress.
        assert migrated_state.worker_states == saved_state.worker_states

    def test_metadata_describes_datasets_with_split_part(self):
        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 1.0)])
        metadata = collect_resume_metadata(self._loader(self.recipe_path)).to_dict()

        def datasets(node: dict) -> list[str]:
            return [
                *node["datasets"],
                *(dataset for child in node["children"] for dataset in datasets(child)),
            ]

        assert datasets(metadata["root"]) == [
            f"{self.dataset_path / name} (split_part='train')" for name in ("a", "b")
        ]

    def test_migrating_without_saved_state_returns_none(self):
        self._write_recipe(self.recipe_path, [("a", 1.0), ("b", 1.0)])
        loader = self._loader(self.recipe_path)

        assert migrate_dataloader_state(loader, None, collect_resume_metadata(loader)) is None


if __name__ == "__main__":
    unittest.main()
