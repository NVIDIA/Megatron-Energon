# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import unittest
from pathlib import Path
from typing import Iterable

import webdataset as wds

from megatron.energon import get_train_dataset
from megatron.energon.checkpoint import (
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

    def test_recipe_blend_add_reorder_preserves_existing_leaf_progress(self):
        def _loader(recipe_path: Path, worker_config: WorkerConfig):
            dataset = get_train_dataset(
                recipe_path,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                repeat=True,
            )
            return get_savable_loader(dataset)

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

        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)
        _write_recipe(self.recipe_path, [("a", 1.0), ("b", 1.0)])
        old_loader = _loader(self.recipe_path, worker_config)

        seen_before_checkpoint: dict[str, set[int]] = {"a": set(), "b": set(), "c": set()}
        old_iter = iter(old_loader)
        while len(seen_before_checkpoint["a"] | seen_before_checkpoint["b"]) < 20 or not (
            seen_before_checkpoint["a"] and seen_before_checkpoint["b"]
        ):
            _record_seen(seen_before_checkpoint, next(old_iter))

        saved_state = old_loader.save_state_rank()
        saved_metadata = collect_resume_metadata(old_loader)
        assert saved_state is not None

        _write_recipe(self.recipe_path, [("b", 1.0), ("c", 1.0), ("a", 1.0)])
        new_loader = _loader(self.recipe_path, worker_config)
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


if __name__ == "__main__":
    unittest.main()
