# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import json
import random
import re
from typing import Any, Optional

import braceexpand
import yaml

from megatron.energon import __version__
from megatron.energon.bracecollapse import collapse
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.config import (
    INFO_JSON_FILENAME,
    INFO_YAML_FILENAME,
    MAIN_FOLDER_NAME,
)
from megatron.energon.flavors.webdataset.structs import ShardInfo, WebdatasetInfo, WebdatasetSplits
from megatron.energon.typed_converter import to_json_object


def build_split_parts(
    shards: list[ShardInfo],
    *,
    split_parts_ratio: Optional[list[tuple[str, float]]] = None,
    split_parts_patterns: Optional[list[tuple[str, str]]] = None,
    shuffle_seed: Optional[int] = None,
) -> dict[str, list[str]]:
    if split_parts_ratio is not None:
        total_ratio = sum(split_ratio for _, split_ratio in split_parts_ratio)
        split_parts_ratio = [
            (split_part, split_ratio / total_ratio) for split_part, split_ratio in split_parts_ratio
        ]

        split_order = list(shards)
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(split_order)

        split_shards = {}
        split_total = 0.0
        split_offset = 0
        for split_part, split_ratio in split_parts_ratio:
            split_total += split_ratio
            split_end = int(len(split_order) * split_total)
            split_shards[split_part] = [shard.name for shard in split_order[split_offset:split_end]]
            split_offset = split_end
    else:
        assert split_parts_patterns is not None, (
            "Require either split_parts_ratio or split_parts_patterns"
        )
        split_shards = {}
        for split_part, split_pattern in split_parts_patterns:
            patterns = [re.compile(pattern) for pattern in braceexpand.braceexpand(split_pattern)]
            split_shards[split_part] = [
                shard.name
                for shard in shards
                if any(pattern.match(shard.name) for pattern in patterns)
            ]

    return {
        split_part: collapse(split_shards[split_part], keep_order=True)
        for split_part in split_shards
    }


def write_manifest_dataset_metadata(
    path: EPath,
    *,
    shards: list[ShardInfo],
    split_config: str,
    split_parts_ratio: Optional[list[tuple[str, float]]] = None,
    split_parts_patterns: Optional[list[tuple[str, str]]] = None,
    shuffle_seed: Optional[int] = None,
    dataset_definition: Optional[dict[str, Any]] = None,
    update_legacy_yaml_info: bool = False,
    fix_local_permissions: bool = False,
    file_perms: Optional[int] = None,
) -> None:
    meta_dir = path / MAIN_FOLDER_NAME
    json_info_config = meta_dir / INFO_JSON_FILENAME
    yaml_info_config = meta_dir / INFO_YAML_FILENAME

    info = WebdatasetInfo(
        energon_version=__version__,
        shard_counts={shard.name: shard.count for shard in shards},
    )
    with json_info_config.open("w") as wf:
        json.dump(to_json_object(info), wf, indent=2)
    if fix_local_permissions and file_perms is not None:
        try:
            json_info_config.local_path().chmod(file_perms)
        except OSError:
            pass

    if update_legacy_yaml_info and yaml_info_config.is_file():
        with yaml_info_config.open("w") as wf:
            yaml.dump(to_json_object(info), wf)

    split_shards = build_split_parts(
        shards,
        split_parts_ratio=split_parts_ratio,
        split_parts_patterns=split_parts_patterns,
        shuffle_seed=shuffle_seed,
    )
    splits_config = WebdatasetSplits(split_parts=split_shards)
    with (meta_dir / split_config).open("w") as wf:
        if split_config.endswith(".yaml"):
            yaml.dump(to_json_object(splits_config), wf, sort_keys=False)
        elif split_config.endswith(".json"):
            json.dump(to_json_object(splits_config), wf, indent=2)
        else:
            raise ValueError(f"Invalid split config extension: {split_config}")
    if fix_local_permissions and file_perms is not None:
        try:
            (meta_dir / split_config).local_path().chmod(file_perms)
        except OSError:
            pass

    if dataset_definition is not None:
        with (meta_dir / "dataset.yaml").open("w") as wf:
            yaml.dump(dataset_definition, wf, sort_keys=False)
