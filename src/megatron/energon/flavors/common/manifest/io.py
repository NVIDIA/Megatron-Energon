# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import json
from enum import Enum
from typing import List

import braceexpand

from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.eyaml import load_yaml, load_yaml_json
from megatron.energon.flavors.common.manifest.empty_dataset_error import EmptyDatasetError
from megatron.energon.flavors.common.manifest.paths import (
    INDEX_SQLITE_FILENAME,
    INFO_JSON_FILENAME,
    INFO_YAML_FILENAME,
    MAIN_FOLDER_NAME,
)
from megatron.energon.flavors.common.manifest.types import (
    ManifestDatasetInfo,
    ManifestSplits,
    ShardInfo,
)
from megatron.energon.typed_converter import JsonParser


class EnergonDatasetType(Enum):
    METADATASET = "metadataset"
    MANIFEST_DATASET = "manifest_dataset"
    JSONL = "jsonl"
    BINIDX = "binidx"
    PARQUET = "parquet"
    FILESYSTEM = "filesystem"
    INVALID = "invalid"


@edataclass
class ShardListMeta:
    """Shard-list metadata loaded from a manifest dataset."""

    sample_excludes: set[str]
    shards: list[ShardInfo]
    split_part_files: list[str]
    info_shard_files: list[str]

    @staticmethod
    def from_config(
        path: EPath,
        *,
        split_part: str,
        split_config: str | ManifestSplits | None = None,
    ) -> "ShardListMeta":
        if split_config is None:
            split_config = "split.yaml"

        parser = JsonParser(strict=True)
        info = parser.raw_to_typed(
            get_dataset_info(path),
            ManifestDatasetInfo,
        )
        if isinstance(split_config, ManifestSplits):
            splits = split_config
        else:
            try:
                splits = parser.raw_to_typed(
                    load_yaml_json(path / MAIN_FOLDER_NAME / split_config),
                    ManifestSplits,
                )
            except FileNotFoundError:
                if split_config == "split.yaml":
                    splits = parser.raw_to_typed(
                        load_yaml_json(path / MAIN_FOLDER_NAME / "split.json"),
                        ManifestSplits,
                    )
                else:
                    raise
        assert split_part in splits.split_parts, f"Invalid split part: {split_part!r}"
        split_excludes = {
            excluded
            for excluded in splits.exclude
            for excluded in braceexpand.braceexpand(excluded)
        }

        all_split_part_files = [
            name
            for name in splits.split_parts[split_part]
            for name in braceexpand.braceexpand(name)
        ]

        split_part_files = [name for name in all_split_part_files if name not in split_excludes]
        if len(split_part_files) == 0:
            raise EmptyDatasetError(f"No shards found in split part {split_part!r}")
        return ShardListMeta(
            sample_excludes={excluded for excluded in split_excludes if "/" in excluded},
            shards=[
                ShardInfo(
                    name=name,
                    path=path / name,
                    count=info.shard_counts[name],
                )
                for name in split_part_files
            ],
            split_part_files=all_split_part_files,
            info_shard_files=list(info.shard_counts.keys()),
        )


def get_info_shard_files(path: EPath) -> List[str]:
    parser = JsonParser(strict=True)
    info = parser.raw_to_typed(
        get_dataset_info(path),
        ManifestDatasetInfo,
    )
    return list(info.shard_counts.keys())


def get_dataset_info(path: EPath) -> dict:
    info_config = path / MAIN_FOLDER_NAME / INFO_JSON_FILENAME
    yaml_info_config = path / MAIN_FOLDER_NAME / INFO_YAML_FILENAME

    if info_config.is_file():
        with info_config.open("r") as rf:
            return json.load(rf)
    elif yaml_info_config.is_file():
        return load_yaml(yaml_info_config.read_bytes())
    else:
        raise ValueError(f"No info config file found at {info_config} or {yaml_info_config}")


def check_dataset_info_present(path: EPath) -> bool:
    return (path / MAIN_FOLDER_NAME / INFO_JSON_FILENAME).is_file() or (
        path / MAIN_FOLDER_NAME / INFO_YAML_FILENAME
    ).is_file()


def get_dataset_type(path: EPath) -> EnergonDatasetType:
    metadata_db = path / MAIN_FOLDER_NAME / INDEX_SQLITE_FILENAME

    if path.is_file():
        if path.name.endswith(".yaml"):
            return EnergonDatasetType.METADATASET
        elif path.name.endswith(".jsonl"):
            return EnergonDatasetType.JSONL
        elif path.name.endswith(".bin"):
            return EnergonDatasetType.BINIDX
        elif path.name.endswith(".parquet"):
            return EnergonDatasetType.PARQUET
        else:
            return EnergonDatasetType.INVALID
    elif check_dataset_info_present(path):
        return EnergonDatasetType.MANIFEST_DATASET
    elif metadata_db.is_file():
        return EnergonDatasetType.FILESYSTEM
    else:
        return EnergonDatasetType.INVALID
