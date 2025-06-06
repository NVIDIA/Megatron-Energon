# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import json
from typing import List, Set

import braceexpand

from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.eyaml import load_yaml, load_yaml_json
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.empty_dataset_error import EmptyDatasetError
from megatron.energon.flavors.webdataset.structs import (
    ShardInfo,
    WebdatasetInfo,
    WebdatasetSplits,
)
from megatron.energon.typed_converter import JsonParser


@edataclass
class WebdatasetMeta:
    """Class for getting metadata from a webdataset."""

    sample_excludes: Set[str]
    shards: List[ShardInfo]
    split_part_files: List[str]
    info_shard_files: List[str]

    @staticmethod
    def from_config(
        path: EPath,
        *,
        split_part: str,
        split_config: str = "split.yaml",
    ) -> "WebdatasetMeta":
        """
        Loads the metadata for a webdataset, i.e. the shards and sample excludes.

        Args:
            split_part: Which part to load (e.g. 'train', 'val', 'test').
            split_config: Config file to use for shard split definitions.
        """
        parser = JsonParser(strict=True)
        info_object = get_dataset_info(path)

        info = parser.raw_to_typed(
            info_object,
            WebdatasetInfo,
        )
        try:
            splits = parser.raw_to_typed(
                load_yaml_json(path / MAIN_FOLDER_NAME / split_config),
                WebdatasetSplits,
            )
        except FileNotFoundError:
            if split_config == "split.yaml":
                # Try split.json instead
                splits = parser.raw_to_typed(
                    load_yaml_json(path / MAIN_FOLDER_NAME / "split.json"),
                    WebdatasetSplits,
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
        return WebdatasetMeta(
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
    """Use this if you don't need the full metadata for split parts, but just the shard files."""
    parser = JsonParser(strict=True)
    info = parser.raw_to_typed(
        get_dataset_info(path),
        WebdatasetInfo,
    )
    return list(info.shard_counts.keys())


def get_dataset_info(path: EPath) -> dict:
    """Given the path to an energon webdataset that contains a .nv-meta folder,
    return the dataset info as a dict.
    """

    info_config = path / MAIN_FOLDER_NAME / ".info.json"
    # YAML for backwards compatibility
    yaml_info_config = path / MAIN_FOLDER_NAME / ".info.yaml"

    if info_config.is_file():
        with info_config.open("r") as rf:
            return json.load(rf)
    elif yaml_info_config.is_file():
        return load_yaml(yaml_info_config.read_bytes())
    else:
        raise ValueError(f"No info config file found at {info_config} or {yaml_info_config}")


def check_dataset_info_present(path: EPath) -> bool:
    """Given the path to an energon webdataset that contains a .nv-meta folder,
    return True if the dataset info is present, False otherwise.
    """
    return (path / MAIN_FOLDER_NAME / ".info.json").is_file() or (
        path / MAIN_FOLDER_NAME / ".info.yaml"
    ).is_file()
