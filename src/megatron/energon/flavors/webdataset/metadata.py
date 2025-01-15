# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List, Set

import braceexpand
import yaml

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.empty_dataset_error import EmptyDatasetError
from megatron.energon.flavors.webdataset.structs import ShardInfo, WebdatasetInfo, WebdatasetSplits
from megatron.energon.typed_converter import raw_to_typed


@dataclass
class WebdatasetMeta:
    """Class for getting metadata from a webdataset."""

    sample_excludes: Set[str]
    shards: List[ShardInfo]

    @staticmethod
    def from_config(
        path: EPath,
        *,
        split_part: str,
        info_config: str = ".info.yaml",
        split_config: str = "split.yaml",
    ) -> "WebdatasetMeta":
        """
        Loads the metadata for a webdataset, i.e. the shards and sample excludes.

        Args:
            split_part: Which part to load (e.g. 'train', 'val', 'test').
            info_config: Config file to use for sample metadata.
            split_config: Config file to use for shard split definitions.
        """
        info = raw_to_typed(
            yaml.safe_load((path / MAIN_FOLDER_NAME / info_config).read_text()),
            WebdatasetInfo,
        )
        splits = raw_to_typed(
            yaml.safe_load((path / MAIN_FOLDER_NAME / split_config).read_text()),
            WebdatasetSplits,
        )
        assert split_part in splits.split_parts, f"Invalid split part: {split_part!r}"
        split_excludes = {
            excluded
            for excluded in splits.exclude
            for excluded in braceexpand.braceexpand(excluded)
        }
        split_part_files = [
            name
            for name in splits.split_parts[split_part]
            for name in braceexpand.braceexpand(name)
            if name not in split_excludes
        ]
        if len(split_part_files) == 0:
            raise EmptyDatasetError(f"No shards found in split part {split_part!r}")
        return WebdatasetMeta(
            sample_excludes={excluded for excluded in split_excludes if "/" in excluded},
            shards=[
                ShardInfo(
                    name=name,
                    path=path / name,
                    offset=0,
                    count=info.shard_counts[name],
                )
                for name in split_part_files
            ],
        )
