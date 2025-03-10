# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import field
from typing import Dict, List, Optional, Tuple, TypedDict

from megatron.energon.dataclass_slots import dataclass_slots
from megatron.energon.epathlib import EPath


@dataclass_slots
class WebdatasetInfo:
    """Info about a webdataset. Format for `.nv-meta/.info.yaml`."""

    # Maps shard name to number of samples in that shard
    shard_counts: Dict[str, int]


@dataclass_slots
class WebdatasetSplits:
    """Info about the splits of a webdataset. Format for `.nv-meta/split.yaml` (or custom user yaml
    of course)."""

    # Maps split part to list of shard names
    split_parts: Dict[str, List[str]]
    # Set of "<shard name>" or "<shard name>/<sample index>" to exclude
    exclude: List[str] = field(default_factory=list)


@dataclass_slots
class ShardInfo:
    """Info about a single shard as passed through internally. Not exposed to the user."""

    #: Name of the shard file (relative path from the nvinfo dir)
    name: str
    #: The path to the shard file
    path: EPath
    #: The number of samples in this shard
    count: int


class FilteredSample(TypedDict):
    """This is just a definition for the internal loaders. Not exposed to the user."""

    #: The key of the sample within the tar file.
    #: If the tar file contains files 12.jpg and 12.txt,
    #: those two files make one sample with the key "12"
    __key__: str
    #: The base name of the shard file e.g. "shard_000"
    __shard__: str
    #: Globally unique key to restore a sample from disk.
    #: For example `("Webdataset", 123)` would restore the sample at index 123.
    __restore_key__: Tuple[str, int]


def reraise_exception(exc: Exception, key: Optional[str]) -> None:
    if key:
        raise Exception(f"For sample {key!r}") from exc
    else:
        raise
