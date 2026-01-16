# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import field
from typing import Dict, List, Optional, Tuple, TypedDict

from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.source_info import SourceInfo


@edataclass
class WebdatasetInfo:
    """Info about a webdataset. Format for `.nv-meta/.info.yaml` or `.nv-meta/.info.json`."""

    #: The version of the energon library that was used to prepare the dataset
    energon_version: Optional[str] = None
    #: Maps shard name to number of samples in that shard
    shard_counts: Dict[str, int]


@edataclass
class WebdatasetSplits:
    """Info about the splits of a webdataset. Format for `.nv-meta/split.yaml` or `.nv-meta/split.json`
    (or custom user yaml/json)."""

    #: Maps split part to list of shard names
    split_parts: Dict[str, List[str]]
    #: Set of "<shard name>" or "<shard name>/<sample index>" to exclude
    exclude: List[str] = field(default_factory=list)


@edataclass
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
    #: The source information for the sample.
    __sources__: tuple[SourceInfo, ...]


@edataclass
class DatasetSubset:
    """A subset of a dataset.
    A range is a tuple of two values, where the first value is the start of the subset and the second value is the end of the subset.

    The sharder uses the (absolute/relative) ranges to compute the subsets:
     * `absolute_range` (unit is samples) is applied first on the (e.g. train/val/test) subset
     * then `range` (where `(0, 1)` would correspond to the whole dataset) is applied as relative ratio on the subset that is left.

    This is the struct used internally for computing the range. The config is loaded via the metadataset_v2.
    """

    range: tuple[float, float] | None = None
    absolute_range: tuple[int, int | None] | None = None

    def compute_subset(
        self,
        total_samples: int,
    ) -> tuple[int, int]:
        """
        Computes the absolute subset of samples from the total number of samples.
        The absolute range is applied first, then the relative range is applied on the subset that is left.
        """
        start_samples = 0
        end_samples = total_samples

        if self.absolute_range is not None:
            start_samples, end_samples = self.absolute_range
            if end_samples is None:
                end_samples = total_samples
            assert end_samples <= total_samples, (
                f"Subset samples {self.absolute_range} {end_samples=} > {total_samples=}"
            )
            assert start_samples <= end_samples, (
                f"Subset samples {self.absolute_range} {start_samples=} > {end_samples=}"
            )
            assert start_samples >= 0, f"Subset samples {self.absolute_range} {start_samples=} < 0"
        if self.range is not None:
            previous_total = end_samples - start_samples
            end_samples = start_samples + int(previous_total * self.range[1])
            start_samples += int(previous_total * self.range[0])
            assert end_samples <= total_samples, (
                f"Subset ratio {self.range} {end_samples=} is larger than total samples {total_samples}"
            )
            assert start_samples <= end_samples, (
                f"Subset ratio {self.range} {start_samples=} > {end_samples=}"
            )
            assert start_samples >= 0, f"Subset ratio {self.range} {start_samples=} < 0"
        return start_samples, end_samples

    def config(self) -> dict:
        return {
            "range": self.range,
            "absolute_range": self.absolute_range,
        }
