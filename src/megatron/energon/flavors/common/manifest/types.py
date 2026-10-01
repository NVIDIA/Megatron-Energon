# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import field
from typing import Dict, List, Optional

from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath


@edataclass
class ManifestDatasetInfo:
    """Info for `.nv-meta/.info.yaml` or `.nv-meta/.info.json`."""

    energon_version: Optional[str] = None
    shard_counts: Dict[str, int]


@edataclass
class ManifestSplits:
    """Split config for `.nv-meta/split.yaml` or `.nv-meta/split.json`."""

    split_parts: Dict[str, List[str]]
    exclude: List[str] = field(default_factory=list)


@edataclass
class ShardInfo:
    """A virtual shard in the common indexed sample address space."""

    name: str
    path: EPath
    count: int


@edataclass
class DatasetSubset:
    """Subset range applied first in absolute sample units, then as a relative ratio."""

    range: tuple[float, float] | None = None
    absolute_range: tuple[int, int | None] | None = None

    def compute_subset(self, total_samples: int) -> tuple[int, int]:
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


WebdatasetInfo = ManifestDatasetInfo
WebdatasetSplits = ManifestSplits
