# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ENERGON_DEBUG_SHARD_PRINT output of indexed datasets."""

import contextlib
import io
import unittest

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_indexed_dataset import _print_shard_slices
from megatron.energon.flavors.common.manifest.types import ShardInfo
from megatron.energon.worker import WorkerConfig


class TestShardSlicePrint(unittest.TestCase):
    def _print(self, slice_offsets) -> list[str]:
        shards = [
            ShardInfo(name=f"s{i}.tar", path=EPath(f"/data/s{i}.tar"), count=10) for i in range(5)
        ]
        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=len(slice_offsets))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _print_shard_slices(worker_config, shards, slice_offsets)
        return out.getvalue().splitlines()

    def test_lists_are_truncated_independently(self):
        (line,) = self._print([[0, 1, 2, 3, 4, 5, 6]])
        # 7 offsets are truncated, the 6 slices between them are not
        assert "indexes=[0, 1, 2 ...<1> 4, 5, 6]" in line
        assert "slices=[s0.tar[0(start), 1], s0.tar[1, 2], s0.tar[2, 3], s0.tar[3, 4]," in line
        assert "s0.tar[5, 6]]" in line

    def test_one_line_per_worker(self):
        lines = self._print([[0], [0, 10], [10, 25]])
        assert len(lines) == 3
        # A worker without samples has a single offset and no slices
        assert lines[0].startswith("rank=0, worker=0: sample_range=[0, 0] in 0 slices")
        assert lines[0].endswith("indexes=[0] slices=[]")
        assert lines[1].endswith("indexes=[0, 10] slices=[s0.tar[0(start), 10(end)]]")
        assert lines[2].startswith("rank=0, worker=2:")
        assert lines[2].endswith("slices=[s1.tar[0(start),]-s2.tar[,5]]")

    def test_long_lists_are_truncated(self):
        offsets = list(range(0, 50, 2))
        (line,) = self._print([offsets])
        assert "in 24 slices" in line
        assert "indexes=[0, 2, 4 ...<19> 44, 46, 48]" in line
        assert (
            "slices=[s0.tar[0(start), 2], s0.tar[2, 4], s0.tar[4, 6] ...<18> "
            "s4.tar[2, 4], s4.tar[4, 6], s4.tar[6, 8]]"
        ) in line


if __name__ == "__main__":
    unittest.main()
