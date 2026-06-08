# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import unittest
from pathlib import Path

import numpy as np

from megatron.energon.epathlib import EPath, EPathNumpyMappedArray, EPathReadMappedArray


class TestEPathMappedArray(unittest.TestCase):
    def test_local_mmap_context_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "values.idx"
            np.asarray([1, 3, 5], dtype=np.uint64).tofile(path)

            with EPath(path).map(dtype=np.uint64) as values:
                assert len(values) == 3
                assert int(values[1]) == 3
                assert isinstance(values, EPathNumpyMappedArray)
                assert values._memmap is not None
            assert values._memmap is None

    def test_file_reader_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "values.idx"
            np.asarray([2, 4, 6, 8], dtype=np.uint64).tofile(path)

            values = EPath(path).map(dtype=np.uint64, use_mmap=False)
            assert int(values[2]) == 6
            assert values[1:3].tolist() == [4, 6]
            assert isinstance(values, EPathReadMappedArray)
            assert values._file is not None
            values.close()
            assert values._file is None
            assert int(values[0]) == 2
            values.close()

    def test_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "values.idx"
            path.write_bytes(b"header!!" + np.asarray([10, 20], dtype=np.int32).tobytes())

            with EPath(path).map(dtype=np.int32, shape=2, offset=8) as values:
                assert values[:].tolist() == [10, 20]


if __name__ == "__main__":
    unittest.main()
