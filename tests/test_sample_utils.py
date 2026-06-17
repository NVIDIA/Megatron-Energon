# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for sample formatting helpers."""

import unittest
from dataclasses import dataclass

import numpy as np
import torch

from megatron.energon.sample_utils import format_sample_detailed


@dataclass
class _FormatProbeDc:
    n: int
    nested: dict[str, int]


@dataclass
class _NestedFormatProbeDc:
    child: _FormatProbeDc
    children: list[_FormatProbeDc]


class TestFormatSampleDetailed(unittest.TestCase):
    def test_format_sample_detailed_complex_types(self) -> None:
        """Exercise nested containers and special summaries."""

        class _Unknown:
            def __repr__(self) -> str:
                return "<unknown-probe>"

        sample = {
            "scalars": {"i": -3, "f": 2.5, "b": False, "n": None},
            "plain_str": "hi",
            "multiline_str": "line1\nline2",
            "primitive_seq": (1, 2, "x"),
            "hetero_list": [{"k": 1}, {"k": 2}],
            "tensor": torch.tensor([0.0, 2.0], dtype=torch.float32),
            "array": np.array([1, 2, 3], dtype=np.int64),
            "dataclass": _FormatProbeDc(n=9, nested={"a": 1, "b": 2}),
            "nested_dataclass": _NestedFormatProbeDc(
                child=_FormatProbeDc(n=10, nested={"inner": 11}),
                children=[_FormatProbeDc(n=12, nested={"leaf": 13})],
            ),
            "unknown": _Unknown(),
        }
        out = format_sample_detailed(sample)

        self.assertEqual(
            out,
            """\
scalars:
  i: -3
  f: 2.5
  b: false
  n: null
plain_str: hi
multiline_str: 'line1

  line2'
primitive_seq:
- 1
- 2
- x
hetero_list:
- k: 1
- k: 2
tensor: Tensor(shape=torch.Size([2]), dtype=torch.float32, device=cpu, min=0.0, max=2.0, values=[0.0, 2.0])
array: np.ndarray(shape=(3,), dtype=int64, min=1, max=3, values=[np.int64(1), np.int64(2), np.int64(3)])
dataclass:
  _FormatProbeDc:
    n: 9
    nested:
      a: 1
      b: 2
nested_dataclass:
  _NestedFormatProbeDc:
    child:
      _FormatProbeDc:
        n: 10
        nested:
          inner: 11
    children:
    - _FormatProbeDc:
        n: 12
        nested:
          leaf: 13
unknown: <unknown-probe>""",
        )


if __name__ == "__main__":
    unittest.main()
