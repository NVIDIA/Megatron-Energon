# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple, TypedDict

from megatron.energon.source_info import SourceInfo


class SampleRecord(TypedDict):
    """Internal indexed sample record returned by format readers."""

    __key__: str
    __shard__: str
    __restore_key__: Tuple[str, int]
    __sources__: tuple[SourceInfo, ...]


FilteredSample = SampleRecord
