# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import TypedDict

from megatron.energon.flavors.base_dataset import RestoreKey
from megatron.energon.source_info import SourceInfo


@dataclass(frozen=True, slots=True, kw_only=True)
class IndexedRestoreKey(RestoreKey):
    """The selected row in an indexed dataset's address space."""

    kind: str
    index: int


class SampleRecord(TypedDict):
    """Internal indexed sample record returned by format readers."""

    __key__: str
    __shard__: str
    __restore_key__: IndexedRestoreKey
    __sources__: tuple[SourceInfo, ...]


FilteredSample = SampleRecord
