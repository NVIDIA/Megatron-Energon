# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from typing import Any, Protocol, TypeVar, runtime_checkable

from megatron.energon.flavors.common.sample_record import SampleRecord
from megatron.energon.source_info import SourceInfo

T_sample_record = TypeVar("T_sample_record", bound=SampleRecord, covariant=True)
T_part = TypeVar("T_part", covariant=True)


@runtime_checkable
class IndexedSampleReader(Protocol[T_sample_record]):
    """Random-access reader over the common indexed sample address space."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> T_sample_record | None: ...

    def close(self) -> None: ...


@runtime_checkable
class SamplePartReader(IndexedSampleReader[T_sample_record], Protocol[T_sample_record]):
    """Reader that can enumerate samples and their physical parts for indexing/file-store use."""

    def list_all_samples(self) -> Iterable[tuple[str, int, int]]: ...

    def list_all_sample_parts(self) -> Iterable[tuple[str, int, int]]: ...

    def list_sample_parts(self, sample_key: str) -> Iterable[tuple[str, int, int]]: ...

    def get_total_size(self) -> int: ...


@runtime_checkable
class PartFileReader(Protocol[T_part]):
    """String-key random access to a concrete sample part."""

    def __getitem__(self, key: str) -> tuple[T_part, SourceInfo]: ...

    def get_path(self) -> str: ...

    def close(self) -> None: ...


@runtime_checkable
class SamplePartFileReader(
    SamplePartReader[T_sample_record],
    PartFileReader[Any],
    Protocol[T_sample_record],
):
    """Reader that supports both indexed samples and string-key part access."""
