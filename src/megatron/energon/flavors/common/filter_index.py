# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Shard-preserving filter index sidecars.

The filter index presents a filtered integer address space while storing the
original global sample indexes that should be read for each filtered sample.
Sidecars live in ``.nv-meta`` as ``filter_<name>.json`` and ``filter_<name>.idx``.
"""

import json
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from typing import BinaryIO

import numpy as np

from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath, EPathMappedArray
from megatron.energon.flavors.common.manifest.paths import INFO_JSON_FILENAME, MAIN_FOLDER_NAME
from megatron.energon.flavors.common.manifest.types import ShardInfo
from megatron.energon.flavors.common.reader import IndexedSampleReader
from megatron.energon.flavors.common.sample_record import SampleRecord
from megatron.energon.typed_converter import JsonParser

FILTER_FILENAME_PREFIX = "filter_"
FILTER_JSON_SUFFIX = ".json"
FILTER_INDEX_SUFFIX = ".idx"
FILTER_INDEX_VERSION = 2
FILTER_INDEX_DTYPE = np.uint64


@edataclass
class FilterShardMetadata:
    name: str
    sample_count: int
    filtered_sample_count: int


@edataclass
class FilterMetadata:
    """Shard order is the exact order used by the binary translation table."""

    version: int
    shards: list[FilterShardMetadata]


def filter_index_paths(dataset_path: EPath, filter_name: str) -> tuple[EPath, EPath]:
    if dataset_path.is_file():
        # Single file dataset
        return (
            dataset_path.parent / f"{dataset_path.name}.{filter_name}{FILTER_JSON_SUFFIX}",
            dataset_path.parent / f"{dataset_path.name}.{filter_name}{FILTER_INDEX_SUFFIX}",
        )
    else:
        # Manifest dataset
        meta_dir = dataset_path / MAIN_FOLDER_NAME
        stem = f"{FILTER_FILENAME_PREFIX}{filter_name}"
        return (
            meta_dir / f"{stem}{FILTER_JSON_SUFFIX}",
            meta_dir / f"{stem}{FILTER_INDEX_SUFFIX}",
        )


class FilterIndex:
    """Metadata and lazy translation map for one named filter."""

    dataset_path: EPath
    filter_name: str
    filtered_shard_counts: dict[str, int]
    total_filtered: int

    _index_path: EPath
    _index_map: EPathMappedArray | None

    def __init__(
        self,
        dataset_path: EPath,
        filter_name: str,
        *,
        copy_index_to_local: bool = False,
    ) -> None:
        # Locate the sidecars and defer mapping the binary index until the first read.
        self.dataset_path = dataset_path
        self.filter_name = filter_name
        json_path, index_path = filter_index_paths(dataset_path, filter_name)
        assert json_path.is_file(), f"Filter metadata not found: {json_path}"
        self._index_path = index_path
        self._index_map = None
        self.copy_index_to_local = copy_index_to_local

        # Check the format version before parsing fields that depend on its schema.
        with json_path.open("r") as f:
            metadata = json.load(f)
        if metadata.get("version") != FILTER_INDEX_VERSION:
            raise ValueError(
                f"Unsupported filter index version {metadata.get('version')}; "
                f"expected {FILTER_INDEX_VERSION}. Rebuild filter {filter_name!r}."
            )

        # Preserve the recorded shard order: it defines the binary index's layout.
        filter_metadata = JsonParser(strict=True).raw_to_typed(metadata, FilterMetadata)
        self.shards = filter_metadata.shards

        # Shard names must be unique so a split can identify each source shard.
        if len({shard.name for shard in self.shards}) != len(self.shards):
            raise ValueError("Duplicate shard names in filter metadata")

        # Record each shard's start in both address spaces: original samples
        # and retained samples. These are sample offsets, not byte offsets.
        self._shard_offsets = {}
        original_start = filtered_start = 0
        for shard in self.shards:
            if not 0 <= shard.filtered_sample_count <= shard.sample_count:
                raise ValueError(f"Invalid filter sample counts for {shard.name!r}")
            self._shard_offsets[shard.name] = (original_start, filtered_start)
            original_start += shard.sample_count
            filtered_start += shard.filtered_sample_count

        # Expose retained counts for worker allocation and the full filter length.
        self.filtered_shard_counts = {
            shard.name: shard.filtered_sample_count for shard in self.shards
        }
        self.total_filtered = filtered_start

    def __len__(self) -> int:
        return self.total_filtered

    def __getitem__(self, filtered_index: int) -> int:
        if not 0 <= filtered_index < self.total_filtered:
            raise IndexError(
                f"Filtered index {filtered_index} out of range [0, {self.total_filtered})"
            )
        return int(self.index_map[filtered_index])

    @property
    def index_map(self) -> EPathMappedArray:
        if self._index_map is None:
            assert self._index_path.is_file(), (
                f"Filter translation index not found: {self._index_path}"
            )
            self._index_map = self._index_path.map(
                dtype=FILTER_INDEX_DTYPE,
                shape=(self.total_filtered,),
                copy_to_local=self.copy_index_to_local,
            )
        return self._index_map

    def close(self) -> None:
        if self._index_map is not None:
            self._index_map.close()
            self._index_map = None

    def __enter__(self) -> "FilterIndex":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def for_shards(self, shards: Sequence[ShardInfo]) -> "FilterIndexView":
        """Translate filtered indexes into a reader's selected, ordered shard layout."""
        return FilterIndexView(self, shards)

    def translate_shards(self, shards: Sequence[ShardInfo]) -> list[ShardInfo]:
        return [
            ShardInfo(
                name=shard.name,
                path=shard.path,
                count=self.filtered_shard_counts.get(shard.name, 0),
            )
            for shard in shards
        ]


class FilterIndexView:
    """Small per-shard offset tables over a shared filter; no per-sample copy."""

    def __init__(self, index: FilterIndex, shards: Sequence[ShardInfo]) -> None:
        self.index = index
        self._ends: list[int] = []
        self._translations: list[tuple[int, int, int, int, int]] = []
        recorded = {shard.name: shard for shard in index.shards}
        filtered_start = reader_start = 0
        for shard in shards:
            source = recorded.get(shard.name)
            if source is None:
                raise ValueError(
                    f"Shard {shard.name!r} is not covered by filter {index.filter_name!r}"
                )
            original_start, sidecar_start = index._shard_offsets[shard.name]
            if source.filtered_sample_count:
                self._translations.append(
                    (filtered_start, sidecar_start, original_start, reader_start, shard.count)
                )
                filtered_start += source.filtered_sample_count
                self._ends.append(filtered_start)
            reader_start += shard.count
        self._length = filtered_start

        # Matching layouts share both address spaces, so reads can use the
        # original translation table directly without finding a shard first.
        self._same_layout = len(shards) == len(index.shards) and all(
            shard.name == source.name and shard.count == source.sample_count
            for shard, source in zip(shards, index.shards)
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, filtered_index: int) -> int:
        if not 0 <= filtered_index < self._length:
            raise IndexError(f"Filtered index {filtered_index} out of range [0, {self._length})")
        if self._same_layout:
            return self.index[filtered_index]
        start, sidecar, original, reader, count = self._translations[
            bisect_right(self._ends, filtered_index)
        ]
        local_index = self.index[sidecar + filtered_index - start] - original
        if not 0 <= local_index < count:
            raise ValueError("Filter entry is outside its source shard; rebuild the filter")
        return reader + local_index

    def close(self) -> None:
        self.index.close()


class TranslatedIndexReader(IndexedSampleReader[SampleRecord]):
    """Reader wrapper that applies a filter translation before delegation."""

    def __init__(
        self,
        inner: IndexedSampleReader[SampleRecord],
        filter_index: FilterIndex | FilterIndexView,
    ) -> None:
        self.inner = inner
        self.filter_index = filter_index

    def __len__(self) -> int:
        return len(self.filter_index)

    def __getitem__(self, filtered_index: int) -> SampleRecord | None:
        original_index = self.filter_index[filtered_index]
        sample = self.inner[original_index]
        if isinstance(sample, dict) and "__restore_key__" in sample:
            sample = dict(sample)
            restore_key = sample["__restore_key__"]
            sample["__restore_key__"] = (restore_key[0], filtered_index)
        return sample

    def close(self) -> None:
        if hasattr(self.inner, "close"):
            self.inner.close()
        self.filter_index.close()


class FilterIndexWriter:
    """Low-level writer for explicit kept sample indexes.

    The primary interface accepts original global sample indexes. Shard-local
    append methods are available for callers that already grouped indexes by
    manifest shard.
    """

    dataset_path: EPath
    filter_name: str
    shard_order: tuple[str, ...]
    original_shard_counts: dict[str, int]
    _original_shard_offsets: dict[str, int]
    _shard_ends: list[int]
    _total_original: int
    filtered_shard_counts: dict[str, int]
    _next_shard_index: int = 0
    _global_shard_index: int = 0
    _active_shard: str | None = None
    _last_original_index: int | None = None
    _last_global_index: int | None = None
    _total_filtered: int = 0
    _json_path: EPath
    _index_path: EPath
    _tmp_json_path: EPath
    _tmp_index_path: EPath
    _index_file: BinaryIO

    def __init__(
        self,
        dataset_path: EPath,
        filter_name: str,
        shards: Sequence[ShardInfo] | None = None,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Args:
            dataset_path: The path to the dataset.
            filter_name: The name of the filter.
            shards: Optional source shard selection/order. If omitted, infer the
                full source layout from the dataset metadata.
            overwrite: Whether to overwrite the existing filter index.
        """
        self.dataset_path = EPath(dataset_path)
        self.filter_name = filter_name
        if shards is None:
            shards = _resolve_shards(self.dataset_path)
        if len({shard.name for shard in shards}) != len(shards):
            raise ValueError("Duplicate shard names in filter source layout")
        self.shard_order = tuple(shard.name for shard in shards)
        self.original_shard_counts = {shard.name: int(shard.count) for shard in shards}
        self._original_shard_offsets = {}
        self._shard_ends = []
        offset = 0
        for shard_name in self.shard_order:
            self._original_shard_offsets[shard_name] = offset
            offset += self.original_shard_counts[shard_name]
            self._shard_ends.append(offset)
        self._total_original = offset
        self.filtered_shard_counts = {}

        json_path, index_path = filter_index_paths(self.dataset_path, filter_name)
        if not overwrite and (json_path.is_file() or index_path.is_file()):
            raise FileExistsError(f"Filter {filter_name!r} already exists in {json_path.parent}")

        self._json_path = json_path
        self._index_path = index_path
        self._tmp_json_path = json_path.with_suffix(f"{FILTER_JSON_SUFFIX}.tmp", replace=False)
        self._tmp_index_path = index_path.with_suffix(f"{FILTER_INDEX_SUFFIX}.tmp", replace=False)
        self._tmp_index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_file = self._tmp_index_path.open("wb")

    def append(self, original_index: int) -> None:
        """Append one kept sample by original global sample index."""

        self.append_global(original_index)

    def append_shard_index(self, shard_name: str, original_index: int) -> None:
        """Append one kept sample by shard-local sample index."""

        self._ensure_active_shard(shard_name)
        assert self._index_file is not None

        self._validate_next_index(shard_name, original_index)
        actual_index = self._original_shard_offsets[shard_name] + original_index
        self._index_file.write(np.asarray([actual_index], dtype=FILTER_INDEX_DTYPE).tobytes())
        self.filtered_shard_counts[shard_name] += 1
        self._total_filtered += 1
        self._last_original_index = original_index

    def append_global(self, original_index: int) -> None:
        """Append one kept sample by original global sample index."""

        shard_name, shard_local_index = self._next_global_to_shard_local(original_index)
        self.append_shard_index(shard_name, shard_local_index)

    def append_globals(self, indexes: Iterable[int]) -> None:
        """Append kept samples by original global sample index."""

        for index in indexes:
            self.append_global(int(index))

    def append_shard(self, shard_name: str, indexes: Iterable[int]) -> None:
        if self._index_file is None:
            raise RuntimeError("FilterIndexWriter is closed")
        self._ensure_active_shard(shard_name)

        index_array = np.fromiter((int(index) for index in indexes), dtype=FILTER_INDEX_DTYPE)
        _validate_index_array(
            shard_name=shard_name,
            indexes=index_array,
            original_count=self.original_shard_counts[shard_name],
            previous_index=self._last_original_index,
        )
        self.filtered_shard_counts[shard_name] += int(index_array.size)
        self._total_filtered += int(index_array.size)
        if index_array.size > 0:
            self._last_original_index = int(index_array[-1])
        actual_indexes = index_array + self._original_shard_offsets[shard_name]
        self._index_file.write(
            actual_indexes.astype(FILTER_INDEX_DTYPE, copy=False).tobytes(order="C")
        )

    def close(self, finalize: bool = True) -> None:
        if self._index_file is None:
            return

        try:
            if finalize:
                self._active_shard = None
                self._last_original_index = None
                while self._next_shard_index < len(self.shard_order):
                    self._ensure_active_shard(self.shard_order[self._next_shard_index])

                metadata = {
                    "version": FILTER_INDEX_VERSION,
                    "shards": [
                        {
                            "name": name,
                            "sample_count": self.original_shard_counts[name],
                            "filtered_sample_count": self.filtered_shard_counts[name],
                        }
                        for name in self.shard_order
                    ],
                }
                with self._tmp_json_path.open("w") as f:
                    json.dump(metadata, f, indent=2, sort_keys=True)
                    f.write("\n")
        finally:
            self._index_file.close()
            self._index_file = None

        if finalize:
            self._tmp_index_path.move(self._index_path)
            self._tmp_json_path.move(self._json_path)
        else:
            if self._tmp_index_path.is_file():
                self._tmp_index_path.unlink()
            if self._tmp_json_path.is_file():
                self._tmp_json_path.unlink()

    def __enter__(self) -> "FilterIndexWriter":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close(finalize=exc_val is None)

    def _ensure_active_shard(self, shard_name: str) -> None:
        if self._active_shard == shard_name:
            return
        if self._next_shard_index >= len(self.shard_order):
            raise ValueError(f"Unexpected extra shard {shard_name!r}")
        expected_shard = self.shard_order[self._next_shard_index]
        if shard_name != expected_shard:
            raise ValueError(
                f"Filter shards must be appended in source layout order: expected "
                f"{expected_shard!r}, got {shard_name!r}"
            )
        self._active_shard = shard_name
        self._last_original_index = None
        self.filtered_shard_counts[shard_name] = 0
        self._next_shard_index += 1

    def _validate_next_index(self, shard_name: str, original_index: int) -> None:
        original_count = self.original_shard_counts[shard_name]
        if original_index < 0 or original_index >= original_count:
            raise ValueError(
                f"Filter index {original_index} for shard {shard_name!r} must be within "
                f"[0, {original_count})"
            )
        if self._last_original_index is not None and original_index <= self._last_original_index:
            raise ValueError(f"Filter indexes for shard {shard_name!r} must be strictly increasing")

    def _next_global_to_shard_local(self, original_index: int) -> tuple[str, int]:
        if self._last_global_index is not None and original_index <= self._last_global_index:
            raise ValueError("Global filter indexes must be strictly increasing")
        if original_index < 0 or original_index >= self._total_original:
            raise ValueError(
                f"Global filter index {original_index} must be within [0, {self._total_original})"
            )
        while original_index >= self._shard_ends[self._global_shard_index]:
            self._global_shard_index += 1
        while self._next_shard_index <= self._global_shard_index:
            self._ensure_active_shard(self.shard_order[self._next_shard_index])
        shard_name = self.shard_order[self._global_shard_index]
        self._last_global_index = original_index
        return shard_name, original_index - self._original_shard_offsets[shard_name]


def build_filter_index(
    dataset_path: EPath,
    filter_name: str,
    indexes: Iterable[int],
    *,
    shards: Sequence[ShardInfo] | None = None,
    overwrite: bool = False,
) -> None:
    """Build a named filter from explicit kept original global sample indexes."""

    with FilterIndexWriter(
        dataset_path,
        filter_name,
        shards,
        overwrite=overwrite,
    ) as writer:
        writer.append_globals(indexes)


def build_filter_index_from_global_indexes(
    dataset_path: EPath,
    filter_name: str,
    indexes: Iterable[int],
    *,
    shards: Sequence[ShardInfo] | None = None,
    overwrite: bool = False,
) -> None:
    """Build a named filter from explicit kept original global sample indexes."""

    build_filter_index(
        dataset_path,
        filter_name,
        indexes,
        shards=shards,
        overwrite=overwrite,
    )


def build_filter_index_from_shard_indexes(
    dataset_path: EPath,
    filter_name: str,
    indexes_by_shard: Mapping[str, Iterable[int]],
    *,
    shards: Sequence[ShardInfo] | None = None,
    overwrite: bool = False,
) -> None:
    """Build a named filter from explicit kept shard-local sample indexes."""

    kept_indexes = {
        shard_name: tuple(int(index) for index in indexes)
        for shard_name, indexes in indexes_by_shard.items()
    }
    with FilterIndexWriter(
        dataset_path,
        filter_name,
        shards,
        overwrite=overwrite,
    ) as writer:
        unknown = kept_indexes.keys() - set(writer.shard_order)
        if unknown:
            raise ValueError(f"Unknown shards in filter selection: {sorted(unknown)}")
        for name in writer.shard_order:
            writer.append_shard(name, kept_indexes.get(name, ()))


def _resolve_shards(dataset_path: EPath) -> list[ShardInfo]:
    if dataset_path.is_file():
        return [
            ShardInfo(
                name=dataset_path.name,
                path=dataset_path,
                count=_count_single_file_samples(dataset_path),
            )
        ]

    info_path = dataset_path / MAIN_FOLDER_NAME / INFO_JSON_FILENAME
    if info_path.is_file():
        with info_path.open("r") as f:
            info = json.load(f)
        return [
            ShardInfo(name=name, path=dataset_path / name, count=int(count))
            for name, count in info["shard_counts"].items()
        ]
    raise FileNotFoundError(
        f"Cannot infer filter shard counts because {info_path} does not exist. "
        "Pass shards explicitly for datasets without .info.json metadata."
    )


def _count_single_file_samples(path: EPath) -> int:
    if path.name.endswith(".jsonl"):
        from megatron.energon.flavors.jsonl.ijsonl import IDX_SUFFIX as JSONL_IDX_SUFFIX

        return int(path.with_suffix(JSONL_IDX_SUFFIX, replace=False).size() // 8 - 1)
    raise ValueError(f"Cannot infer sample count for single-file dataset {path}")


def _validate_index_array(
    *,
    shard_name: str,
    indexes: np.ndarray,
    original_count: int,
    previous_index: int | None = None,
) -> None:
    if original_count < 0:
        raise ValueError(f"Original count for shard {shard_name!r} must be non-negative")
    if indexes.size == 0:
        return
    if previous_index is not None and int(indexes[0]) <= previous_index:
        raise ValueError(f"Filter indexes for shard {shard_name!r} must be strictly increasing")
    if int(indexes[0]) < 0 or int(indexes[-1]) >= original_count:
        raise ValueError(
            f"Filter indexes for shard {shard_name!r} must be within [0, {original_count})"
        )
    if np.any(indexes[1:] <= indexes[:-1]):
        raise ValueError(f"Filter indexes for shard {shard_name!r} must be strictly increasing")
