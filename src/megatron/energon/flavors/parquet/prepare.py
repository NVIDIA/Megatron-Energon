# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Parquet footer scanning helpers for single files and manifest shard lists."""

from typing import Any, Iterable, List, Mapping, Sequence, Tuple

from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.common.manifest.paths import MAIN_FOLDER_NAME
from megatron.energon.flavors.common.manifest.types import ShardInfo
from megatron.energon.typed_converter import JsonParser


@edataclass
class ParquetFileEntry:
    """One Parquet file under a reader base path."""

    rel_path: str
    num_rows: int


@edataclass
class ParquetLayout:
    """In-memory Parquet file list and schema metadata."""

    version: int
    columns: List[str]
    files: List[ParquetFileEntry]
    total_rows: int


def parse_parquet_layout(raw: Mapping[str, Any]) -> ParquetLayout:
    """Parse layout from structured mapping, same as other energon configs."""

    return JsonParser(strict=True).raw_to_typed(dict(raw), ParquetLayout)


def _read_parquet_footer(path: EPath) -> Tuple[int, Tuple[str, ...]]:
    """Return (num_rows, column names) using footer metadata only."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    with path.open("rb") as raw:
        pf = pq.ParquetFile(pa.PythonFile(raw))
        cols = tuple(pf.schema_arrow.names)
        return pf.metadata.num_rows, cols


def discover_parquet_files(dataset_root: EPath) -> List[EPath]:
    """Sorted Parquet paths under ``dataset_root`` (recursive, excluding ``.nv-meta``)."""

    out: List[EPath] = []
    for p in sorted(dataset_root.glob("**/*.parquet")):
        rel = p.relative_to(dataset_root)
        rel_parts = rel.split("/")
        if rel_parts and rel_parts[0] == MAIN_FOLDER_NAME:
            continue
        out.append(p)
    return out


def _layout_from_entries(entries: Iterable[tuple[str, EPath, int | None]]) -> ParquetLayout:
    rows_and_cols: List[Tuple[int, Tuple[str, ...]]] = []
    rel_entries: List[ParquetFileEntry] = []
    for rel_path, path, expected_rows in entries:
        rows, cols = _read_parquet_footer(path)
        if expected_rows is not None and rows != expected_rows:
            raise ValueError(
                f"Parquet row count mismatch for {path}: metadata={expected_rows}, footer={rows}"
            )
        rel_entries.append(ParquetFileEntry(rel_path=rel_path, num_rows=rows))
        rows_and_cols.append((rows, cols))

    if not rows_and_cols:
        raise ValueError("No parquet files found")

    col_sets = [set(cols) for _, cols in rows_and_cols]
    unified = col_sets[0]
    for idx, cols in enumerate(col_sets[1:], start=1):
        if cols != unified:
            raise ValueError(
                f"Parquet schema mismatch: file 0 columns {sorted(unified)} vs "
                f"file {idx} columns {sorted(cols)}"
            )

    columns_sorted = sorted(unified)
    return ParquetLayout(
        version=1,
        columns=columns_sorted,
        files=rel_entries,
        total_rows=sum(entry.num_rows for entry in rel_entries),
    )


def scan_parquet_file(path: EPath) -> ParquetLayout:
    """Scan one `.parquet` file and return a single-file layout."""

    path = EPath(path)
    assert path.is_file(), f"Parquet path must be a file: {path}"
    return _layout_from_entries([(path.name, path, None)])


def scan_parquet_shards(shards: Sequence[ShardInfo]) -> ParquetLayout:
    """Scan manifest Parquet shards and verify manifest counts against file footers."""

    return _layout_from_entries((shard.name, shard.path, shard.count) for shard in shards)


def scan_parquet_dataset(dataset_root: EPath) -> ParquetLayout:
    """Scan a single parquet file, or recursively scan a directory for compatibility/tooling."""

    dataset_root = EPath(dataset_root)
    if dataset_root.is_file():
        return scan_parquet_file(dataset_root)

    files = discover_parquet_files(dataset_root)
    if not files:
        raise ValueError(f"No .parquet files found under {dataset_root}")
    return _layout_from_entries(
        (str(file_path.relative_to(dataset_root)), file_path, None) for file_path in files
    )


def assert_layout_columns_subset(
    layout_columns: Sequence[str], read_columns: Sequence[str]
) -> None:
    known_columns = set(layout_columns)
    for column in read_columns:
        if column not in known_columns:
            raise ValueError(f"Column {column!r} not in layout columns {sorted(known_columns)}")
