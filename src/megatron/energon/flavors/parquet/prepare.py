# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Discover Parquet files under a dataset root and build an in-memory layout (no on-disk manifest)."""

from typing import Any, List, Mapping, Sequence, Tuple

from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from megatron.energon.typed_converter import JsonParser


@edataclass
class ParquetFileEntry:
    """One Parquet file under the dataset root (relative path + row count from footer)."""

    rel_path: str
    num_rows: int


@edataclass
class ParquetLayout:
    """In-memory Parquet file list and schema metadata (discovered at load time)."""

    version: int
    columns: List[str]
    files: List[ParquetFileEntry]
    total_rows: int


def parse_parquet_layout(raw: Mapping[str, Any]) -> ParquetLayout:
    """Parse layout from structured mapping (e.g. JSON/YAML), same as other energon configs."""
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


def scan_parquet_dataset(dataset_root: EPath) -> ParquetLayout:
    """
    Scan Parquet files, verify a single unified column set, return layout (purely in memory).

    Raises:
        ValueError: If no files found or schemas disagree.
    """
    dataset_root = EPath(dataset_root)
    files = discover_parquet_files(dataset_root)
    if not files:
        raise ValueError(f"No .parquet files found under {dataset_root}")

    rows_and_cols: List[Tuple[int, Tuple[str, ...]]] = []
    rel_entries: List[ParquetFileEntry] = []
    for fp in files:
        n, cols = _read_parquet_footer(fp)
        rel = str(fp.relative_to(dataset_root))
        rel_entries.append(ParquetFileEntry(rel_path=rel, num_rows=n))
        rows_and_cols.append((n, cols))

    col_sets = [set(c) for _, c in rows_and_cols]
    unified = col_sets[0]
    for i, cs in enumerate(col_sets[1:], start=1):
        if cs != unified:
            raise ValueError(
                f"Parquet schema mismatch: {files[0]} columns {sorted(unified)} vs "
                f"{files[i]} columns {sorted(cs)}"
            )

    columns_sorted = sorted(unified)
    total = sum(n for n, _ in rows_and_cols)
    sorted_entries = sorted(rel_entries, key=lambda e: e.rel_path)
    return ParquetLayout(
        version=1,
        columns=columns_sorted,
        files=sorted_entries,
        total_rows=total,
    )


def assert_layout_columns_subset(
    layout_columns: Sequence[str], read_columns: Sequence[str]
) -> None:
    ms = set(layout_columns)
    for c in read_columns:
        if c not in ms:
            raise ValueError(f"Column {c!r} not in layout columns {sorted(ms)}")
