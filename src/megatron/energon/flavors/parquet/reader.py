# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Random-access and streaming reader over a Parquet-tabular in-memory layout (global row
index)."""

from __future__ import annotations

import contextlib
from bisect import bisect_right
from collections import OrderedDict
from typing import Any, Callable, Generator, List, Sequence, Tuple

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.common.reader import SamplePartFileReader, SequentialSampleReader
from megatron.energon.flavors.common.sample_record import FilteredSample
from megatron.energon.flavors.parquet.prepare import ParquetLayout
from megatron.energon.source_info import SourceInfo


def _import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Parquet flavor requires pyarrow. Install with: pip install megatron-energon[parquet]"
        ) from e
    return pa, pq


def _table_row_to_dict(table: Any, row_idx: int, column_names: Sequence[str]) -> dict[str, Any]:
    """Build a dict for one row (0-based within ``table``)."""
    out: dict[str, Any] = {}
    for name in column_names:
        col = table.column(name)
        val = col[row_idx].as_py()
        out[name] = val
    return out


class IParquetReader(SamplePartFileReader[FilteredSample], SequentialSampleReader[FilteredSample]):
    """
    Parquet access over rows described by a :class:`ParquetLayout`.

    Random access decodes the whole row group of the row. Consecutive rows should be read with
    :meth:`iter_slice`, which decodes them incrementally in batches.

    Args:
        dataset_root: The base path for the relative Parquet file paths of the layout.
        layout: The Parquet files and their row counts.
        read_columns: The columns to read.
        sample_filter: An optional filter function to select samples by their key.
        parquet_file_cache_size: The number of Parquet files to keep open at the same time.
        disable_cache: If True, disables caching of open Parquet files and opens a fresh file
            for every read. This mode avoids sharing Parquet file handles across threads and is
            intended to be thread-safe. Streams from :meth:`iter_slice` always open their own
            file.
        stream_batch_rows: The number of rows :meth:`iter_slice` decodes at a time.
    """

    dataset_root: EPath
    layout: ParquetLayout
    read_columns: list[str]
    sample_filter: Callable[[str], bool] | None
    disable_cache: bool

    def __init__(
        self,
        dataset_root: EPath,
        layout: ParquetLayout,
        read_columns: Sequence[str],
        *,
        sample_filter: Callable[[str], bool] | None = None,
        parquet_file_cache_size: int = 5,
        disable_cache: bool = False,
        stream_batch_rows: int = 256,
    ):
        self.dataset_root = EPath(dataset_root)
        self.layout = layout
        self.read_columns = list(read_columns)
        self.sample_filter = sample_filter
        self.disable_cache = disable_cache
        self._pa, self._pq = _import_pyarrow()

        files = layout.files
        self._file_rows = [f.num_rows for f in files]
        self._cumsum: List[int] = []
        s = 0
        for n in self._file_rows:
            self._cumsum.append(s)
            s += n
        self._total = s
        # For each file, the first row of each row group
        self._row_group_starts: List[List[int]] = []
        for f in files:
            starts = []
            s = 0
            for n in f.row_group_rows:
                starts.append(s)
                s += n
            assert s == f.num_rows, f"Row group sizes of {f.rel_path} do not sum up to its rows"
            self._row_group_starts.append(starts)
        self._parquet_cache: OrderedDict[int, Tuple[Any, Any]] = OrderedDict()
        self._parquet_cache_max = max(1, parquet_file_cache_size)
        self._stream_batch_rows = stream_batch_rows

    def __len__(self) -> int:
        return self._total

    def __str__(self) -> str:
        return (
            f"IParquetReader(root={self.dataset_root}, rows={self._total}, "
            f"files={len(self.layout.files)})"
        )

    def _locate(self, global_idx: int) -> Tuple[int, int]:
        if global_idx < 0 or global_idx >= self._total:
            raise IndexError(f"Sample index {global_idx} out of range [0, {self._total})")
        # cumsum[f] <= global_idx < cumsum[f] + file_rows[f]
        f = bisect_right(self._cumsum, global_idx) - 1
        local = global_idx - self._cumsum[f]
        return f, local

    def _open_parquet_file(self, file_idx: int) -> Tuple[Any, Any]:
        rel = self.layout.files[file_idx].rel_path
        path = self.dataset_root / rel
        raw = path.open("rb")
        return raw, self._pq.ParquetFile(self._pa.PythonFile(raw))

    def _get_parquet_file(self, file_idx: int):
        if file_idx in self._parquet_cache:
            self._parquet_cache.move_to_end(file_idx)
            return self._parquet_cache[file_idx]
        self._parquet_cache[file_idx] = self._open_parquet_file(file_idx)
        self._parquet_cache.move_to_end(file_idx)
        while len(self._parquet_cache) > self._parquet_cache_max:
            old_idx, (old_raw, old_pf) = self._parquet_cache.popitem(last=False)
            old_pf.close()
            old_raw.close()
        return self._parquet_cache[file_idx]

    def _locate_row_group(self, file_idx: int, row_in_file: int) -> Tuple[int, int]:
        """Returns the row group index and the row index within that row group."""
        starts = self._row_group_starts[file_idx]
        row_group = bisect_right(starts, row_in_file) - 1
        return row_group, row_in_file - starts[row_group]

    @contextlib.contextmanager
    def _access_parquet_file(self, file_idx: int) -> Generator[Any, None, None]:
        """
        Context manager to access the ParquetFile of a file.

        - If caching is enabled, yields the cached ParquetFile and does not close it.
        - If caching is disabled, opens a fresh ParquetFile for the duration of the context.
        """
        if not self.disable_cache:
            yield self._get_parquet_file(file_idx)[1]
            return

        raw, pf = self._open_parquet_file(file_idx)
        try:
            yield pf
        finally:
            pf.close()
            raw.close()

    def _get_item(self, idx: int, columns: Sequence[str] | None = None) -> dict[str, Any] | None:
        key = str(idx)
        if self.sample_filter is not None and not self.sample_filter(key):
            return None
        cols = list(self.read_columns if columns is None else columns)
        f_idx, local = self._locate(idx)
        row_group, local_rg = self._locate_row_group(f_idx, local)
        with self._access_parquet_file(f_idx) as pf:
            table = pf.read_row_group(row_group, columns=cols)
        return self._make_sample(idx, f_idx, cols, _table_row_to_dict(table, local_rg, cols))

    def iter_slice(self, start: int, stop: int) -> Generator[FilteredSample | None, None, None]:
        """Yields the rows ``start`` to ``stop - 1``, decoding them incrementally in batches. Each
        row is decoded once. Starting within a row group decodes its preceding rows."""
        while start < stop:
            f_idx, local = self._locate(start)
            file_stop = min(stop, self._cumsum[f_idx] + self._file_rows[f_idx])
            yield from self._iter_file_rows(f_idx, local, local + file_stop - start)
            start = file_stop

    def _iter_file_rows(
        self, file_idx: int, start: int, stop: int
    ) -> Generator[FilteredSample | None, None, None]:
        first_row_group, skip = self._locate_row_group(file_idx, start)
        last_row_group, _ = self._locate_row_group(file_idx, stop - 1)
        idx = self._cumsum[file_idx] + start
        remaining = stop - start
        # Every stream has its own file handle, cached handles may be closed while it is active
        raw, pf = self._open_parquet_file(file_idx)
        try:
            for batch in pf.iter_batches(
                batch_size=self._stream_batch_rows,
                row_groups=list(range(first_row_group, last_row_group + 1)),
                columns=self.read_columns,
                use_threads=False,
            ):
                if skip >= batch.num_rows:
                    skip -= batch.num_rows
                    continue
                rows = batch.slice(skip, remaining).to_pylist()
                skip = 0
                for row in rows:
                    if self.sample_filter is not None and not self.sample_filter(str(idx)):
                        yield None
                    else:
                        yield self._make_sample(idx, file_idx, self.read_columns, row)
                    idx += 1
                remaining -= len(rows)
                if remaining == 0:
                    return
            raise RuntimeError(f"Parquet file {self.layout.files[file_idx].rel_path} ended early")
        finally:
            pf.close()
            raw.close()

    def _make_sample(
        self, idx: int, file_idx: int, cols: Sequence[str], row: dict[str, Any]
    ) -> dict[str, Any]:
        key = str(idx)
        shard_name = self.layout.files[file_idx].rel_path
        file_names = tuple(f"{key}.{c}" for c in cols)
        return dict(
            __key__=key,
            __shard__=shard_name,
            __restore_key__=("Webdataset", idx),
            __sources__=(
                SourceInfo(
                    dataset_path=str(self.dataset_root),
                    index=idx,
                    shard_name=shard_name,
                    file_names=file_names,
                ),
            ),
            **row,
        )

    def __getitem__(self, idx: int | str) -> FilteredSample | tuple[Any, SourceInfo] | None:
        full_entry_name = False
        columns: Sequence[str] | None = None
        if isinstance(idx, str):
            num_idx = idx
            if "." in idx:
                num_idx, column_name = idx.split(".", 1)
                if column_name not in self.read_columns:
                    raise KeyError(
                        f"Column {column_name!r} of {idx!r} is not read, "
                        f"read columns: {self.read_columns}"
                    )
                full_entry_name = True
                columns = (column_name,)
            try:
                idx = int(num_idx)
            except ValueError:
                raise ValueError(f"Invalid parquet sample key: {idx}") from None

        assert isinstance(idx, int)
        sample = self._get_item(idx, columns=columns)
        if sample is None:
            return None
        if full_entry_name:
            assert len(sample["__sources__"]) == 1
            return sample[column_name], sample["__sources__"][0]
        return sample

    def list_all_samples(self) -> Generator[Tuple[str, int, int], None, None]:
        file_sizes = [self.dataset_root / fe.rel_path for fe in self.layout.files]
        approx_row = [max(1, p.size() // max(1, n)) for p, n in zip(file_sizes, self._file_rows)]
        for i in range(self._total):
            f_idx, _ = self._locate(i)
            yield str(i), approx_row[f_idx], 0

    def list_all_sample_parts(self) -> Generator[Tuple[str, int, int], None, None]:
        for i in range(self._total):
            for col in self.read_columns:
                yield f"{i}.{col}", 1, 0

    def list_sample_parts(self, sample_key: str) -> Generator[Tuple[str, int, int], None, None]:
        try:
            sample_idx = int(sample_key)
        except ValueError:
            raise ValueError(f"Invalid parquet sample key: {sample_key}") from None
        if sample_idx < 0 or sample_idx >= self._total:
            raise ValueError(f"Invalid parquet sample key: {sample_key}")
        for col in self.read_columns:
            yield f"{sample_key}.{col}", 1, 0

    def get_total_size(self) -> int:
        total = 0
        for fe in self.layout.files:
            p = self.dataset_root / fe.rel_path
            total += p.size()
        return total

    def close(self) -> None:
        while self._parquet_cache:
            _, (raw, pf) = self._parquet_cache.popitem(last=False)
            pf.close()
            raw.close()
