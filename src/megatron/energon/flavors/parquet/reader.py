# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Random-access reader over a Parquet-tabular in-memory layout (global row index)."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from typing import Any, Callable, Generator, List, Sequence, Tuple

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.parquet.prepare import ParquetLayout
from megatron.energon.flavors.webdataset.structs import FilteredSample
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


class IParquetReader:
    """
    Parquet random access over rows described by a :class:`ParquetLayout`.
    """

    dataset_root: EPath
    layout: ParquetLayout
    read_columns: list[str]
    sample_filter: Callable[[str], bool] | None

    def __init__(
        self,
        dataset_root: EPath,
        layout: ParquetLayout,
        read_columns: Sequence[str],
        *,
        sample_filter: Callable[[str], bool] | None = None,
        parquet_file_cache_size: int = 5,
    ):
        self.dataset_root = EPath(dataset_root)
        self.layout = layout
        self.read_columns = list(read_columns)
        self.sample_filter = sample_filter
        self._pa, self._pq = _import_pyarrow()

        files = layout.files
        self._file_rows = [f.num_rows for f in files]
        self._cumsum: List[int] = []
        s = 0
        for n in self._file_rows:
            self._cumsum.append(s)
            s += n
        self._total = s
        self._parquet_cache: OrderedDict[int, Tuple[Any, Any]] = OrderedDict()
        self._parquet_cache_max = max(1, parquet_file_cache_size)

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

    def _get_parquet_file(self, file_idx: int):
        if file_idx in self._parquet_cache:
            self._parquet_cache.move_to_end(file_idx)
            return self._parquet_cache[file_idx]
        rel = self.layout.files[file_idx].rel_path
        path = self.dataset_root / rel
        raw = path.open("rb")
        pf = self._pq.ParquetFile(self._pa.PythonFile(raw))
        self._parquet_cache[file_idx] = (raw, pf)
        self._parquet_cache.move_to_end(file_idx)
        while len(self._parquet_cache) > self._parquet_cache_max:
            old_idx, (old_raw, old_pf) = self._parquet_cache.popitem(last=False)
            old_pf.close()
            old_raw.close()
        return self._parquet_cache[file_idx]

    def _get_item(self, idx: int, columns: Sequence[str] | None = None) -> dict[str, Any] | None:
        key = str(idx)
        if self.sample_filter is not None and not self.sample_filter(key):
            return None
        cols = list(self.read_columns if columns is None else columns)
        f_idx, local = self._locate(idx)
        shard_name = self.layout.files[f_idx].rel_path
        _, pf = self._get_parquet_file(f_idx)
        off = 0
        row_in_file = local
        for rg in range(pf.num_row_groups):
            nr = pf.metadata.row_group(rg).num_rows
            if row_in_file < off + nr:
                table = pf.read_row_group(rg, columns=cols)
                local_rg = row_in_file - off
                d = _table_row_to_dict(table, local_rg, cols)
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
                    **d,
                )
            off += nr
        raise RuntimeError(f"Row {row_in_file} out of range for file index {f_idx}")

    def __getitem__(self, idx: int | str) -> FilteredSample | tuple[Any, SourceInfo] | None:
        full_entry_name = False
        if isinstance(idx, str):
            num_idx = idx
            if "." in idx:
                num_idx, column_name = idx.split(".", 1)
                full_entry_name = True
                columns = (column_name,)
            try:
                idx = int(num_idx)
            except ValueError:
                raise ValueError(f"Invalid parquet sample key: {idx}") from None
        else:
            columns = None

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
