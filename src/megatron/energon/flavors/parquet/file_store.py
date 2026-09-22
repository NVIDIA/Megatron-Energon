# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Sequence

from megatron.energon.cache.base import FileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.common.manifest.io import get_dataset_info
from megatron.energon.flavors.common.manifest.types import ShardInfo
from megatron.energon.flavors.parquet.prepare import (
    assert_layout_columns_subset,
    scan_parquet_file,
    scan_parquet_shards,
)
from megatron.energon.flavors.parquet.reader import IParquetReader


class ParquetFileStore(IParquetReader, FileStore[Any]):
    """Random access to a single Parquet file or manifest Parquet shard list."""

    def __init__(
        self,
        dataset_path: EPath | str,
        *,
        shards: Sequence[ShardInfo] | None = None,
        part_filter: Callable[[str], bool] | None = None,
        parquet_file_cache_size: int = 5,
    ):
        path = EPath(dataset_path)
        if path.is_file():
            reader_base_path = path.parent
            layout = scan_parquet_file(path)
        else:
            if shards is None:
                info = get_dataset_info(path)
                shards = [
                    ShardInfo(name=name, path=path / name, count=int(count))
                    for name, count in info["shard_counts"].items()
                ]
            reader_base_path = path
            layout = scan_parquet_shards(list(shards))

        layout_cols = list(layout.columns)
        if part_filter is None:
            read_columns = layout_cols
        else:
            read_columns = [column for column in layout_cols if part_filter(column)]
        if not read_columns:
            raise ValueError(
                "part_filter excluded all Parquet columns; nothing to load. "
                f"Layout columns: {layout_cols}"
            )
        assert_layout_columns_subset(layout_cols, read_columns)
        super().__init__(
            reader_base_path,
            layout,
            read_columns,
            parquet_file_cache_size=parquet_file_cache_size,
        )
        self.dataset_path = path

    def get_path(self) -> str:
        return str(self.dataset_path)
