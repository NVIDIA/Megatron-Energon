# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "DefaultParquetDatasetFactory",
    "DefaultParquetShardListDatasetFactory",
    "IParquetReader",
    "ParquetDatasetFactory",
    "ParquetFileEntry",
    "ParquetFileStore",
    "ParquetLayout",
    "ParquetPreparator",
    "ParquetShardListDatasetFactory",
    "parse_parquet_layout",
    "scan_parquet_file",
    "scan_parquet_dataset",
    "scan_parquet_shards",
]


def __getattr__(name: str):
    if name in (
        "DefaultParquetDatasetFactory",
        "DefaultParquetShardListDatasetFactory",
        "ParquetDatasetFactory",
        "ParquetPreparator",
        "ParquetShardListDatasetFactory",
    ):
        from megatron.energon.flavors.parquet import dataset as _dataset

        return getattr(_dataset, name)
    if name in (
        "parse_parquet_layout",
        "scan_parquet_file",
        "scan_parquet_dataset",
        "scan_parquet_shards",
        "ParquetFileEntry",
        "ParquetLayout",
    ):
        from megatron.energon.flavors.parquet import prepare as _prepare

        return getattr(_prepare, name)
    if name == "IParquetReader":
        from megatron.energon.flavors.parquet import reader as _reader

        return _reader.IParquetReader
    if name == "ParquetFileStore":
        from megatron.energon.flavors.parquet.file_store import ParquetFileStore

        return ParquetFileStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
