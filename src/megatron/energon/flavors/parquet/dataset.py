# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Any, Callable, Dict, List, Union

from megatron.energon.cache import FileStore
from megatron.energon.decoders import DEFAULT_DECODER, SampleDecoder
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_manifest_dataset import BaseManifestShardListDatasetFactory
from megatron.energon.flavors.base_single_file_dataset import BaseSingleFileDatasetFactory
from megatron.energon.flavors.common.manifest.io import ShardListMeta
from megatron.energon.flavors.common.manifest.types import DatasetSubset, ManifestSplits
from megatron.energon.flavors.common.sample_record import SampleRecord
from megatron.energon.flavors.crude import CrudeSample
from megatron.energon.flavors.dataset_factory_resolver import (
    PRIORITY_SINGLE_FILE,
    register_dataset_factory_provider,
)
from megatron.energon.flavors.dataset_type import EnergonDatasetType
from megatron.energon.flavors.parquet.prepare import (
    assert_layout_columns_subset,
    scan_parquet_file,
    scan_parquet_shards,
)
from megatron.energon.flavors.parquet.reader import IParquetReader
from megatron.energon.worker import WorkerConfig

logger = logging.getLogger(__name__)

_PARQUET_SAMPLE_META = frozenset(
    {"__key__", "__shard__", "__restore_key__", "__sources__"},
)


def _select_columns(
    layout_columns: List[str], part_filter: Callable[[str], bool] | None
) -> list[str]:
    if part_filter is None:
        read_columns = layout_columns
    else:
        read_columns = [column for column in layout_columns if part_filter(column)]
    if not read_columns:
        raise ValueError(
            "part_filter excluded all Parquet columns; nothing to load. "
            f"Layout columns: {layout_columns}"
        )
    assert_layout_columns_subset(layout_columns, read_columns)
    return read_columns


class ParquetPreparator:
    """Validates and counts rows for a Parquet file or manifest shard list."""

    @classmethod
    def prepare_dataset(cls, path: Union[str, EPath]) -> int:
        layout = scan_parquet_file(EPath(path))
        return layout.total_rows


class ParquetDatasetFactory(
    BaseSingleFileDatasetFactory[CrudeSample],
    ParquetPreparator,
):
    """Crude dataset over one Parquet file."""

    __sample_type__ = CrudeSample

    path: EPath
    training: bool
    worker_config: WorkerConfig

    def __init__(
        self,
        path: EPath,
        *,
        training: bool,
        worker_config: WorkerConfig,
        shuffle_over_epochs: int | None = 1,
        parallel_shard_iters: int | None = None,
        max_samples_per_sequence: int | None = None,
        subset: DatasetSubset | None = None,
        part_filter: Callable[[str], bool] | None = None,
        filter_name: str | None = None,
    ):
        path = EPath(path)
        assert path.is_file(), f"Parquet dataset path must be a file: {path}"
        self._layout = scan_parquet_file(path)
        self._reader_base_path = path.parent
        self._read_columns = _select_columns(list(self._layout.columns), part_filter)
        super().__init__(
            path,
            sample_count=self._layout.total_rows,
            training=training,
            worker_config=worker_config,
            shuffle_over_epochs=shuffle_over_epochs,
            parallel_shard_iters=parallel_shard_iters,
            max_samples_per_sequence=max_samples_per_sequence,
            subset=subset,
            part_filter=part_filter,
            filter_name=filter_name,
        )

    def _build_reader(
        self,
        *,
        parallel_shard_iters: int,
        part_filter: Callable[[str], bool] | None,
    ):
        columns = (
            [column for column in self._read_columns if part_filter(column)]
            if part_filter is not None
            else self._read_columns
        )
        return IParquetReader(
            self._reader_base_path,
            self._layout,
            columns,
            parquet_file_cache_size=parallel_shard_iters,
        )

    def as_file_store(self) -> FileStore:
        from megatron.energon.flavors.parquet.file_store import ParquetFileStore

        return ParquetFileStore(self.path, part_filter=self.part_filter)

    def load_sample(self, sample: SampleRecord) -> CrudeSample:
        return CrudeSample(sample)

    def _load_sample(self, sample: SampleRecord) -> CrudeSample:
        return self.load_sample(sample)

    def config(self) -> Dict[str, Any]:
        return dict(
            type=type(self).__qualname__,
            training=self.training,
            _path=str(self.path),
            parquet_filename=self.path.name,
            count=len(self),
            shuffle_over_epochs=self.shuffle_over_epochs,
            parallel_shard_iters=self.parallel_shard_iters,
            max_samples_per_sequence=self.max_samples_per_sequence,
            subset=self.subset.config() if self.subset is not None else None,
            filter_name=self.filter_name,
        )

    def __str__(self):
        return f"{type(self).__name__}(path={self.path})"


class ParquetShardListDatasetFactory(BaseManifestShardListDatasetFactory[CrudeSample]):
    """Crude dataset over a manifest-defined list of Parquet shards."""

    __sample_type__ = CrudeSample

    def __init__(
        self,
        path: EPath,
        *,
        training: bool,
        worker_config: WorkerConfig,
        split_config: str | ManifestSplits = "split.yaml",
        split_part: str = "train",
        shuffle_over_epochs: int | None = 1,
        parallel_shard_iters: int | None = None,
        max_samples_per_sequence: int | None = None,
        subset: DatasetSubset | None = None,
        part_filter: Callable[[str], bool] | None = None,
        filter_name: str | None = None,
    ):
        self.split_config = split_config
        self.split_part = split_part
        self._read_columns: list[str] = []
        super().__init__(
            path,
            split_part=split_part,
            split_config=split_config,
            training=training,
            worker_config=worker_config,
            shuffle_over_epochs=shuffle_over_epochs,
            parallel_shard_iters=parallel_shard_iters,
            max_samples_per_sequence=max_samples_per_sequence,
            subset=subset,
            part_filter=part_filter,
            filter_name=filter_name,
        )
        self._read_columns = _select_columns(list(self._layout.columns), part_filter)

    def _validate_manifest_meta(self, meta: ShardListMeta) -> None:
        if meta.sample_excludes:
            raise ValueError("Parquet shard-list datasets do not support sample-level excludes")
        self._layout = scan_parquet_shards(meta.shards)

    def _build_reader(
        self,
        *,
        parallel_shard_iters: int,
        part_filter: Callable[[str], bool] | None,
    ):
        columns = (
            [column for column in self._read_columns if part_filter(column)]
            if part_filter is not None
            else self._read_columns
        )
        return IParquetReader(
            self.path,
            self._layout,
            columns,
            parquet_file_cache_size=parallel_shard_iters,
        )

    def as_file_store(self) -> FileStore:
        from megatron.energon.flavors.parquet.file_store import ParquetFileStore

        return ParquetFileStore(self.path, shards=self.shards, part_filter=self.part_filter)

    def load_sample(self, sample: SampleRecord) -> CrudeSample:
        return CrudeSample(sample)

    def config(self) -> Dict[str, Any]:
        return dict(
            type=type(self).__qualname__,
            training=self.training,
            _path=str(self.path),
            parquet_shard_count=len(self.shards),
            count=len(self),
            split_config=self.split_config if isinstance(self.split_config, str) else "<inline>",
            split_part=self.split_part,
            shuffle_over_epochs=self.shuffle_over_epochs,
            parallel_shard_iters=self.parallel_shard_iters,
            max_samples_per_sequence=self.max_samples_per_sequence,
            subset=self.subset.config() if self.subset is not None else None,
            filter_name=self.filter_name,
        )

    def __str__(self):
        return f"{type(self).__name__}(path={self.path}, shards={len(self.shards)})"


class _DefaultParquetMixin:
    def _init_default_parquet(
        self,
        *,
        subflavors: dict[str, Any] | None,
        decoder: SampleDecoder | None,
        decode_map: dict[str, str] | None,
    ) -> None:
        self.subflavors = subflavors or {}
        self._decoder = decoder
        self._decode_map = decode_map or {}

    def load_sample(self, sample: SampleRecord) -> CrudeSample:
        if self._decoder is not None:
            for key, extension in self._decode_map.items():
                if key in sample:
                    sample[key] = self._decoder.decode(
                        f"{sample['__key__']}.{extension}", sample[key]
                    )
        sample["__subflavors__"] = self.subflavors
        return super().load_sample(sample)

    def config(self) -> Dict[str, Any]:
        return dict(
            **super().config(),
            subflavors=self.subflavors,
            decode_map=self._decode_map,
            **(self._decoder.config() if self._decoder is not None else {}),
        )


@register_dataset_factory_provider(priority=PRIORITY_SINGLE_FILE)
class DefaultParquetDatasetFactory(_DefaultParquetMixin, ParquetDatasetFactory):
    """Single-file Parquet factory that decodes selected columns and attaches subflavors."""

    @classmethod
    def detect_path(cls, path: EPath) -> EnergonDatasetType | None:
        if path.name.endswith(".parquet") and path.is_file():
            return EnergonDatasetType.PARQUET
        return None

    @classmethod
    def from_path(
        cls,
        path: EPath,
        *,
        dataset_config: str | None = None,
        split_config: str | None = None,
        sample_type: type | None = None,
        **kwargs,
    ) -> "DefaultParquetDatasetFactory":
        assert sample_type is CrudeSample or sample_type is None, (
            f"Sample type must be CrudeSample for Parquet datasets, but got {sample_type}"
        )
        assert dataset_config is None, (
            f"Dataset config must be None for Parquet datasets, but got {dataset_config}"
        )
        assert split_config is None, (
            f"Split config must be None for Parquet datasets, but got {split_config}"
        )
        kwargs.pop("split_part", None)
        return cls(
            path,
            **kwargs,
        )

    def __init__(
        self,
        path: EPath,
        *,
        subflavors: dict[str, Any] | None = None,
        decoder: SampleDecoder | None = DEFAULT_DECODER,
        decode_map: dict[str, str] | None = None,
        **kwargs,
    ):
        if "decoder" in kwargs:
            del kwargs["decoder"]
        super().__init__(path, **kwargs)
        self._init_default_parquet(
            subflavors=subflavors,
            decoder=decoder,
            decode_map=decode_map,
        )


class DefaultParquetShardListDatasetFactory(_DefaultParquetMixin, ParquetShardListDatasetFactory):
    """Manifest Parquet factory that decodes selected columns and attaches subflavors."""

    def __init__(
        self,
        path: EPath,
        *,
        subflavors: dict[str, Any] | None = None,
        decoder: SampleDecoder | None = DEFAULT_DECODER,
        decode_map: dict[str, str] | None = None,
        **kwargs,
    ):
        if "decoder" in kwargs:
            del kwargs["decoder"]
        super().__init__(path, **kwargs)
        self._init_default_parquet(
            subflavors=subflavors,
            decoder=decoder,
            decode_map=decode_map,
        )
