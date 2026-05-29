# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Any, Callable, Dict, List, Union

from megatron.energon.cache import FileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import (
    BaseCoreDatasetFactory,
    SavableDataset,
)
from megatron.energon.flavors.crude import CrudeSample
from megatron.energon.flavors.parquet.prepare import (
    assert_layout_columns_subset,
    scan_parquet_dataset,
)
from megatron.energon.flavors.parquet.reader import IParquetReader
from megatron.energon.flavors.webdataset.base_webdataset import _print_shard_slices
from megatron.energon.flavors.webdataset.sample_decoder import DEFAULT_DECODER, SampleDecoder
from megatron.energon.flavors.webdataset.sample_loader import (
    RawSampleData,
    WebdatasetSampleLoaderDataset,
)
from megatron.energon.flavors.webdataset.sharder import Sharder
from megatron.energon.flavors.webdataset.structs import (
    DatasetSubset,
    FilteredSample,
    ShardInfo,
)
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.map_dataset import MapDataset

logger = logging.getLogger(__name__)

_PARQUET_SAMPLE_META = frozenset(
    {"__key__", "__shard__", "__restore_key__", "__sources__"},
)


class ParquetPreparator:
    """Validates and counts rows for a Parquet dataset directory (like :class:`JsonlPreparator`)."""

    @classmethod
    def prepare_dataset(cls, path: Union[str, EPath]) -> int:
        layout = scan_parquet_dataset(EPath(path))
        return layout.total_rows


class ParquetDatasetFactory(
    BaseCoreDatasetFactory[CrudeSample],
    ParquetPreparator,
    Sharder,
):
    """Crude dataset over a directory of Parquet files (layout discovered at load time)."""

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
    ):
        assert self.__sample_type__ is not None, f"Class {type(self)} must define __sample_type__"
        self.path = EPath(path)
        assert self.path.is_dir(), f"Parquet dataset path must be a directory: {self.path}"
        self.paths = [self.path]
        self.training = training
        self.worker_config = worker_config
        self.shuffle_over_epochs = shuffle_over_epochs
        self.parallel_shard_iters = parallel_shard_iters
        self.max_samples_per_sequence = max_samples_per_sequence
        self.subset = subset
        self.part_filter = part_filter

        self._layout = scan_parquet_dataset(self.path)
        layout_cols: List[str] = list(self._layout.columns)
        if part_filter is None:
            read_columns = layout_cols
        else:
            read_columns = [c for c in layout_cols if part_filter(c)]
        if not read_columns:
            raise ValueError(
                "part_filter excluded all Parquet columns; nothing to load. "
                f"Layout columns: {layout_cols}"
            )
        assert_layout_columns_subset(layout_cols, read_columns)
        self._read_columns = read_columns

        # part_filter selects which Parquet columns are read into each sample;
        # it does not remove samples from the dataset (length is always full layout row count).
        self._len = self._layout.total_rows
        self._virtual_shards = [
            ShardInfo(
                name=fe.rel_path,
                path=self.path / fe.rel_path,
                count=fe.num_rows,
            )
            for fe in self._layout.files
        ]

    def __len__(self) -> int:
        return self._len

    def build(
        self, worker_rotation_offset: int = 0, part_filter: Callable[[str], bool] | None = None
    ) -> SavableDataset[CrudeSample]:
        if self.parallel_shard_iters is None:
            parallel_shard_iters = 16 if self.training else 1
        else:
            parallel_shard_iters = self.parallel_shard_iters

        effective_pf = part_filter
        if self.part_filter is not None:
            if effective_pf is not None:
                inner_pf, outer_pf = effective_pf, self.part_filter
                effective_pf = lambda p, _i=inner_pf, _o=outer_pf: _o(p) and _i(p)
            else:
                effective_pf = self.part_filter

        if effective_pf is not None:
            columns = [c for c in self._read_columns if effective_pf(c)]
        else:
            columns = self._read_columns

        workers_sample_slice_offsets = self.shard_workers(
            self._virtual_shards,
            worker_config=self.worker_config,
            max_samples_per_sequence=self.max_samples_per_sequence,
            rotation_offset=worker_rotation_offset,
            subset=self.subset,
        )
        _print_shard_slices(self.worker_config, self._virtual_shards, workers_sample_slice_offsets)

        reader = IParquetReader(
            self.path,
            self._layout,
            columns,
            parquet_file_cache_size=parallel_shard_iters,
        )

        dataset = WebdatasetSampleLoaderDataset(
            join_readers=[reader],
            workers_sample_slice_offsets=workers_sample_slice_offsets,
            worker_config=self.worker_config,
            shuffle_over_epochs=self.shuffle_over_epochs if self.training else None,
            parallel_slice_iters=parallel_shard_iters,
        )
        return MapDataset(
            dataset,
            self._load_sample_raw,
            stateless_map_fn=True,
            map_fn_config=self.config,
            worker_config=self.worker_config,
        )

    def as_file_store(self) -> FileStore:
        from megatron.energon.cache.file_store import ParquetFileStore

        return ParquetFileStore(self.path, part_filter=self.part_filter)

    def _load_sample(self, sample: FilteredSample) -> CrudeSample:
        return CrudeSample(sample)

    def _load_sample_raw(self, raw_sample: RawSampleData) -> CrudeSample:
        assert len(raw_sample.data) == 1 and raw_sample.data[0] is not None
        return self._load_sample(raw_sample.data[0])

    def config(self) -> Dict[str, Any]:
        return dict(
            type=type(self).__qualname__,
            training=self.training,
            _path=str(self.path),
            count=self._len,
            shuffle_over_epochs=self.shuffle_over_epochs,
            parallel_shard_iters=self.parallel_shard_iters,
            max_samples_per_sequence=self.max_samples_per_sequence,
            subset=self.subset.config() if self.subset is not None else None,
        )

    def __str__(self):
        return f"{type(self).__name__}(path={self.path})"


class DefaultParquetDatasetFactory(ParquetDatasetFactory):
    """
    Builds a ``row`` dict of Parquet column values, optionally runs :class:`SampleDecoder` on
    selected columns (binary cells), and attaches subflavors.

    ``decode_map`` maps **Parquet column names** to a
    **synthetic filename extension** (e.g. ``\"png\"`` or ``\"jpg\"``) passed to
    :meth:`SampleDecoder.decode` so webdataset image/video handlers can run on raw bytes.
    Non-bytes-like cell values are left unchanged.
    """

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
        self.subflavors = subflavors or {}
        self._decoder = decoder
        self._decode_map = decode_map or {}

    def _load_sample(self, sample: FilteredSample) -> CrudeSample:
        if self._decoder is not None:
            for k, v in self._decode_map.items():
                if k in sample:
                    sample[k] = self._decoder.decode(f"{sample['__key__']}.{v}", sample[k])
        sample["__subflavors__"] = self.subflavors
        return super()._load_sample(sample)

    def config(self) -> Dict[str, Any]:
        return dict(
            **super().config(),
            subflavors=self.subflavors,
            decode_map=self._decode_map,
            **(self._decoder.config() if self._decoder is not None else {}),
        )
