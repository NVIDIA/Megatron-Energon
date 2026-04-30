# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Any, Callable, Dict, Optional

import numpy

from megatron.energon.cache import FileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import (
    BaseCoreDatasetFactory,
    SavableDataset,
)
from megatron.energon.flavors.binidx.binidx_reader import IBinIdxReader
from megatron.energon.flavors.crude import CrudeSample
from megatron.energon.flavors.webdataset.base_webdataset import _print_shard_slices
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


class BinIdxDatasetFactory(
    BaseCoreDatasetFactory[CrudeSample],
    Sharder,
):
    """
    Factory class for creating a crude dataset from Megatron-LM bin-idx files.

    This factory creates datasets from pre-tokenized binary files (.bin + .idx)
    where the .idx file contains sequence offsets and the .bin file contains token data.
    The samples are returned as CrudeSample objects containing the raw `tokens` array.
    """

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
        shuffle_over_epochs: Optional[int] = 1,
        parallel_shard_iters: Optional[int] = None,
        max_samples_per_sequence: Optional[int] = None,
        subset: Optional[DatasetSubset] = None,
        part_filter: Optional[Callable[[str], bool]] = None,
    ):
        """
        Factory for a bin-idx file pair as a crude dataset.

        Args:
            path: Path to the .bin file.
            training: If true, apply shuffling and loop the dataset.
            worker_config: Configuration for the workers.
            shuffle_over_epochs: Only effective if training=True.
                How many epochs to shuffle over if training.
            parallel_shard_iters: Number of parallel opened shards per worker.
            max_samples_per_sequence: Maximum number of samples per sequence.
            subset: If specified, the dataset will be subsetted.
            part_filter: (internal) Function for filtering by dict keys.
        """
        assert self.__sample_type__ is not None, f"Class {type(self)} must define __sample_type__"
        self.path = EPath(path)
        self.paths = [self.path]
        self.training = training
        self.worker_config = worker_config
        self.shuffle_over_epochs = shuffle_over_epochs
        self.parallel_shard_iters = parallel_shard_iters
        self.max_samples_per_sequence = max_samples_per_sequence
        self.subset = subset
        self.part_filter = part_filter
        self._len = IBinIdxReader.count_samples(self.path)

    def __len__(self) -> int:
        return self._len

    def build(
        self, worker_rotation_offset: int = 0, part_filter: Callable[[str], bool] | None = None
    ) -> SavableDataset[CrudeSample]:
        if self.parallel_shard_iters is None:
            if self.training:
                parallel_shard_iters = 16
            else:
                parallel_shard_iters = 1
        else:
            parallel_shard_iters = self.parallel_shard_iters

        if self.part_filter is not None:
            if part_filter is not None:
                inner_pf, outer_pf = part_filter, self.part_filter
                part_filter = lambda p, _i=inner_pf, _o=outer_pf: _o(p) and _i(p)
            else:
                part_filter = self.part_filter

        virtual_shards = [
            ShardInfo(
                name=self.path.name,
                path=self.path,
                count=self._len,
            )
        ]

        workers_sample_slice_offsets = self.shard_workers(
            virtual_shards,
            worker_config=self.worker_config,
            max_samples_per_sequence=self.max_samples_per_sequence,
            rotation_offset=worker_rotation_offset,
            subset=self.subset,
        )
        _print_shard_slices(self.worker_config, virtual_shards, workers_sample_slice_offsets)

        reader = IBinIdxReader(
            self.path,
            index_cache_size=parallel_shard_iters,
        )

        dataset = WebdatasetSampleLoaderDataset(
            join_readers=[reader],
            workers_sample_slice_offsets=workers_sample_slice_offsets,
            worker_config=self.worker_config,
            shuffle_over_epochs=self.shuffle_over_epochs if self.training else None,
            parallel_slice_iters=parallel_shard_iters,
        )
        if part_filter is not None and not part_filter("tokens"):

            def load_fn(sample: RawSampleData) -> CrudeSample:
                sample.data[0].pop("tokens", None)
                return self._load_sample_raw(sample)
        else:
            load_fn = self._load_sample_raw
        return MapDataset(
            dataset,
            load_fn,
            stateless_map_fn=True,
            map_fn_config=self.config,
            worker_config=self.worker_config,
        )

    def as_file_store(self) -> "FileStore":
        from megatron.energon.cache.file_store import BinIdxFileStore

        return BinIdxFileStore(self.path)

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
            bin_filename=self.path.name,
            count=self._len,
            shuffle_over_epochs=self.shuffle_over_epochs,
            parallel_shard_iters=self.parallel_shard_iters,
            max_samples_per_sequence=self.max_samples_per_sequence,
            subset=self.subset.config() if self.subset is not None else None,
        )

    def __str__(self):
        return f"{type(self).__name__}(path={self.path})"


class DefaultBinIdxDatasetFactory(BinIdxDatasetFactory):
    """
    Adds subflavors to the sample and decodes token bytes back to a numpy array.
    """

    def __init__(self, path: EPath, *, subflavors: dict[str, Any] | None = None, **kwargs):
        if "decoder" in kwargs:
            del kwargs["decoder"]
        super().__init__(path, **kwargs)
        self.subflavors = subflavors or {}
        self._dtype = IBinIdxReader.read_dtype(self.path)

    def _load_sample(self, sample: FilteredSample) -> CrudeSample:
        sample["__subflavors__"] = self.subflavors
        if "tokens" in sample:
            sample["tokens"] = numpy.frombuffer(sample["tokens"], dtype=self._dtype)
        return CrudeSample(sample)

    def config(self) -> Dict[str, Any]:
        return dict(
            **super().config(),
            subflavors=self.subflavors,
        )
