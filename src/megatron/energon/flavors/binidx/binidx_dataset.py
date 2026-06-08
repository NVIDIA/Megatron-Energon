# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Any, Callable, Dict, Optional

import numpy

from megatron.energon.cache import FileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_single_file_dataset import BaseSingleFileDatasetFactory
from megatron.energon.flavors.binidx.binidx_reader import BinIdxReader
from megatron.energon.flavors.common.dataset_sampler import RawSampleData
from megatron.energon.flavors.common.manifest.types import DatasetSubset
from megatron.energon.flavors.common.sample_record import SampleRecord
from megatron.energon.flavors.crude import CrudeSample
from megatron.energon.worker import WorkerConfig

logger = logging.getLogger(__name__)


class BinIdxDatasetFactory(BaseSingleFileDatasetFactory[CrudeSample]):
    """Factory class for creating a crude dataset from Megatron-LM bin-idx files."""

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
        filter_name: Optional[str] = None,
    ):
        path = EPath(path)
        super().__init__(
            path,
            sample_count=BinIdxReader.count_samples(path),
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
        return BinIdxReader(
            self.path,
            index_cache_size=parallel_shard_iters,
        )

    def _load_fn(
        self, part_filter: Callable[[str], bool] | None
    ) -> Callable[[RawSampleData], CrudeSample]:
        if part_filter is not None and not part_filter("tokens"):

            def load_fn(sample: RawSampleData) -> CrudeSample:
                assert sample.data[0] is not None
                sample.data[0].pop("tokens", None)
                return self._load_sample_raw(sample)

            return load_fn
        return self._load_sample_raw

    def as_file_store(self) -> FileStore:
        from megatron.energon.flavors.binidx.file_store import BinIdxFileStore

        return BinIdxFileStore(self.path)

    def load_sample(self, sample: SampleRecord) -> CrudeSample:
        return CrudeSample(sample)

    def _load_sample(self, sample: SampleRecord) -> CrudeSample:
        return self.load_sample(sample)

    def config(self) -> Dict[str, Any]:
        return dict(
            type=type(self).__qualname__,
            training=self.training,
            _path=str(self.path),
            bin_filename=self.path.name,
            count=len(self),
            shuffle_over_epochs=self.shuffle_over_epochs,
            parallel_shard_iters=self.parallel_shard_iters,
            max_samples_per_sequence=self.max_samples_per_sequence,
            subset=self.subset.config() if self.subset is not None else None,
            filter_name=self.filter_name,
        )

    def __str__(self):
        return f"{type(self).__name__}(path={self.path})"


class DefaultBinIdxDatasetFactory(BinIdxDatasetFactory):
    """Adds subflavors to the sample and decodes token bytes back to a numpy array."""

    def __init__(self, path: EPath, *, subflavors: Optional[Dict[str, Any]] = None, **kwargs):
        if "decoder" in kwargs:
            del kwargs["decoder"]
        super().__init__(path, **kwargs)
        self.subflavors = subflavors
        self._dtype = BinIdxReader.read_dtype(self.path)

    def load_sample(self, sample: SampleRecord) -> CrudeSample:
        sample["__subflavors__"] = self.subflavors
        sample["tokens"] = numpy.frombuffer(sample["tokens"], dtype=self._dtype)
        return super().load_sample(sample)

    def config(self) -> Dict[str, Any]:
        return dict(
            **super().config(),
            subflavors=self.subflavors,
        )
