# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import json
import logging
from typing import Any, Callable, Dict, Optional

from megatron.energon.cache import FileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import (
    BaseCoreDatasetFactory,
    SavableDataset,
)
from megatron.energon.flavors.crude import CrudeSample
from megatron.energon.flavors.jsonl.ijsonl import IJsonlIndexReader
from megatron.energon.flavors.jsonl.jsonl_prepare import JsonlPreparator
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


class CrudeJsonlDatasetFactory(
    BaseCoreDatasetFactory[CrudeSample],
    JsonlPreparator,
    Sharder,
):
    """
    Factory class for creating a crude dataset from JSONL (JSON Lines) files.

    This factory creates datasets from JSONL files where each line contains a JSON object.
    The samples are returned as CrudeSample objects (dictionary-like) containing the raw JSON data.
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
        Factory for a jsonl file as a crude dataset.

        Args:
            path: Path to the jsonl file.
            training: If true, apply shuffling and loop the dataset.
            worker_config: Configuration for the workers.
            shuffle_over_epochs: Only effective if training=True.
                How many epochs to shuffle over if training.
                If = 1, every sample is seen exactly once per epoch.
                If > 1, samples (or rather shard slices) are shuffled within this number of epochs
                (i.e. randomly selected without replacement).
                If -1, the shards are effectively shuffle over infinite epochs (i.e. shard slices
                are drawn with replacement).
            parallel_shard_iters: Number of parallel opened shards per worker, shuffling between.
            max_samples_per_sequence: Maximum number of samples per sequence (=how many samples
                    will be sequentially iterated).
            subset: If specified, the dataset will be subsetted.
            part_filter: (internal) Function for filtering tar files by dict keys
        """
        assert self.__sample_type__ is not None, f"Class {type(self)} must define __sample_type__"
        self.path = path
        self.paths = [path]
        self.training = training
        self.worker_config = worker_config
        self.shuffle_over_epochs = shuffle_over_epochs
        self.parallel_shard_iters = parallel_shard_iters
        self.max_samples_per_sequence = max_samples_per_sequence
        self.subset = subset
        self.part_filter = part_filter
        if part_filter is None or part_filter("json"):
            self._len = IJsonlIndexReader.count_samples(path)
        else:
            self._len = 0
        assert self.path.size() == IJsonlIndexReader.size(path), (
            "The index of the jsonl file does not match the file. Regenerate the index."
        )

    def __len__(self) -> int:
        return self._len

    def build(self, worker_rotation_offset: int = 0) -> SavableDataset[CrudeSample]:
        from megatron.energon.flavors.jsonl.ijsonl_reader import IJsonlReader

        if self.parallel_shard_iters is None:
            if self.training:
                # 16 seems to be a good choice since we don't want too many file handles open
                parallel_shard_iters = 16
            else:
                parallel_shard_iters = 1
        else:
            parallel_shard_iters = self.parallel_shard_iters

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

        itar_reader = IJsonlReader(
            self.path,
            index_cache_size=parallel_shard_iters,
        )

        dataset = WebdatasetSampleLoaderDataset(
            join_readers=[itar_reader],
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

    def as_file_store(self) -> "FileStore":
        from megatron.energon.cache.file_store import JsonlFileStore

        return JsonlFileStore(self.path)

    def _load_sample(self, sample: FilteredSample) -> CrudeSample:
        return CrudeSample(sample)

    def _load_sample_raw(self, raw_sample: RawSampleData) -> CrudeSample:
        # Just a wrapper for the inner tuple. Tuple should be of length 1.
        assert len(raw_sample.data) == 1 and raw_sample.data[0] is not None
        return self._load_sample(raw_sample.data[0])

    def config(self) -> Dict[str, Any]:
        return dict(
            type=type(self).__qualname__,
            training=self.training,
            _path=str(self.path),
            jsonl_filename=self.path.name,
            count=self._len,
            shuffle_over_epochs=self.shuffle_over_epochs,
            parallel_shard_iters=self.parallel_shard_iters,
            max_samples_per_sequence=self.max_samples_per_sequence,
            subset=self.subset.config() if self.subset is not None else None,
        )

    def __str__(self):
        return f"{type(self).__name__}(path={self.path})"


class DefaultCrudeJsonlDatasetFactory(CrudeJsonlDatasetFactory):
    """
    Adds subflavors to the sample and loads the json.
    """

    def __init__(self, path: EPath, *, subflavors: Optional[Dict[str, Any]] = None, **kwargs):
        if "decoder" in kwargs:
            del kwargs["decoder"]
        super().__init__(path, **kwargs)
        self.subflavors = subflavors

    def _load_sample(self, sample: FilteredSample) -> CrudeSample:
        sample["__subflavors__"] = self.subflavors

        # Instead of using a decoder, we just load the json here, as we know it's json.
        sample["json"] = json.loads(sample["json"])

        return super()._load_sample(sample)

    def config(self) -> Dict[str, Any]:
        return dict(
            **super().config(),
            subflavors=self.subflavors,
        )
