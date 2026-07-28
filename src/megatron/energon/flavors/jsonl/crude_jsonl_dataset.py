# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import json
import logging
from typing import Any, Callable, Dict, Optional

from megatron.energon.cache import FileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_manifest_dataset import BaseManifestShardListDatasetFactory
from megatron.energon.flavors.base_single_file_dataset import BaseSingleFileDatasetFactory
from megatron.energon.flavors.common.dataset_sampler import RawSampleData
from megatron.energon.flavors.common.manifest.io import ShardListMeta
from megatron.energon.flavors.common.manifest.types import DatasetSubset, ManifestSplits
from megatron.energon.flavors.common.sample_record import SampleRecord
from megatron.energon.flavors.crude import CrudeSample
from megatron.energon.flavors.dataset_factory_resolver import (
    PRIORITY_SINGLE_FILE,
    register_dataset_factory_provider,
)
from megatron.energon.flavors.dataset_type import EnergonDatasetType
from megatron.energon.flavors.jsonl.file_store import MultiJsonlFileStore
from megatron.energon.flavors.jsonl.ijsonl import IJsonlIndexReader
from megatron.energon.flavors.jsonl.jsonl_prepare import JsonlPreparator
from megatron.energon.flavors.jsonl.multi_jsonl_reader import MultiJsonlReader
from megatron.energon.worker import WorkerConfig

logger = logging.getLogger(__name__)


class CrudeJsonlDatasetFactory(
    BaseSingleFileDatasetFactory[CrudeSample],
    JsonlPreparator,
):
    """Factory class for creating a crude dataset from a single JSONL file."""

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
        original_len = IJsonlIndexReader.count_samples(path)
        super().__init__(
            path,
            sample_count=original_len,
            training=training,
            worker_config=worker_config,
            shuffle_over_epochs=shuffle_over_epochs,
            parallel_shard_iters=parallel_shard_iters,
            max_samples_per_sequence=max_samples_per_sequence,
            subset=subset,
            part_filter=part_filter,
            filter_name=filter_name,
        )
        assert IJsonlIndexReader.is_current(path), (
            "The index of the jsonl file does not match the file. Regenerate the index."
        )

    def _build_reader(
        self,
        *,
        parallel_shard_iters: int,
        part_filter: Callable[[str], bool] | None,
    ):
        from megatron.energon.flavors.jsonl.ijsonl_reader import IJsonlReader

        return IJsonlReader(
            self.path,
            index_cache_size=parallel_shard_iters,
        )

    def _load_fn(
        self, part_filter: Callable[[str], bool] | None
    ) -> Callable[[RawSampleData], CrudeSample]:
        if (part_filter is not None and not part_filter("json")) or (
            self.part_filter is not None and not self.part_filter("json")
        ):

            def load_fn(sample: RawSampleData) -> CrudeSample:
                assert sample.data[0] is not None
                sample.data[0].pop("json", None)
                return self._load_sample_raw(sample)

            return load_fn
        return self._load_sample_raw

    def as_file_store(self) -> FileStore:
        from megatron.energon.flavors.jsonl.file_store import JsonlFileStore

        return JsonlFileStore(self.path)

    def load_sample(self, sample: SampleRecord) -> CrudeSample:
        return CrudeSample(sample)

    def _load_sample(self, sample: SampleRecord) -> CrudeSample:
        return self.load_sample(sample)

    def config(self) -> Dict[str, Any]:
        return dict(
            type=type(self).__qualname__,
            training=self.training,
            _path=str(self.path),
            jsonl_filename=self.path.name,
            count=len(self),
            shuffle_over_epochs=self.shuffle_over_epochs,
            parallel_shard_iters=self.parallel_shard_iters,
            max_samples_per_sequence=self.max_samples_per_sequence,
            subset=self.subset.config() if self.subset is not None else None,
            filter_name=self.filter_name,
        )

    def __str__(self):
        return f"{type(self).__name__}(path={self.path})"


class CrudeJsonlShardListDatasetFactory(
    BaseManifestShardListDatasetFactory[CrudeSample],
):
    """Factory for a prepared logical dataset composed of many JSONL shards."""

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
        split_config: str | ManifestSplits = "split.yaml",
        split_part: str = "train",
        shuffle_over_epochs: Optional[int] = 1,
        parallel_shard_iters: Optional[int] = None,
        max_samples_per_sequence: Optional[int] = None,
        subset: Optional[DatasetSubset] = None,
        part_filter: Optional[Callable[[str], bool]] = None,
        reader_cache_size: int = 16,
        filter_name: Optional[str] = None,
    ):
        self.split_config = split_config
        self.split_part = split_part
        self.reader_cache_size = reader_cache_size
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
        self.jsonl_paths = [shard.path for shard in self.shards]

    def _validate_manifest_meta(self, meta: ShardListMeta) -> None:
        if meta.sample_excludes:
            raise ValueError("Prepared JSONL shard datasets do not support sample-level excludes")
        for shard in meta.shards:
            actual_count = IJsonlIndexReader.count_samples(shard.path)
            assert shard.count == actual_count, (
                f"JSONL shard count mismatch for {shard.path}: "
                f"metadata={shard.count}, index={actual_count}"
            )
            assert IJsonlIndexReader.is_current(shard.path), (
                "The index of the jsonl file does not match the file. Regenerate the index: "
                f"{shard.path}"
            )

    def _build_reader(
        self,
        *,
        parallel_shard_iters: int,
        part_filter: Callable[[str], bool] | None,
    ):
        return MultiJsonlReader(
            self.path,
            self.jsonl_paths,
            index_cache_size=parallel_shard_iters,
            reader_cache_size=self.reader_cache_size,
        )

    def _load_fn(
        self, part_filter: Callable[[str], bool] | None
    ) -> Callable[[RawSampleData], CrudeSample]:
        if (part_filter is not None and not part_filter("json")) or (
            self.part_filter is not None and not self.part_filter("json")
        ):

            def load_fn(sample: RawSampleData) -> CrudeSample:
                assert sample.data[0] is not None
                sample.data[0].pop("json", None)
                return self._load_sample_raw(sample)

            return load_fn
        return self._load_sample_raw

    def as_file_store(self) -> FileStore:
        return MultiJsonlFileStore(
            self.path,
            self.jsonl_paths,
            reader_cache_size=self.reader_cache_size,
        )

    def load_sample(self, sample: SampleRecord) -> CrudeSample:
        return CrudeSample(sample)

    def config(self) -> Dict[str, Any]:
        return dict(
            type=type(self).__qualname__,
            training=self.training,
            _path=str(self.path),
            jsonl_shard_count=len(self.shards),
            count=len(self),
            split_config=self.split_config if isinstance(self.split_config, str) else "<inline>",
            split_part=self.split_part,
            shuffle_over_epochs=self.shuffle_over_epochs,
            parallel_shard_iters=self.parallel_shard_iters,
            max_samples_per_sequence=self.max_samples_per_sequence,
            subset=self.subset.config() if self.subset is not None else None,
            reader_cache_size=self.reader_cache_size,
            filter_name=self.filter_name,
        )

    def __str__(self):
        return f"{type(self).__name__}(path={self.path}, shards={len(self.shards)})"


@register_dataset_factory_provider(priority=PRIORITY_SINGLE_FILE)
class DefaultCrudeJsonlDatasetFactory(CrudeJsonlDatasetFactory):
    """Adds tags to the sample and loads the JSON payload."""

    @classmethod
    def detect_path(cls, path: EPath) -> EnergonDatasetType | None:
        if path.name.endswith(".jsonl") and path.is_file():
            return EnergonDatasetType.JSONL
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
    ) -> "DefaultCrudeJsonlDatasetFactory":
        assert sample_type is CrudeSample or sample_type is None, (
            f"Sample type must be CrudeSample for jsonl datasets, but got {sample_type}"
        )
        assert dataset_config is None, (
            f"Dataset config must be None for jsonl datasets, but got {dataset_config}"
        )
        assert split_config is None, (
            f"Split config must be None for jsonl datasets, but got {split_config}"
        )
        kwargs.pop("split_part", None)
        return cls(
            path,
            **kwargs,
        )

    def __init__(self, path: EPath, *, tags: Optional[Dict[str, Any]] = None, **kwargs):
        if "decoder" in kwargs:
            del kwargs["decoder"]
        super().__init__(path, **kwargs)
        self.tags = tags

    def load_sample(self, sample: SampleRecord) -> CrudeSample:
        sample["__tags__"] = self.tags
        if "json" in sample:
            sample["json"] = json.loads(sample["json"])
        return super().load_sample(sample)

    def config(self) -> Dict[str, Any]:
        return dict(
            **super().config(),
            tags=self.tags,
        )


class DefaultCrudeJsonlShardListDatasetFactory(CrudeJsonlShardListDatasetFactory):
    """Adds tags to samples and loads JSON for prepared JSONL shard datasets."""

    def __init__(self, path: EPath, *, tags: Optional[Dict[str, Any]] = None, **kwargs):
        if "decoder" in kwargs:
            del kwargs["decoder"]
        super().__init__(path, **kwargs)
        self.tags = tags

    def load_sample(self, sample: SampleRecord) -> CrudeSample:
        sample["__tags__"] = self.tags
        if "json" in sample:
            sample["json"] = json.loads(sample["json"])
        return super().load_sample(sample)

    def config(self) -> Dict[str, Any]:
        return dict(
            **super().config(),
            tags=self.tags,
        )
