# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from megatron.energon.cache import FileStore
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import (
    BaseCoreDatasetFactory,
    Sample,
    SavableDataset,
    legacy_handler,
    set_sample_restore_key,
)
from megatron.energon.flavors.webdataset.base_webdataset import BaseWebdatasetFactory
from megatron.energon.flavors.webdataset.error_handler import ErrorHandler
from megatron.energon.flavors.webdataset.indexing import JoinIndexReader
from megatron.energon.flavors.webdataset.itar_reader import JoinIndexFileITarReader
from megatron.energon.flavors.webdataset.sample_loader import (
    RawSampleData,
    WebdatasetSampleLoaderDataset,
)
from megatron.energon.flavors.webdataset.sharder import Sharder
from megatron.energon.flavors.webdataset.structs import ShardInfo, reraise_exception
from megatron.energon.source_info import SourceInfo
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.map_dataset import MapDataset

T_sample = TypeVar("T_sample", covariant=True)


class JoinedWebdatasetFactory(
    BaseCoreDatasetFactory[T_sample], Sharder, ErrorHandler[T_sample], Generic[T_sample], ABC
):
    """
    Base class for all webdataset loaders. Applies proper sharding across workers. Can join multiple datasets.
    """

    training: bool
    worker_config: WorkerConfig
    shuffle_over_epochs: Optional[int] = 1
    parallel_shard_iters: Optional[int]
    max_samples_per_sequence: Optional[int]
    join_index: EPath
    handler: Callable[[Exception, Optional[str], Optional[list[SourceInfo]]], None]

    shards: List[Sequence[ShardInfo]]
    part_datasets: SavableDataset[T_sample]

    inner_datasets: List[BaseWebdatasetFactory]
    inner_dataset_keys: Optional[List[str]]
    _sample_joiner: Callable[..., T_sample]

    def __init__(
        self,
        inner_datasets: Union[Sequence[BaseWebdatasetFactory], Mapping[str, BaseWebdatasetFactory]],
        *,
        training: bool,
        worker_config: WorkerConfig,
        shuffle_over_epochs: Optional[int] = 1,
        parallel_shard_iters: Optional[int] = None,
        max_samples_per_sequence: Optional[int] = None,
        join_index: EPath,
        joiner: Union[Type[T_sample], Callable[..., T_sample]],
        handler: Callable[
            [Exception, Optional[str], Optional[list[SourceInfo]]], None
        ] = reraise_exception,
    ):
        """
        Constructs the loader for a joined webdataset. The samples from the inner datasets are joined into a single
        sample using the joiner function.

        Args:
            inner_dataset: The inner datasets. Must be loaded internally with `_is_composed=True`.
                Either a list (\\*args for joiner) or a dict (\\*\\*kwargs for joiner) of datasets,
                where the samples will be passed to the joiner function as \\*args or \\*\\*kwargs.
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
            join_index: Path to the join index file. Only required for join_method="left".
            joiner: Type of the joined samples or a method for joining the samples.
            handler: Exception handler. Args: (exception, key).
        """
        self.__sample_type__ = joiner
        assert all(not hasattr(d, "dataset") for d in inner_datasets), (
            "Inner dataset was not instantiated with _is_composed=True"
        )
        if isinstance(joiner, type) and issubclass(joiner, Sample):
            joiner = joiner.from_joined
        else:
            assert callable(joiner), f"Joiner {joiner} must be a callable or a Sample subclass"
        if isinstance(inner_datasets, Mapping):
            inner_keys = list(inner_datasets.keys())
            self.inner_dataset_keys = inner_keys
            # Wrap the joiner to pass the samples as kwargs
            self._sample_joiner = lambda *samples: joiner(**dict(zip(inner_keys, samples)))
            inner_datasets = list(inner_datasets.values())
        else:
            assert isinstance(inner_datasets, Sequence)
            self._sample_joiner = joiner
            self.inner_dataset_keys = None

        self.join_index = join_index
        self.inner_datasets = inner_datasets
        self.shards = list(zip(*(dataset.shards for dataset in self.inner_datasets)))
        self.training = training
        self.worker_config = worker_config
        self.shuffle_over_epochs = shuffle_over_epochs
        self.parallel_shard_iters = parallel_shard_iters
        self.max_samples_per_sequence = max_samples_per_sequence
        self.handler = legacy_handler(handler)

    def __len__(self) -> int:
        return sum(shard.count for shard in self.inner_datasets[0].shards)

    def build(self, worker_rotation_offset: int = 0) -> SavableDataset[T_sample]:
        if self.parallel_shard_iters is None:
            if self.training:
                # 16 seems to be a good choice since we don't want too many file handles open
                parallel_shard_iters = 16
            else:
                parallel_shard_iters = 1
        else:
            parallel_shard_iters = self.parallel_shard_iters

        # Get join index, get size, distribute samples
        # Get samples for each worker on current rank
        assert self.join_index.is_file(), (
            f"Join index {self.join_index} does not exist, did you prepare the metadataset? "
            "If you already prepared the metadataset, the join index might be outdated due to "
            "modifications to the inner datasets. In this case, you need to re-prepare the metadataset."
        )

        with JoinIndexReader(self.join_index) as jir:
            total_samples = len(jir)

        workers_sample_slice_offsets = self.slice_workers(
            total_samples,
            worker_config=self.worker_config,
            max_samples_per_sequence=self.max_samples_per_sequence,
            rotation_offset=worker_rotation_offset,
        )

        for worker_idx, sample_slice_offsets in enumerate(workers_sample_slice_offsets):
            start_idx = sample_slice_offsets[0]
            end_idx = sample_slice_offsets[-1]

            if len(sample_slice_offsets) > 6:
                offset_str = f"{', '.join(str(o) for o in sample_slice_offsets[:3])} ...<{len(sample_slice_offsets) - 6}> {', '.join(str(o) for o in sample_slice_offsets[-3:])}"
            else:
                offset_str = ", ".join(str(o) for o in sample_slice_offsets)

            print(
                f"rank={self.worker_config.rank}, worker={worker_idx}: sample_range=[{start_idx}, {end_idx}) in {len(sample_slice_offsets) - 1} slices, "
                f"sum(count)={end_idx - start_idx}: [{offset_str}]"
            )

        itar_readers = [
            JoinIndexFileITarReader(
                index_file=self.join_index,
                column=col_idx,
                tar_filenames=indexed_dataset.split_part_files,
                base_path=indexed_dataset.path,
                part_filter=indexed_dataset.part_filter,
                itar_cache_size=parallel_shard_iters,
            )
            for col_idx, indexed_dataset in enumerate(self.inner_datasets)
        ]

        dataset = WebdatasetSampleLoaderDataset(
            join_readers=itar_readers,
            workers_sample_slice_offsets=workers_sample_slice_offsets,
            worker_config=self.worker_config,
            shuffle_over_epochs=self.shuffle_over_epochs if self.training else None,
            parallel_slice_iters=parallel_shard_iters,
        )
        return self._process_samples(dataset)

    def as_file_store(self) -> FileStore:
        raise NotImplementedError("Not supported on joined datasets")

    @property
    def paths(self) -> List[EPath]:
        return [dataset.path for dataset in self.inner_datasets]

    def _process_samples(self, dataset: SavableDataset[RawSampleData]) -> SavableDataset[T_sample]:
        """Internally loads the sample."""
        return MapDataset(
            dataset,
            self.load_sample,
            error_handler=self.error_handler,
            stateless_map_fn=True,
            map_fn_config=self.config,
            worker_config=self.worker_config,
        )

    def load_sample(self, samples: RawSampleData) -> T_sample:
        assert len(samples.data) > 0 and samples.data[0] is not None, "Always need primary sample"
        # First call the loaders of all inner datasets
        loaded_samples = tuple(
            None if sample is None else dataset.load_sample(sample)
            for dataset, sample in zip(self.inner_datasets, samples.data)
        )
        # Then combine the loaded smaples into the final type
        return set_sample_restore_key(
            self._sample_joiner(*loaded_samples),
            *samples.__restore_key__,
            src=self,
            fail_otherwise=True,
        )

    def config(self) -> Dict[str, Any]:
        return dict(
            type=type(self).__qualname__,
            joined_datasets=[dataset.config() for dataset in self.inner_datasets],
            training=self.training,
            shuffle_over_epochs=self.shuffle_over_epochs,
            parallel_shard_iters=self.parallel_shard_iters,
            max_samples_per_sequence=self.max_samples_per_sequence,
        )

    def __str__(self):
        return f"{type(self).__name__}(paths={self.paths})"
