# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import (
    BaseCoreDatasetFactory,
    Sample,
    SavableDataset,
    set_sample_restore_key,
)
from megatron.energon.flavors.webdataset.base_webdataset import BaseWebdatasetFactory
from megatron.energon.flavors.webdataset.error_handler import ErrorHandler
from megatron.energon.flavors.webdataset.sample_loader import WebdatasetSampleLoaderDataset
from megatron.energon.flavors.webdataset.sharder import Sharder
from megatron.energon.flavors.webdataset.structs import FilteredSample, ShardInfo, reraise_exception
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
    shuffle_over_epochs: int = 1
    parallel_shard_iters: Optional[int]
    max_samples_per_sequence: Optional[int]
    part_filter: Optional[Callable[[str], bool]]
    join_method: Literal["inner_match"]
    handler: Callable[[Exception, Optional[str]], None]

    shards: List[Sequence[ShardInfo]]
    part_datasets: SavableDataset[T_sample]

    inner_datasets: List[BaseWebdatasetFactory]
    inner_dataset_keys: Optional[List[str]]
    _sample_joiner: Callable[..., T_sample]

    def __init__(
        self,
        inner_datasets: Union[List[BaseWebdatasetFactory], Dict[str, BaseWebdatasetFactory]],
        *,
        training: bool,
        worker_config: WorkerConfig,
        shuffle_over_epochs: int = 1,
        parallel_shard_iters: Optional[int] = None,
        max_samples_per_sequence: Optional[int] = None,
        part_filter: Optional[Callable[[str], bool]] = None,
        join_method: Literal["inner_match"] = "inner_match",
        joiner: Union[Type[T_sample], Callable[..., T_sample]],
        handler: Callable[[Exception, Optional[str]], None] = reraise_exception,
    ):
        """
        Constructs the loader for a joined webdataset. The samples from the inner datasets are joined into a single
        sample using the joiner function.

        Args:
            inner_dataset: The inner datasets. Must be loaded internally with `_is_composed=True`.
                Either a list (*args for joiner) or a dict (**kwargs for joiner) of datasets,
                where the samples will be passed to the joiner function as *args or **kwargs.
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
            part_filter: (internal) Function for filtering tar files by dict keys
            join_method: How to join the samples from the datasets.
                inner_match: All samples must match 1:1 of the merged datasets.
                This might be extended to further modes in the future, but those will require a new index, which
            joiner: Type of the joined samples or a method for joining the samples.
            handler: Exception handler. Args: (exception, key).
        """
        self.__sample_type__ = joiner
        assert all(
            not hasattr(d, "dataset") for d in inner_datasets
        ), "Inner dataset was not instantiated with _is_composed=True"
        if isinstance(joiner, type) and issubclass(joiner, Sample):
            joiner = joiner.from_joined
        else:
            assert callable(joiner), f"Joiner {joiner} must be a callable or a Sample subclass"
        if isinstance(inner_datasets, dict):
            inner_keys = list(inner_datasets.keys())
            self.inner_dataset_keys = inner_keys
            # Wrap the joiner to pass the samples as kwargs
            self._sample_joiner = lambda *samples: joiner(**dict(zip(inner_keys, samples)))
            inner_datasets = list(inner_datasets.values())
        else:
            self._sample_joiner = joiner
            self.inner_dataset_keys = None
        assert all(
            len(dataset.shards) == len(inner_datasets[0].shards) for dataset in inner_datasets[1:]
        ), f"Dataset structures do not match, shards differ"
        self.sample_exclude = inner_datasets[0].sample_excludes
        assert all(
            self.sample_exclude == dataset.sample_excludes for dataset in inner_datasets[1:]
        ), f"Sample excludes must be the same for all paths"

        if join_method == "inner_match":
            assert all(
                shard1.count == shard2.count
                for dataset in inner_datasets[1:]
                for shard1, shard2 in zip(dataset.shards, inner_datasets[0].shards)
            ), "When joining datasets with the 'inner_match' method, all shards must have the same count"
        else:
            assert False, f"Invalid join method {join_method}"

        self.inner_datasets = inner_datasets
        self.shards = list(zip(*(dataset.shards for dataset in self.inner_datasets)))
        self.training = training
        self.worker_config = worker_config
        self.shuffle_over_epochs = shuffle_over_epochs
        self.parallel_shard_iters = parallel_shard_iters
        self.max_samples_per_sequence = max_samples_per_sequence
        self.part_filter = part_filter
        self.join_method = join_method
        self.handler = handler

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

        rank_shards = self.shard_workers(
            self.shards,
            self.worker_config,
            max_samples_per_sequence=self.max_samples_per_sequence,
            rotation_offset=worker_rotation_offset,
        )

        for rank_idx, shards in enumerate(rank_shards):
            shards_text = ", ".join(
                f"{subshard[0].name}[{subshard[0].offset}, {subshard[0].offset+subshard[0].count})"
                for subshard in shards[:3]
            )
            if len(shards) > 6:
                shards_text += f", ...<{len(shards) - 6}>, " + ", ".join(
                    f"{subshards[0].name}[{subshards[0].offset}, {subshards[0].offset+subshards[0].count})"
                    for subshards in shards[-3:]
                )
            elif len(shards) > 3:
                shards_text += ", " + ", ".join(
                    f"{subshards[0].name}[{subshards[0].offset}, {subshards[0].offset+subshards[0].count})"
                    for subshards in shards[3:]
                )
            print(
                f"rank={self.worker_config.rank}, worker={rank_idx}: shard_range="
                f"[{shards_text}] "
                f"sum(count)={sum(subshards[0].count for subshards in shards)}"
            )

        dataset = WebdatasetSampleLoaderDataset(
            rank_shards=rank_shards,
            worker_config=self.worker_config,
            part_filter=self.part_filter,
            exclude=self.sample_exclude,
            shuffle_over_epochs=self.shuffle_over_epochs if self.training else None,
            parallel_shard_iters=parallel_shard_iters,
            dataset_join_method=self.join_method,
            handler=self.sample_error_handler,
        )
        return self._process_samples(dataset)

    @property
    def paths(self) -> List[EPath]:
        return [dataset.path for dataset in self.inner_datasets]

    def _process_samples(
        self, dataset: SavableDataset[Tuple[Optional[FilteredSample], ...]]
    ) -> SavableDataset[T_sample]:
        """Internally loads the sample."""
        return MapDataset(
            dataset,
            self.load_sample,
            error_handler=self.error_handler,
            stateless_map_fn=True,
            map_fn_config=self.config,
            worker_config=self.worker_config,
        )

    def load_sample(self, samples: Tuple[Optional[FilteredSample], ...]) -> T_sample:
        assert len(samples) > 0 and samples[0] is not None, "Always need primary sample"
        # Combine the restore key. This must be in accordance to the ShardReader's restore unpacking
        restore_key = [
            *samples[0]["__restore_key__"],
        ]
        for sample in samples[1:]:
            if sample is None:
                restore_key.append("")
                restore_key.append(-1)
            else:
                restore_key.extend(sample["__restore_key__"][1:3])

        # First call the loaders of all inner datasets
        loaded_samples = tuple(
            None if sample is None else dataset.load_sample(sample)
            for dataset, sample in zip(self.inner_datasets, samples)
        )
        # Then combine the loaded smaples into the final type
        return set_sample_restore_key(
            self._sample_joiner(*loaded_samples),
            *restore_key,
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
