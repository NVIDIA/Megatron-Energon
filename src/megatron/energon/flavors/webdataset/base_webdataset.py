# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import (
    BaseCoreDataset,
    MergedState,
    SavableDataset,
    State,
)
from megatron.energon.flavors.webdataset.error_handler import ErrorHandler
from megatron.energon.flavors.webdataset.metadata import WebdatasetMeta
from megatron.energon.flavors.webdataset.prepare import WebdatasetPreparator
from megatron.energon.flavors.webdataset.sample_loader import WebdatasetSampleLoaderDataset
from megatron.energon.flavors.webdataset.sharder import Sharder
from megatron.energon.flavors.webdataset.structs import FilteredSample, ShardInfo, reraise_exception
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.map_dataset import MapDataset

T_sample = TypeVar("T_sample", covariant=True)
T = TypeVar("T", covariant=True)

logger = logging.getLogger(__name__)


@dataclass
class WebdatasetState(State):
    dataset_state: State


@dataclass
class WebdatasetMergedState(MergedState):
    dataset_state: MergedState


class BaseWebdataset(
    BaseCoreDataset[T_sample], WebdatasetPreparator, Sharder, ErrorHandler, Generic[T_sample], ABC
):
    """
    Base class for all webdataset loaders. Applies proper sharding across workers.
    """

    path: EPath

    training: bool
    worker_config: WorkerConfig

    shards: List[ShardInfo]
    dataset: SavableDataset[T_sample]
    rank_shards: List[List[Sequence[ShardInfo]]]

    def __init__(
        self,
        path: EPath,
        *,
        split_part: str,
        training: bool,
        worker_config: WorkerConfig,
        shuffle_over_epochs: int = 1,
        parallel_shard_iters: Optional[int] = None,
        max_samples_per_sequence: Optional[int] = None,
        info_config: str = ".info.yaml",
        split_config: str = "split.yaml",
        part_filter: Optional[Callable[[str], bool]] = None,
        handler: Callable[[Exception, Optional[str]], None] = reraise_exception,
        _is_composed: bool = False,
    ):
        """
        Constructs the webdataset loader.

        Args:
            path: Path to the dataset.
            split_part: Which part to load (e.g. 'train', 'val', 'test').
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
            info_config: Config file to use for sample metadata.
            split_config: Config file to use for shard split definitions.
            part_filter: (internal) Function for filtering tar files by dict keys
            handler: Exception handler. Args: (exception, key).
            _is_composed: Internal flag, specifying if this is part of a merged dataset, thus
                should not load the dataset and shard the ranks.
        """
        assert self.__sample_type__ is not None, f"Class {type(self)} must define __sample_type__"
        wds_meta = WebdatasetMeta.from_config(
            path=path, split_part=split_part, info_config=info_config, split_config=split_config
        )
        self.path = path
        self.paths = [path]
        self.shards = wds_meta.shards
        self.training = training
        self.worker_config = worker_config
        self.handler = handler

        if not _is_composed:

            if parallel_shard_iters is None:
                if training:
                    # 16 seems to be a good choice since we don't want too many file handles open
                    parallel_shard_iters = 16
                else:
                    parallel_shard_iters = 1

            self.rank_shards = self.shard_workers(
                [(shard,) for shard in self.shards],
                self.worker_config,
                max_samples_per_sequence=max_samples_per_sequence,
            )

            self.rank_total = sum(
                subshard[0].count for shards in self.rank_shards for subshard in shards
            )
            for rank_idx, inner_shards in enumerate(self.rank_shards):
                shards_text = ", ".join(
                    f"{subshard[0].name}[{subshard[0].offset}, {subshard[0].offset+subshard[0].count})"
                    for subshard in inner_shards[:3]
                )
                if len(self.shards) > 6:
                    shards_text += f", ...<{len(self.shards) - 6}>, " + ", ".join(
                        f"{subshards[0].name}[{subshards[0].offset}, {subshards[0].offset+subshards[0].count})"
                        for subshards in inner_shards[-3:]
                    )
                elif len(self.shards) > 3:
                    shards_text += ", " + ", ".join(
                        f"{subshards[0].name}[{subshards[0].offset}, {subshards[0].offset+subshards[0].count})"
                        for subshards in inner_shards[3:]
                    )
                print(
                    f"rank={self.worker_config.rank}, worker={rank_idx}: shard_range="
                    f"[{shards_text}] "
                    f"sum(count)={sum(subshards[0].count for subshards in inner_shards)}"
                )

            dataset = WebdatasetSampleLoaderDataset(
                rank_shards=self.rank_shards,
                worker_config=self.worker_config,
                part_filter=part_filter,
                exclude=wds_meta.sample_excludes,
                loop=training,
                shuffle_over_epochs=shuffle_over_epochs if training else None,
                parallel_shard_iters=parallel_shard_iters,
                handler=self.sample_error_handler,
            )
            self.dataset = self._process_samples(dataset)
        else:
            self.sample_excludes = wds_meta.sample_excludes
            self.shuffle_over_epochs = shuffle_over_epochs
            self.parallel_shard_iters = parallel_shard_iters
            self.max_samples_per_sequence = max_samples_per_sequence
            self.part_filter = part_filter

    def _process_samples(
        self, dataset: SavableDataset[Tuple[Optional[FilteredSample], ...]]
    ) -> SavableDataset[T_sample]:
        """Internally loads the sample."""
        return MapDataset(
            dataset,
            self._load_sample_raw,
            error_handler=self.error_handler,
            stateless_map_fn=True,
            worker_config=self.worker_config,
        )

    def _load_sample_raw(self, sample: Tuple[Optional[FilteredSample], ...]) -> T_sample:
        # Just a wrapper for the inner tuple. Tuple should be of length 1.
        assert len(sample) == 1 and sample[0] is not None
        return self.load_sample(sample[0])

    @abstractmethod
    def load_sample(self, raw_data: FilteredSample) -> T_sample:
        """Loads the sample from the dataset."""
        ...

    def __len__(self):
        # In the training case, the result is an approximation (i.e. number of different samples)
        return self.rank_total

    def __iter__(self) -> Iterator[T_sample]:
        yield from self.dataset

    def worker_has_samples(self) -> bool:
        return self.dataset.worker_has_samples()

    def save_state(self) -> WebdatasetState:
        return WebdatasetState(
            dataset_state=self.dataset.save_state(),
        )

    def merge_states(self, states: List[WebdatasetState]) -> WebdatasetMergedState:
        assert all(s is None or isinstance(s, WebdatasetState) for s in states)
        return WebdatasetMergedState(
            dataset_state=self.dataset.merge_states(
                [None if s is None else s.dataset_state for s in states]
            ),
        )

    def restore_state(self, state: Optional[WebdatasetMergedState]) -> None:
        if state is None:
            self.dataset.restore_state(None)
        else:
            assert isinstance(state, WebdatasetMergedState)
            self.dataset.restore_state(state.dataset_state)

    def can_restore_sample(self) -> bool:
        return self.dataset.can_restore_sample()

    def assert_can_restore(self):
        self.dataset.assert_can_restore()

    def restore_sample(self, key: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        return self.dataset.restore_sample(key)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "training": self.training,
            "path": str(self.path),
            "worker_config": self.worker_config.config(),
            "dataset": self.dataset.config(),
        }

    def __str__(self):
        return f"{type(self).__name__}(path={self.path}, dataset={self.dataset})"
