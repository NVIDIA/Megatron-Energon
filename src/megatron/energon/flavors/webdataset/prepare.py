# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import json
import logging
import random
import re
import tarfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import braceexpand
import yaml

from megatron.energon import __version__
from megatron.energon.bracecollapse import collapse
from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.eyaml import load_yaml
from megatron.energon.flavors.webdataset.aggregator_pool import AggregatorPool, BaseAggregator
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME, skip_meta_re, split_name_re
from megatron.energon.flavors.webdataset.indexing import SqliteIndexWriter
from megatron.energon.flavors.webdataset.itar import TarIndexWriter
from megatron.energon.flavors.webdataset.structs import ShardInfo, WebdatasetInfo, WebdatasetSplits
from megatron.energon.typed_converter import to_json_object

logger = logging.getLogger(__name__)

T = TypeVar("T", covariant=True)


@edataclass
class IndexAggregatable:
    """
    A base class for all objects that can be returned/yielded by `_preprocess_tar` and
    received by `SqliteIndexWriterAggregator.on_item`.
    """

    ...


@edataclass
class IndexSample(IndexAggregatable):
    tar_file_id: int
    sample_key: str
    sample_index: int
    byte_offset: int
    byte_size: int


@edataclass
class IndexSamplePart(IndexAggregatable):
    tar_file_id: int
    sample_index: int
    part_name: str
    content_byte_offset: int
    content_byte_size: int


@edataclass
class IndexShardInfo(IndexAggregatable):
    shard_info: ShardInfo
    parts: Set[str]


class SqliteIndexWriterAggregator(
    BaseAggregator[
        Tuple[ShardInfo, Set[str]], Tuple[List[ShardInfo], Set[str], bool, List[Tuple[str, int]]]
    ]
):
    sqlite_path: EPath
    total_tasks: int
    progress_fn: Optional[Callable]
    writer: Optional[SqliteIndexWriter]
    had_update: bool
    shards: List[ShardInfo]
    found_parts: Set[str]
    prog_iter: Iterator

    def __init__(
        self,
        sqlite_path: EPath,
        total_tasks: int,
        progress_fn: Optional[Callable[[Iterator[Any], int], Iterator[T]]] = None,
    ):
        self.sqlite_path = sqlite_path
        self.total_tasks = total_tasks
        self.writer = None
        self.had_update = False
        self.shards = []
        self.found_parts = set()

        if progress_fn is not None:
            self.prog_iter = progress_fn(iter(range(self.total_tasks)), self.total_tasks)
        else:
            self.prog_iter = iter(range(self.total_tasks))

    def on_start(self, aggregator_pool: AggregatorPool) -> None:
        self.writer = SqliteIndexWriter(self.sqlite_path)

    def on_item(
        self,
        item: IndexAggregatable,
        aggregator_pool: AggregatorPool,
    ) -> None:
        assert self.writer is not None, "Writer is not initialized."
        if isinstance(item, IndexSample):
            self.writer.append_sample(**asdict(item))
            self.had_update = True
        elif isinstance(item, IndexSamplePart):
            self.writer.append_part(**asdict(item))
        elif isinstance(item, IndexShardInfo):
            # This is a (shard_info, parts) tuple
            next(self.prog_iter)

            shard_info, cur_parts = item.shard_info, item.parts
            assert shard_info.count != 0, f"Shard {shard_info.name} has no samples."
            self.shards.append(shard_info)
            if len(self.found_parts) < 50:
                self.found_parts.update(cur_parts)

    def on_finish(self, aggregator_pool: AggregatorPool) -> None:
        assert self.writer is not None, "Writer is not initialized."
        self.writer.close()

    def get_final_result_data(
        self,
    ) -> Tuple[List[ShardInfo], Set[str], bool, List[Tuple[str, int]]]:
        assert self.writer is not None, "Writer is not initialized."
        return self.shards, self.found_parts, self.had_update, self.writer.duplicates


class WebdatasetPreparator:
    @staticmethod
    def _preprocess_tar(
        path: str,
        shard_to_idx: Dict[str, int],
        parent_path: EPath,
        max_parts: int,
    ) -> Generator[IndexAggregatable, None, None]:
        """Process a single tar file, i.e. read the tarinfos, generate the tar index and return
        stats.
        This method is passed to the `user_produce_data` argument of AggregatorPool.

        Args:
            path: Path to the tar file.
            shard_to_idx: Mapping from shard path to its index
            parent_path: Root path of the dataset.
            max_parts: Maximum number of different parts to return

        Returns:
            A generator of items that will be processed by SqliteIndexWriterAggregator.
            See method `on_item` of SqliteIndexWriterAggregator.
            The items are either:
            - A sample dictionary with information about the offset, key etc.
            - Or a tuple of shard info and a set of found parts for statistics.
        """
        shard_info = ShardInfo(name=path, path=parent_path / path, count=0)

        try:
            # Note: Write to .tmp file first, then remove .tmp extension, to make sure only complete
            # files are used.
            tar: tarfile.TarFile
            with shard_info.path.open("rb") as f:
                with (
                    tarfile.open(fileobj=f, mode="r:*") as tar,
                    TarIndexWriter(shard_info.path) as iw,
                ):
                    count = 0

                    # The parts set is used to collect various file endings that are
                    # available in the dataset. This is used for the interactive prepare wizard.
                    parts = set()

                    last_base_name = None
                    member: tarfile.TarInfo

                    next_index_sample = None

                    for member in tar:
                        if not member.isreg():
                            continue
                        if member.name is None:
                            continue
                        if skip_meta_re.match(member.name):
                            continue

                        name_match = split_name_re.match(member.name)
                        if name_match is None:
                            continue

                        base_name = name_match.group(1)
                        if len(parts) < max_parts:
                            parts.add(name_match.group(2))

                        if last_base_name != base_name:
                            iw.append(member.offset)

                            if next_index_sample is not None:
                                next_index_sample["byte_size"] = (
                                    member.offset - next_index_sample["byte_offset"]
                                )
                                yield IndexSample(**next_index_sample)

                            next_index_sample = dict(
                                tar_file_id=shard_to_idx[path],
                                sample_key=base_name,
                                sample_index=count,
                                byte_offset=member.offset,
                            )
                            last_base_name = base_name
                            count += 1

                        # Yield this part of the sample to the aggregator
                        yield IndexSamplePart(
                            tar_file_id=shard_to_idx[path],
                            sample_index=count - 1,
                            part_name=name_match.group(2),
                            content_byte_offset=member.offset_data,
                            content_byte_size=member.size,
                        )

                    shard_info.count = count
                    iw.append(tar.offset)
                    if next_index_sample is not None:
                        next_index_sample["byte_size"] = (
                            tar.offset - next_index_sample["byte_offset"]
                        )
                        yield IndexSample(**next_index_sample)
            yield IndexShardInfo(shard_info=shard_info, parts=parts)
            return
        except BaseException:
            logger.exception(f"Shard failed to load: {path!r}. Skipping it.")
            yield IndexShardInfo(shard_info=shard_info, parts=set())
            return

    @staticmethod
    def iter_dataset_content(
        path: Union[str, EPath],
        extract_keys: Container[str] = (),
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yield example dataset content for a few samples.

        Args:
            path: Path to the tar file.
        """
        path = EPath(path)
        with path.open("rb") as f:
            tar: tarfile.TarFile
            with tarfile.open(fileobj=f, mode="r:*") as tar:
                last_base_name = None
                sample = {}
                member: tarfile.TarInfo
                for member in tar:
                    if not member.isreg():
                        continue
                    if member.name is None:
                        continue
                    if skip_meta_re.match(member.name):
                        continue

                    name_match = split_name_re.match(member.name)
                    if name_match is None:
                        continue

                    base_name = name_match.group(1)
                    if last_base_name != base_name:
                        if sample:
                            yield sample
                        sample = {}
                        last_base_name = base_name
                    if name_match:
                        if name_match.group(2) in extract_keys:
                            sample[name_match.group(2)] = tar.extractfile(member).read()
                        else:
                            sample[name_match.group(2)] = None
                if sample:
                    yield sample

    @classmethod
    def prepare_dataset(
        cls,
        parent_path: Union[Path, EPath],
        paths: List[str],
        *,
        split_parts_ratio: Optional[List[Tuple[str, float]]] = None,
        split_parts_patterns: Optional[List[Tuple[str, str]]] = None,
        split_config: str = "split.yaml",
        shuffle_seed: Optional[int] = 42,
        progress_fn: Callable[[Iterator[Any], int], Iterator[T]] = (lambda x, y: x),
        workers: int = 32,
        tar_index_only: bool = False,
    ) -> Tuple[Set[str], List[Tuple[str, int]]]:
        """
        Preprocess the shards and write the split config. Preprocessing is done in parallel.
        Counts the number of samples in each shard.

        Args:
            parent_path: Common parent path for the shards
            paths: Paths to the shards
            split_parts_ratio: Names of splits and their ratio (will be normalized)
            split_parts_patterns: Names of splits and their path patterns
            split_config: Filename for the split config (`parent_path / '.nv-meta' / split_config`), may be yaml or json
            shuffle_seed: Seed for shuffling shards before splitting into split_parts. None to
                disable.
            progress_fn: Callback for progress bar
            workers: Number of parallel workers for reading each shard
            tar_index_only: Only create tar-index, then exit

        Returns:
            The set of all parts found in the shards. But at most 50.
        """
        parent_path = EPath(parent_path)

        paths = [path for path in paths for path in braceexpand.braceexpand(path)]

        # Construct a mapping from relative shard path to its index
        shard_to_idx = {path: idx for idx, path in enumerate(paths)}

        (parent_path / MAIN_FOLDER_NAME).mkdir(exist_ok=True)

        aggregator = SqliteIndexWriterAggregator(
            parent_path / MAIN_FOLDER_NAME / "index.sqlite",
            total_tasks=len(paths),
            progress_fn=progress_fn,
        )

        process_tar = functools.partial(
            cls._preprocess_tar,
            shard_to_idx=shard_to_idx,
            parent_path=parent_path,
            max_parts=50,
        )

        pool = AggregatorPool(
            num_workers=workers,
            user_produce_data=process_tar,
            aggregator=aggregator,
        )

        for path in paths:
            pool.submit_task(path)

        shards, found_parts, had_update, duplicates = pool.process()

        if had_update:
            logger.info("Regenerating dataset UUID...")
            with (parent_path / MAIN_FOLDER_NAME / "index.uuid").open("w") as f:
                f.write(str(uuid.uuid4()))

        json_info_config = parent_path / MAIN_FOLDER_NAME / ".info.json"
        yaml_info_config = parent_path / MAIN_FOLDER_NAME / ".info.yaml"

        if tar_index_only:
            if yaml_info_config.is_file() and not json_info_config.is_file():
                # Convert legacy .info.yaml to .info.json
                with json_info_config.open("w") as f:
                    json.dump(load_yaml(yaml_info_config.read_bytes()), f, indent=2)

            return found_parts, duplicates

        assert len(shards) == len(shard_to_idx), (
            f"Lengths of shards and shard_to_idx do not match: {len(shards)} != {len(shard_to_idx)}"
        )

        # Sort the shards according to the order in the input list
        shards.sort(key=lambda shard: shard_to_idx[shard.name])

        # Save info
        assert [shard.name for shard in shards] == list(shard_to_idx.keys()), (
            "Shards are not in the same order as in the input list."
        )

        info = WebdatasetInfo(
            energon_version=__version__,
            shard_counts={shard.name: shard.count for shard in shards},
        )
        print(f"Saving info to {json_info_config}")

        with json_info_config.open("w") as wf:
            json.dump(to_json_object(info), wf, indent=2)

        if yaml_info_config.is_file():
            # If a .info.yaml existed previously, let's also update it
            # to keep them in sync
            with yaml_info_config.open("w") as wf:
                yaml.dump(to_json_object(info), wf)

        if split_parts_ratio is not None:
            # Normalize ratio
            total_ratio = sum(split_ratio for _, split_ratio in split_parts_ratio)
            split_parts_ratio = [
                (split_part, split_ratio / total_ratio)
                for split_part, split_ratio in split_parts_ratio
            ]
            # Sample from shards based on the split ratio from split parts
            split_shards = {}
            if shuffle_seed is not None:
                random.Random(shuffle_seed).shuffle(shards)
            split_total = 0
            split_offset = 0
            for split_part, split_ratio in split_parts_ratio:
                split_total += split_ratio
                split_end = int(len(shards) * split_total)
                split_shards[split_part] = [shard.name for shard in shards[split_offset:split_end]]
                split_offset = split_end
        else:
            assert split_parts_patterns is not None, (
                "Require either split_parts_ratio or split_parts_patterns"
            )
            # Sample from shards based on the split patterns from split parts
            split_shards = {}
            for split_part, split_pattern in split_parts_patterns:
                patterns = [
                    re.compile(pattern) for pattern in braceexpand.braceexpand(split_pattern)
                ]
                split_shards[split_part] = [
                    shard.name
                    for shard in shards
                    if any(pattern.match(shard.name) for pattern in patterns)
                ]

        # Optimize the split parts by trying to bracecollapse the shard names
        print("Collapsing split parts... ", flush=True, end="")
        for split_part in split_shards:
            split_shards[split_part] = collapse(split_shards[split_part], keep_order=True)
        print("Done", flush=True)

        # Save split config
        splits_config = WebdatasetSplits(split_parts=split_shards)
        with (parent_path / MAIN_FOLDER_NAME / split_config).open("w") as wf:
            if split_config.endswith(".yaml"):
                yaml.dump(to_json_object(splits_config), wf, sort_keys=False)
            elif split_config.endswith(".json"):
                json.dump(to_json_object(splits_config), wf, indent=2)
            else:
                raise ValueError(f"Invalid split config extension: {split_config}")

        return found_parts, duplicates
