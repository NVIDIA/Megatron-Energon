# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import logging
import multiprocessing as mp
import random
import re
import tarfile
from pathlib import Path
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import braceexpand
import yaml

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME, skip_meta_re, split_name_re
from megatron.energon.flavors.webdataset.itar import TarIndexWriter
from megatron.energon.flavors.webdataset.structs import ShardInfo, WebdatasetInfo, WebdatasetSplits
from megatron.energon.typed_converter import to_json_object

logger = logging.getLogger(__name__)

T = TypeVar("T", covariant=True)


class WebdatasetPreparator:

    @staticmethod
    def _preprocess_tar(
        path: Union[str, EPath], parent_path: Union[str, EPath], max_parts: int
    ) -> Tuple[ShardInfo, Set[str]]:
        """Process a single tar file, i.e. read the tarinfos, generate the tar index and return
        stats.

        Args:
            path: Path to the tar file.
            parent_path: Root path of the dataset.
            max_parts: Maximum number of different parts to return

        Returns:
            Tuple of shard info and found keys of the loaded dicts.
        """
        EPath.prepare_forked_process()  # Multiproc with fork

        path = EPath(path)
        shard_info = ShardInfo(name=path.relpath, path=path, offset=0, count=0)

        if not shard_info.path.is_absolute():
            parent_path = EPath(parent_path)
            assert parent_path.is_absolute(), f"Parent path must be absolute: {parent_path}"
            shard_info.path = parent_path / path

        try:
            # Note: Write to .tmp file first, then remove .tmp extension, to make sure only complete
            # files are used.
            tar: tarfile.TarFile
            with shard_info.path.open("rb") as f:
                with tarfile.open(fileobj=f, mode="r:*") as tar, TarIndexWriter(
                    shard_info.path
                ) as iw:
                    count = 0
                    parts = set()
                    last_base_name = None
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
                        if len(parts) < max_parts:
                            parts.add(name_match.group(2))

                        if last_base_name != base_name:
                            iw.append(member.offset)
                            last_base_name = base_name
                            count += 1
                    shard_info.count = count
                    iw.append(tar.offset)
            return shard_info, parts
        except BaseException:
            logger.exception(f"Shard failed to load: {path!r}. Skipping it.")
            return shard_info, set()

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
        info_config: str = ".info.yaml",
        split_config: str = "split.yaml",
        shuffle_seed: Optional[int] = 42,
        progress_fn: Callable[[List[T], int], Iterable[T]] = (lambda x, l: x),
        workers: int = 32,
        tar_index_only: bool = False,
    ) -> Set[str]:
        """
        Preprocess the shards and write the split config. Preprocessing is done in parallel.
        Counts the number of samples in each shard.

        Args:
            parent_path: Common parent path for the shards
            paths: Paths to the shards
            split_parts_ratio: Names of splits and their ratio (will be normalized)
            split_parts_patterns: Names of splits and their path patterns
            info_config: Filename for the info config (`parent_path / '.nv-meta' / info_config`)
            split_config: Filename for the info config (`parent_path / '.nv-meta' / split_config`)
            shuffle_seed: Seed for shuffling shards before splitting into split_parts. None to
                disable.
            progress_fn: Callback for progress bar
            workers: Number of parallel workers for reading each shard
            tar_index_only: Only create tar-index, then exit

        Returns:
            The set of all parts found in the shards. But at most 50.
        """
        parent_path = EPath(parent_path).absolute()

        found_parts = set()
        paths = [path for path in paths for path in braceexpand.braceexpand(path)]
        shards: List[ShardInfo] = []

        assert parent_path.is_absolute(), f"Parent path must be absolute: {parent_path}"

        # use functools partial to pass parent_path to process_tar
        process_tar = functools.partial(
            cls._preprocess_tar,
            parent_path=parent_path.url,  # convert to url string, to avoid EPath in multiprocessing
            max_parts=50,
        )

        with mp.Pool(workers) as pool:
            shard_info: ShardInfo
            cur_parts: Set[str]
            for shard_info, cur_parts in progress_fn(pool.imap(process_tar, paths), len(paths)):
                if shard_info.count == 0:
                    # This shard failed to load. Skip it.
                    continue
                shards.append(shard_info)
                if len(found_parts) < 50:
                    found_parts.update(cur_parts)

        if tar_index_only:
            return found_parts

        (parent_path / MAIN_FOLDER_NAME).mkdir(exist_ok=True)

        # Save info
        info = WebdatasetInfo(
            shard_counts={shard.name: shard.count for shard in shards},
        )
        with (parent_path / MAIN_FOLDER_NAME / info_config).open("w") as wf:
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
            assert (
                split_parts_patterns is not None
            ), "Require either split_parts_ratio or split_parts_patterns"
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
        # Save split config
        splits_config = WebdatasetSplits(split_parts=split_shards)
        with (parent_path / MAIN_FOLDER_NAME / split_config).open("w") as wf:
            yaml.dump(to_json_object(splits_config), wf)

        return found_parts
