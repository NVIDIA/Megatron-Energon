# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from bisect import bisect_right
from typing import Callable, Dict, List, Optional, Tuple, Union, overload

from megatron.energon.dataclass_slots import dataclass_slots
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.config import skip_meta_re, split_name_re
from megatron.energon.flavors.webdataset.indexing import JoinIndexReader
from megatron.energon.flavors.webdataset.itar import CachedItarOffsetReader, ITarFile
from megatron.energon.flavors.webdataset.structs import FilteredSample, ShardInfo


@dataclass_slots
class ITarSamplePointer:
    """
    Points to a sample inside some tar file on disk.
    The tar_file_id refers to the tar_filenames in the reader.
    """

    tar_file_id: int
    byte_offset: int
    byte_size: int


class ITarReader(ABC):
    """
    An abstract base class for reading a sequence of tar files containing samples.

    The common usage patterns and random-access interfaces are provided here.

    Args:
        base_path: The base path for the relative tar_filepaths.
        tar_filenames: The list of tar file names.
        tar_filepaths: The corresponding list of full paths to the tar files.
        part_filter: An optional filter function to select parts of the samples.
        itar_cache_size: The number of tar readers to keep open at the same time.
    """

    base_path: EPath
    tar_filenames: List[str]
    tar_filepaths: List[EPath]
    part_filter: Optional[Callable[[str], bool]]
    itar_files_cache: Dict[int, ITarFile]
    sample_filter: Optional[Callable[[str], bool]]

    def __init__(
        self,
        base_path: EPath,
        tar_filenames: List[str],
        tar_filepaths: List[EPath],
        part_filter: Optional[Callable[[str], bool]] = None,
        itar_cache_size: int = 5,
        sample_filter: Optional[Callable[[str], bool]] = None,
    ):
        assert len(tar_filenames) == len(tar_filepaths), (
            f"tar_filenames length ({len(tar_filenames)}) does not match "
            f"tar_filepaths length ({len(tar_filepaths)})"
        )
        self.base_path = base_path
        self.tar_filenames = tar_filenames
        self.tar_filepaths = tar_filepaths
        self.part_filter = part_filter
        self.itar_files_cache = {}
        self.itar_cache_size = itar_cache_size
        self.sample_filter = sample_filter

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of samples in the reader."""
        raise NotImplementedError

    @abstractmethod
    def _get_itar_sample_pointer(self, idx: int) -> ITarSamplePointer:
        """Get the ITarSample object for the given index."""
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """
        Must return a descriptive string of the concrete reader.
        """
        raise NotImplementedError

    def _get_itarfile_cached(self, tar_file_id: int) -> ITarFile:
        """
        Get the ITarFile object for the given tar file id.
        If the file is not already open, open it. If we exceed
        the global cache limit, close the least recently used file.
        """
        if tar_file_id not in self.itar_files_cache:
            file_object = self.tar_filepaths[tar_file_id].open(mode="rb")
            tar_file = ITarFile.open(fileobj=file_object, mode="r:")
            self.itar_files_cache[tar_file_id] = tar_file

        # If we hit the limit of open files, close the least recently used file
        while len(self.itar_files_cache) > self.itar_cache_size:
            # Get the oldest file
            lru_key = next(iter(self.itar_files_cache))

            self.itar_files_cache[lru_key].fileobj.close()
            self.itar_files_cache[lru_key].close()
            del self.itar_files_cache[lru_key]

        return self.itar_files_cache[tar_file_id]

    @overload
    def __getitem__(self, key: int) -> Optional[FilteredSample]: ...

    @overload
    def __getitem__(self, key: slice) -> "ITarReader": ...

    def __getitem__(self, key: Union[slice, int]) -> Union["ITarReader", FilteredSample, None]:
        """
        Get a sample from the dataset or slice it.
        """

        if isinstance(key, slice):
            # Return a new reader with a sliced samples tensor
            raise NotImplementedError("Slicing is not yet implemented")
        elif isinstance(key, int):
            idx = key
        else:
            raise TypeError("Invalid argument type for __getitem__")

        sample = self._get_itar_sample_pointer(idx)

        # Open the tar file (cached)
        tar_file = self._get_itarfile_cached(sample.tar_file_id)
        shard_name = self.tar_filenames[sample.tar_file_id]
        sample_base_name = None
        sample_name = None
        group_parts: Dict[str, bytes] = {}

        # Position the tar file at the correct offset
        tar_file.offset = sample.byte_offset

        while tar_file.offset < sample.byte_offset + sample.byte_size:
            tarinfo = tar_file.next()
            if tarinfo is None:
                raise ValueError(
                    f"Unexpected end of tar file: {self.tar_filenames[sample.tar_file_id]}"
                )
            fname = tarinfo.name
            if not tarinfo.isfile() or fname is None:
                continue
            if skip_meta_re.match(fname):
                continue

            # Extract the base_name and extension
            m = split_name_re.match(fname)
            if not m:
                continue
            cur_base_name, cur_ext = m.groups()

            if sample_base_name is None:
                sample_base_name = cur_base_name
                sample_name = f"{shard_name}/{cur_base_name}"
                if self.sample_filter is not None and not self.sample_filter(sample_name):
                    return None
            else:
                if sample_base_name != cur_base_name:
                    raise ValueError(
                        f"Inconsistent sample base name: {sample_base_name} vs {cur_base_name}"
                    )

            if self.part_filter is None or self.part_filter(cur_ext):
                member_bytes = tar_file.extractfile(tarinfo).read()
                group_parts[cur_ext] = member_bytes

        if sample_base_name is None:
            raise ValueError(f"No valid files found in sample {idx}")

        return FilteredSample(
            __key__=f"{shard_name}/{sample_base_name}",
            __shard__=self.tar_filenames[sample.tar_file_id],
            __restore_key__=("Webdataset", idx),
            **group_parts,
        )


class JoinIndexFileITarReader(ITarReader):
    """
    A concrete ITarReader that reads samples from a join index file (via JoinIndexReader).
    """

    index_file: EPath
    column: int
    index_reader_cache: Dict[int, JoinIndexReader]
    index_reader_cache_size: int

    def __init__(
        self,
        index_file: EPath,
        column: int,
        tar_filenames: List[str],
        base_path: EPath,
        part_filter: Optional[Callable[[str], bool]] = None,
        itar_cache_size: int = 5,
        sample_filter: Optional[Callable[[str], bool]] = None,
    ):
        self.index_file = index_file
        self.column = column

        # Create the full path to each tar file
        tar_filepaths = [base_path / fn for fn in tar_filenames]

        self.index_reader_cache = {}
        self.index_reader_cache_size = itar_cache_size

        super().__init__(
            base_path=base_path,
            tar_filenames=tar_filenames,
            tar_filepaths=tar_filepaths,
            part_filter=part_filter,
            itar_cache_size=itar_cache_size,
            sample_filter=sample_filter,
        )

    def _get_join_index_reader_cached(self, samp_idx: int) -> JoinIndexReader:
        """
        Get the JoinIndexReader object for the given sample index, or create it if it doesn't exist.
        """

        if samp_idx not in self.index_reader_cache:
            index_reader = JoinIndexReader(self.index_file, column=self.column)
            self.index_reader_cache[samp_idx] = index_reader

        # If we hit the limit of open files, close the least recently used file
        while len(self.index_reader_cache) > self.index_reader_cache_size:
            # Get the oldest file
            lru_key = next(iter(self.index_reader_cache))

            self.index_reader_cache[lru_key].close()
            del self.index_reader_cache[lru_key]

        return self.index_reader_cache[samp_idx]

    def _get_itar_sample_pointer(self, samp_idx: int) -> ITarSamplePointer:
        """
        Get the ITarSample object for the given index.
        """
        index_reader = self._get_join_index_reader_cached(samp_idx)
        row = index_reader[samp_idx]

        # Update cache entry
        new_offset = index_reader.tell_row()
        del self.index_reader_cache[samp_idx]
        self.index_reader_cache[new_offset] = index_reader

        assert len(row) == 1
        shard_idx, byte_offset, byte_size = row[0]

        return ITarSamplePointer(
            tar_file_id=shard_idx,
            byte_offset=byte_offset,
            byte_size=byte_size,
        )

    def __len__(self) -> int:
        try:
            # Get any reader, they will all work
            index_reader = next(iter(self.index_reader_cache.values()))
        except StopIteration:
            # If there's no reader yet, we need to create one to get the length
            index_reader = self._get_join_index_reader_cached(0)

        return len(index_reader)

    def __str__(self) -> str:
        return (
            f"JoinIndexFileITarReader("
            f"len={len(self)}, base_path={self.base_path}, "
            f"len(shards)={len(self.tar_filenames)}, "
            f"shards=[{self.tar_filenames[0] if self.tar_filenames else 'N/A'}, ...])"
        )


class ShardInfosITarReader(ITarReader):
    """
    A concrete ITarReader that constructs its internal sample list from a list of ShardInfos.
    """

    shard_infos: List[ShardInfo]
    shard_tar_file_idxs: List[int]
    shard_count_cumsum: List[int]
    cached_offset_reader: CachedItarOffsetReader

    def __init__(
        self,
        base_path: EPath,
        shard_infos: List[ShardInfo],
        part_filter: Optional[Callable[[str], bool]] = None,
        itar_cache_size: int = 5,
        sample_filter: Optional[Callable[[str], bool]] = None,
    ):
        # Build the tar_filenames and tar_filepaths from shard_infos,
        # constructing the samples tensor as we go.
        cur_tar_files: Dict[str, Tuple[int, EPath]] = {}

        self.shard_infos = shard_infos

        # Compute the cumsum of the shard counts, so that we can look up
        # the shard index for a given sample index.
        # Get all tar files from the shard_infos

        self.shard_count_cumsum = [0]
        self.shard_tar_file_idxs = []
        sample_idx = 0
        for shardinfo in shard_infos:
            filepath = shardinfo.path
            filename = shardinfo.name

            if filename not in cur_tar_files:
                cur_tar_files[filename] = (len(cur_tar_files), filepath)

            sample_idx += shardinfo.count
            self.shard_count_cumsum.append(sample_idx)
            self.shard_tar_file_idxs.append(cur_tar_files[filename][0])

        tar_filenames = list(cur_tar_files.keys())
        tar_filepaths = [p[1] for p in cur_tar_files.values()]

        # Instantiate cached reader for the .tar.idx files
        self.cached_offset_reader = CachedItarOffsetReader(cache_size=itar_cache_size)

        super().__init__(
            base_path=base_path,
            tar_filenames=tar_filenames,
            tar_filepaths=tar_filepaths,
            part_filter=part_filter,
            itar_cache_size=itar_cache_size,
            sample_filter=sample_filter,
        )

    def _get_itar_sample_pointer(self, idx: int) -> ITarSamplePointer:
        """
        Get the ITarSample object for the given index.
        """

        # Find the shard index using binary search
        shard_idx = bisect_right(self.shard_count_cumsum, idx) - 1
        if shard_idx < 0 or shard_idx >= len(self.shard_infos):
            raise IndexError(f"Index out of bounds: {idx}")

        # Get the shard info for the given index
        shard = self.shard_infos[shard_idx]
        sample_idx_in_shard_file = idx - self.shard_count_cumsum[shard_idx]

        # Now we know the tar file and the sample offset in the file.
        # We need to figure out the byte offset and size of the sample,
        # by looking it up in the .tar.idx file.
        byte_offset, byte_size = self.cached_offset_reader.get_itar_byte_offset(
            shard.path, sample_idx_in_shard_file
        )

        return ITarSamplePointer(
            tar_file_id=self.shard_tar_file_idxs[shard_idx],
            byte_offset=byte_offset,
            byte_size=byte_size,
        )

    def __len__(self) -> int:
        return self.shard_count_cumsum[-1]

    def __str__(self) -> str:
        return (
            f"ShardInfosITarReader("
            f"len={len(self)}, base_path={self.base_path}, "
            f"len(shards)={len(self.tar_filenames)}, "
            f"shards=[{self.tar_filenames[0] if self.tar_filenames else 'N/A'}, ...])"
        )
