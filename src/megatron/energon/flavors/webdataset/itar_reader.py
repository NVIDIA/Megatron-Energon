# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from bisect import bisect_right
from typing import (
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.config import skip_meta_re, split_name_re
from megatron.energon.flavors.webdataset.indexing import JoinIndexReader, SqliteIndexReader
from megatron.energon.flavors.webdataset.itar import (
    CachedItarOffsetReader,
    ITarFile,
    ITarSamplePointer,
)
from megatron.energon.flavors.webdataset.metadata import get_info_shard_files
from megatron.energon.flavors.webdataset.structs import FilteredSample, ShardInfo
from megatron.energon.source_info import SourceInfo

T_index = TypeVar("T_index", covariant=False)


class ITarReader(ABC, Generic[T_index]):
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
    def __str__(self) -> str:
        """
        Must return a descriptive string of the concrete reader.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_itar_sample_pointer(self, idx: T_index) -> ITarSamplePointer:
        """Get the ITarSample object for the given index."""
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

    def _get_item_by_sample_pointer(
        self,
        sample_pointer: ITarSamplePointer,
        restore_index: Union[str, int],
        entry_match_fn: Optional[Callable[[str], bool]] = None,
    ) -> Union["ITarReader", FilteredSample, None]:
        """
        Get a sample from the dataset or slice it.

        Args:
            sample_pointer: The sample pointer to get the sample from.
            sample_index: The global index of the sample in the dataset.
            entry_match_fn: An optional function to filter the entries in the sample.

        Returns:
            The sample or None if the sample is not found.
        """

        # Open the tar file (cached)
        tar_file = self._get_itarfile_cached(sample_pointer.tar_file_id)
        shard_name = self.tar_filenames[sample_pointer.tar_file_id]
        sample_base_name = None
        sample_name = None
        group_parts: Dict[str, bytes] = {}
        file_names: list[str] = []

        # Position the tar file at the correct offset
        tar_file.offset = sample_pointer.byte_offset

        while tar_file.offset < sample_pointer.byte_offset + sample_pointer.byte_size:
            tarinfo = tar_file.next()
            if tarinfo is None:
                raise ValueError(
                    f"Unexpected end of tar file: {self.tar_filenames[sample_pointer.tar_file_id]}"
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

            if entry_match_fn is not None:
                # If entry_match_fn is provided, use it to determine if we should take this entry
                take_entry = entry_match_fn(fname)
            else:
                # If no entry_match_fn is provided, use the part_filter to determine if we should take this entry
                take_entry = self.part_filter is None or self.part_filter(cur_ext)

            if take_entry:
                member_bytes = tar_file.extractfile(tarinfo).read()
                group_parts[cur_ext] = member_bytes
                file_names.append(fname)
        if sample_base_name is None:
            raise ValueError(f"No valid files found in sample {sample_pointer}")

        return FilteredSample(
            __key__=f"{shard_name}/{sample_base_name}",
            __shard__=self.tar_filenames[sample_pointer.tar_file_id],
            __restore_key__=("Webdataset", restore_index),
            __sources__=(
                SourceInfo(
                    dataset_path=self.base_path,
                    index=restore_index,
                    shard_name=shard_name,
                    file_names=tuple(file_names),
                ),
            ),
            **group_parts,
        )

    @overload
    def __getitem__(self, key: T_index) -> Optional[FilteredSample]: ...

    @overload
    def __getitem__(self, key: slice) -> "ITarReader": ...

    def __getitem__(self, key: Union[slice, T_index]) -> Union["ITarReader", FilteredSample, None]:
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

        sample_pointer = self._get_itar_sample_pointer(idx)

        return self._get_item_by_sample_pointer(sample_pointer, idx)


class JoinIndexFileITarReader(ITarReader[int]):
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

    def _get_join_index_reader_cached(self, sample_idx: int) -> JoinIndexReader:
        """
        Get the JoinIndexReader object for the given sample index, or create it if it doesn't exist.
        """

        if sample_idx not in self.index_reader_cache:
            index_reader = JoinIndexReader(self.index_file, column=self.column)
            self.index_reader_cache[sample_idx] = index_reader

        # If we hit the limit of open files, close the least recently used file
        while len(self.index_reader_cache) > self.index_reader_cache_size:
            # Get the oldest file
            lru_key = next(iter(self.index_reader_cache))

            self.index_reader_cache[lru_key].close()
            del self.index_reader_cache[lru_key]

        return self.index_reader_cache[sample_idx]

    def _get_itar_sample_pointer(self, sample_idx: int) -> ITarSamplePointer:
        """
        Get the ITarSample object for the given index.
        """
        index_reader = self._get_join_index_reader_cached(sample_idx)
        row = index_reader[sample_idx]

        # Update cache entry
        new_offset = index_reader.tell_row()
        del self.index_reader_cache[sample_idx]
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


class ShardInfosITarReader(ITarReader[int]):
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


class SqliteITarEntryReader(ITarReader[str]):
    """
    A concrete ITarReader that constructs its internal sample list from a SQLite database.
    """

    sqlite_reader: SqliteIndexReader

    def __init__(
        self,
        base_path: EPath,
        part_filter: Optional[Callable[[str], bool]] = None,
        itar_cache_size: int = 5,
        sample_filter: Optional[Callable[[str], bool]] = None,
        key_is_full_entryname: bool = False,
    ):
        from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
        from megatron.energon.flavors.webdataset.indexing import SqliteIndexReader

        # shard_name_to_info_idx = {name: i for i, name in enumerate(wds_meta.info_shard_files)}
        tar_filenames = get_info_shard_files(base_path)
        tar_filepaths = [base_path / fn for fn in tar_filenames]

        # Initialize the SQLite reader
        sqlite_path = base_path / MAIN_FOLDER_NAME / "index.sqlite"
        self.sqlite_reader = SqliteIndexReader(sqlite_path)

        self.key_is_full_entryname = key_is_full_entryname

        super().__init__(
            base_path=base_path,
            tar_filenames=tar_filenames,
            tar_filepaths=tar_filepaths,
            part_filter=part_filter,
            itar_cache_size=itar_cache_size,
            sample_filter=sample_filter,
        )

    def _get_itar_sample_pointer(self, sample_key: str) -> ITarSamplePointer:
        """
        Get the ITarSample object for the given index.
        """

        return self.sqlite_reader.get_sample_pointer_by_key(sample_key)

    def list_all_samples(self) -> Generator[Tuple[str, int, int], None, None]:
        return self.sqlite_reader.list_all_samples()

    def list_all_sample_parts(self) -> Generator[Tuple[str, int, int], None, None]:
        return self.sqlite_reader.list_all_sample_parts()

    def list_sample_parts(
        self, sample_key: str, slow_mode: bool = False
    ) -> Generator[Tuple[str, int, int], None, None]:
        """Given a sample key, list all its parts. (E.g. given 001, list 001.jpg, 001.json, etc.)

        If allow_fallback is True, and the database is an older version, which
        does not contain the sample_parts table, we will try to find the sample_parts
        in the tar files.

        Args:
            sample_key: The sample key to list the parts of.
            allow_fallback: If True, and the database is an older version, which
                does not contain the sample_parts table, we will try to find the sample_parts
                in the tar files.

        Returns:
            A generator of tuples of (part_name, size, tar_file_id)
        """

        if not slow_mode:
            yield from self.sqlite_reader.list_sample_parts(sample_key)
        else:
            sample_pointer = self._get_itar_sample_pointer(sample_key)

            sample = self._get_item_by_sample_pointer(sample_pointer, 0, entry_match_fn=None)
            assert isinstance(sample, dict), f"Sample not found: {sample_pointer}"

            for ext in sample.keys():
                if not ext.startswith("__"):
                    yield ext, len(sample[ext]), sample_pointer.tar_file_id

    def get_total_size(self) -> int:
        return self.sqlite_reader.get_total_size()

    @overload
    def __getitem__(self, key: str) -> Union[FilteredSample, tuple[bytes, SourceInfo]]: ...

    @overload
    def __getitem__(self, key: slice) -> "ITarReader": ...

    def __getitem__(
        self, key: Union[slice, str]
    ) -> Union[FilteredSample, tuple[bytes, SourceInfo], ITarReader]:
        """
        Either get a sample from the dataset by the sample key including all its entries,
        or get the bytes of a specific entry by the full filename of the entry inside the tar.
        """

        if isinstance(key, slice):
            # Return a new reader with a sliced samples tensor
            raise NotImplementedError("Slicing is not yet implemented")
        assert isinstance(key, str), "Invalid argument type for __getitem__"

        if self.key_is_full_entryname:
            m = split_name_re.match(key)
            if not m:
                raise ValueError(f"Invalid file name: {key}")

            sample_key, sample_ext = m.groups()
            entry_match_fn = lambda fname: key == fname
        else:
            sample_key = key
            sample_ext = None
            entry_match_fn = None

        sample_pointer = self._get_itar_sample_pointer(sample_key)

        sample = self._get_item_by_sample_pointer(
            sample_pointer, key, entry_match_fn=entry_match_fn
        )
        assert sample is not None, f"Sample not found: {sample_key}"

        if self.key_is_full_entryname:
            assert isinstance(sample_ext, str)
            assert len(sample["__sources__"]) == 1
            # Return the bytes directly
            return sample[sample_ext], sample["__sources__"][0]
        else:
            return sample  # Return the FilteredSample

    def __len__(self) -> int:
        """Return the total number of samples in the database."""
        return self.sqlite_reader.get_sample_count()

    def __str__(self) -> str:
        """Return a descriptive string of this reader."""
        return (
            f"SqliteITarEntryReader("
            f"len={len(self)}, base_path={self.base_path}, "
            f"len(shards)={len(self.tar_filenames)}, "
            f"shards=[{self.tar_filenames[0] if self.tar_filenames else 'N/A'}, ...])"
        )

    def close(self):
        """Close the SQLite reader and any open ITarFiles."""
        # Close the SQLite reader
        if hasattr(self, "sqlite_reader") and self.sqlite_reader is not None:
            self.sqlite_reader.close()

        # Close any open ITarFiles (using parent class implementation)
        for tar_file_id in list(self.itar_files_cache.keys()):
            tar_file = self.itar_files_cache[tar_file_id]
            if (
                tar_file is not None
                and hasattr(tar_file, "fileobj")
                and tar_file.fileobj is not None
            ):
                tar_file.fileobj.close()
            if tar_file is not None and hasattr(tar_file, "close"):
                tar_file.close()
            del self.itar_files_cache[tar_file_id]
