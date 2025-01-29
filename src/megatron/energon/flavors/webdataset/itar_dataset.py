# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from io import BytesIO
from typing import Callable, Dict, List, Optional, Tuple, Union, overload

import torch

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.config import skip_meta_re, split_name_re
from megatron.energon.flavors.webdataset.indexing import JoinIndexReader
from megatron.energon.flavors.webdataset.itar import CachedItarOffsetReader, ITarFile
from megatron.energon.flavors.webdataset.structs import FilteredSample, ShardInfo, reraise_exception

# The cache size determines how many tar files are kept open at the same time.
ITAR_CACHE_SIZE = 5


class ITarDataset:
    """
    An ITarDataset keeps a list of pointers to samples in various tar files.
    Each sample can be in a different tar file and at a different offset.
    It supports random access to any sample at any time, but in case of sequential access,
    it is more efficient.

    Internally, ITarDataset keeps the most recently used tar files open as ITarFile objects.
    When a new sample is requested, it first checks if the sample is in the currently open tar files,
    or even at the current offset in the current tar file.

    For memory efficiency, the sample list is stored as a torch tensor without strings.
    """

    tar_filenames: List[str]
    tar_filepaths: List[EPath]
    itar_files_cache: Dict[int, ITarFile]
    samples: (
        torch.Tensor
    )  # Shape [num_samples, 3] with columns [tar_file_id, byte_offset, byte_size]
    part_filter: Optional[Callable[[str], bool]]

    COL_TAR_FILE_ID = 0
    COL_BYTE_OFFSET = 1
    COL_BYTE_SIZE = 2

    def __init__(
        self,
        tar_filenames: List[str],
        tar_filepaths: List[EPath],
        samples: torch.Tensor,
        part_filter: Optional[Callable[[str], bool]] = None,
    ):
        assert len(tar_filenames) == len(tar_filepaths)
        self.tar_filenames = tar_filenames
        self.tar_filepaths = tar_filepaths
        self.itar_files_cache = dict()
        self.samples = samples
        self.part_filter = part_filter

    @staticmethod
    def from_join_index_file(
        index_file: EPath,
        column: int,
        tar_filenames: List[str],
        base_path: EPath,
        part_filter: Optional[Callable[[str], bool]] = None,
    ) -> "ITarDataset":
        """
        Create an ITarDataset from one column of a join index file.

        The tar_filenames are assumed to be relative to the base_path.
        """
        with JoinIndexReader(index_file) as jir:
            all_cols = jir.get_as_tensor()

        samples = all_cols[:, column, :].clone()

        # Create the full path to the tar files
        tar_filepaths = []
        for tar_filename in tar_filenames:
            tar_filepaths.append(base_path / tar_filename)

        return ITarDataset(tar_filenames, tar_filepaths, samples, part_filter=part_filter)

    @staticmethod
    def from_shardinfos(
        shardinfos: List[ShardInfo], part_filter: Optional[Callable[[str], bool]] = None
    ) -> "ITarDataset":
        """
        Create an ITarDataset from a list of ShardInfos.
        """

        # TODO: We could remove excluded shards and samples here. Or remove exclusion support.

        cur_tar_files: Dict[str, Tuple[int, EPath]] = dict()

        num_samples = sum(shard.count for shard in shardinfos)

        samples = torch.zeros((num_samples, 3), dtype=torch.int64, device="cpu")

        cached_offset_reader = CachedItarOffsetReader()

        samp_idx = 0
        for shardinfo in shardinfos:
            filepath = shardinfo.path
            filename = shardinfo.name

            if filename in cur_tar_files:
                cur_tar_file_idx = cur_tar_files[filename][0]
            else:
                cur_tar_file_idx = len(cur_tar_files)
                cur_tar_files[filename] = (cur_tar_file_idx, filepath)

            # Add the samples of the current shard to the samples tensor
            samp_idx_end = samp_idx + shardinfo.count

            # Set same tar file for all
            samples[samp_idx:samp_idx_end, ITarDataset.COL_TAR_FILE_ID] = cur_tar_file_idx

            tar_index_reader = cached_offset_reader.tar_index_reader(shardinfo.path)
            for i in range(shardinfo.count):
                sample_idx_in_shard = shardinfo.offset + i

                # Get the byte offset and size of that sample in that tar file
                byte_offset, byte_size = cached_offset_reader.get_itar_byte_offset_with_reader(
                    tar_index_reader, sample_idx_in_shard
                )

                samples[samp_idx, ITarDataset.COL_BYTE_OFFSET] = byte_offset
                samples[samp_idx, ITarDataset.COL_BYTE_SIZE] = byte_size
                samp_idx += 1

        assert samp_idx == num_samples

        return ITarDataset(
            list(cur_tar_files.keys()),
            [x[1] for x in cur_tar_files.values()],
            samples,
            part_filter=part_filter,
        )

    def __len__(self):
        return self.samples.size(0)

    def _get_itarfile_cached(self, tar_file_id: int) -> ITarFile:
        """
        Get the ITarFile object for the given tar file id.
        If the file is not already open, open it.
        """

        if tar_file_id not in self.itar_files_cache:
            file_object = open(str(self.tar_filepaths[tar_file_id]), "rb")
            tar_file = ITarFile.open(fileobj=file_object, mode="r:")
            self.itar_files_cache[tar_file_id] = tar_file

        # If we hit the limit of open files, close the least recently used file
        while len(self.itar_files_cache) > ITAR_CACHE_SIZE:
            # Get the oldest file
            lru_key = next(iter(self.itar_files_cache))

            self.itar_files_cache[lru_key].fileobj.close()
            self.itar_files_cache[lru_key].close()
            del self.itar_files_cache[lru_key]

        return self.itar_files_cache[tar_file_id]

    @overload
    def __getitem__(self, key: int) -> FilteredSample: ...

    @overload
    def __getitem__(self, key: slice) -> "ITarDataset": ...

    def __getitem__(self, key: Union[slice, int]) -> Union["ITarDataset", FilteredSample]:
        """
        Get a sample from the dataset or slice it.
        """

        if isinstance(key, slice):
            return ITarDataset(
                tar_filenames=self.tar_filenames,
                tar_filepaths=self.tar_filepaths,
                samples=self.samples[key],
                part_filter=self.part_filter,
            )
        elif isinstance(key, int):
            idx = key
        else:
            raise TypeError("Invalid argument type")

        # Get the sample
        sample = self.samples[idx]

        # Get the tar file id
        tar_file_id = int(sample[ITarDataset.COL_TAR_FILE_ID].item())

        # Get the byte offset and size
        byte_offset = int(sample[ITarDataset.COL_BYTE_OFFSET].item())
        byte_size = int(sample[ITarDataset.COL_BYTE_SIZE].item())

        # Open the tar file if it is not already open
        tar_file = self._get_itarfile_cached(tar_file_id)

        # Read the sample
        sample_base_name = None

        # Here we set the byte offset in the tar file
        # Note that if the the offset is already at the correct position, this is a no-op
        tar_file.offset = byte_offset

        group_parts = dict()

        while tar_file.offset < byte_offset + byte_size:
            tarinfo = tar_file.next()

            if tarinfo is None:
                raise ValueError(f"Unexpected end of tar file: {self.tar_filenames[tar_file_id]}")

            fname = tarinfo.name
            if not tarinfo.isfile():
                continue
            if fname is None:
                continue
            if skip_meta_re.match(fname):
                continue

            # Get base_name and extension if available
            m = split_name_re.match(fname)
            if not m:
                continue
            cur_base_name, cur_ext = m.groups()

            if sample_base_name is None:
                sample_base_name = cur_base_name
            else:
                assert (
                    sample_base_name == cur_base_name
                ), f"Expected {sample_base_name}, got {cur_base_name}"

            if self.part_filter is None or self.part_filter(cur_ext):
                member_bytes = tar_file.extractfile(tarinfo).read()
                group_parts[cur_ext] = member_bytes

        return FilteredSample(
            __key__=cur_base_name,
            __shard__=self.tar_filenames[tar_file_id],
            __restore_key__=("itar", "", idx),
            **group_parts,
        )
