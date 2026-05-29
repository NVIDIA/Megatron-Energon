# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import struct
from abc import ABC, abstractmethod
from typing import BinaryIO, Callable, Generator, Optional, Tuple

import numpy as np

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.structs import FilteredSample
from megatron.energon.source_info import SourceInfo

_INDEX_HEADER = b"MMIDIDX\x00\x00"

# Dtype codes matching megatron.core.datasets.indexed_dataset.DType
_DTYPE_FROM_CODE = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}


def _idx_path(bin_path: EPath) -> EPath:
    return bin_path.parent / (bin_path.name.removesuffix(".bin") + ".idx")


class IIdxReader(ABC):
    """
    Base class for .idx file readers.

    The .idx format (from megatron/core/datasets/indexed_dataset.py):
        Header:  b"MMIDIDX\\x00\\x00" (9 bytes)
        Version: uint64 (must be 1)
        DType:   uint8 code
        SeqCnt:  uint64
        DocCnt:  uint64
        Then:    int32[SeqCnt]  sequence_lengths
                 int64[SeqCnt]  sequence_pointers (byte offsets into .bin)
                 int64[DocCnt]  document_indices
    """

    idx_path: EPath

    dtype: np.dtype
    dtype_size: int
    sequence_count: int
    document_count: int
    _length_offset: int
    _pointer_offset: int

    def __init__(self, file: BinaryIO) -> None:
        # Parse the .idx header
        header = file.read(9)
        assert header == _INDEX_HEADER, f"Bad header in {self.idx_path}"

        version = struct.unpack("<Q", file.read(8))[0]
        assert version == 1, f"Unsupported version {version} in {self.idx_path}"

        dtype_code = struct.unpack("<B", file.read(1))[0]
        self.dtype = _DTYPE_FROM_CODE[dtype_code]
        self.dtype_size = np.dtype(self.dtype).itemsize

        self.sequence_count = struct.unpack("<Q", file.read(8))[0]
        self.document_count = struct.unpack("<Q", file.read(8))[0]

        self._length_offset = file.tell()
        self._pointer_offset = self._length_offset + self.sequence_count * 4

    def __len__(self) -> int:
        return self.sequence_count

    @abstractmethod
    def length(self, idx: int) -> int: ...

    @abstractmethod
    def pointer(self, idx: int) -> int: ...

    @abstractmethod
    def close(self): ...


class IdxReader(IIdxReader):
    """
    Reader for .idx files that uses direct file access.
    """

    def __init__(self, idx_path: EPath):
        self.idx_path = idx_path
        self._idx_file = idx_path.open("rb")
        try:
            super().__init__(self._idx_file)
        except:
            self._idx_file.close()
            raise

    def length(self, idx: int) -> int:
        self._idx_file.seek(self._length_offset + idx * 4)
        return struct.unpack("<i", self._idx_file.read(4))[0]

    def pointer(self, idx: int) -> int:
        self._idx_file.seek(self._pointer_offset + idx * 8)
        return struct.unpack("<q", self._idx_file.read(8))[0]

    def close(self):
        self._idx_file.close()


class MMapIdxReader(IIdxReader):
    """
    Reader for .idx files that uses memory mapping.
    """

    def __init__(self, idx_path: EPath):
        self.idx_path = idx_path
        self._idx_file = idx_path.open("rb")
        try:
            super().__init__(self._idx_file)

            self._sequence_lengths = np.memmap(
                self._idx_file,
                dtype=np.int32,
                mode="r",
                offset=self._length_offset,
                shape=(self.sequence_count,),
            )
            self._sequence_pointers = np.memmap(
                self._idx_file,
                dtype=np.int64,
                mode="r",
                offset=self._pointer_offset,
                shape=(self.sequence_count,),
            )
        except:
            self._idx_file.close()
            raise

    def pointer(self, idx: int) -> int:
        return self._sequence_pointers[idx]

    def length(self, idx: int) -> int:
        return self._sequence_lengths[idx]

    def close(self):
        del self._sequence_lengths
        del self._sequence_pointers
        self._idx_file.close()


class BinIdxReader:
    """
    Reader for Megatron-LM pre-tokenized binary dataset files (.bin + .idx).
    """

    bin_path: EPath
    sample_filter: Optional[Callable[[str], bool]]

    def __init__(
        self,
        bin_path: EPath,
        sample_filter: Optional[Callable[[str], bool]] = None,
        index_cache_size: int = 5,
    ):
        self.bin_path = EPath(bin_path)
        self.sample_filter = sample_filter

        idx_path = _idx_path(self.bin_path)

        self._bin_size = self.bin_path.size()

        if bin_path.is_local():
            self._idx_reader = MMapIdxReader(idx_path)
        else:
            self._idx_reader = IdxReader(idx_path)

        try:
            # Keep .bin file open for seek+read per sample (no mmap needed)
            self._bin_file = self.bin_path.open("rb")
        except:
            self._idx_reader.close()
            raise

    def __getitem__(self, idx: int | str) -> FilteredSample | tuple[bytes, SourceInfo] | None:
        full_entry_name = False
        if isinstance(idx, str):
            if idx.endswith(".tokens"):
                num_idx = idx.removesuffix(".tokens")
                full_entry_name = True
            else:
                num_idx = idx
            try:
                idx = int(num_idx)
            except ValueError:
                raise ValueError(f"Invalid bin-idx sample key: {idx}")

        key = str(idx)
        if self.sample_filter is not None and not self.sample_filter(key):
            return None

        # Read token data from .bin via seek+read
        byte_offset = self._idx_reader.pointer(idx)
        length = self._idx_reader.length(idx)
        byte_length = length * self._idx_reader.dtype_size

        self._bin_file.seek(byte_offset)
        token_bytes = self._bin_file.read(byte_length)

        source_info = SourceInfo(
            dataset_path=self.bin_path,
            index=idx,
            shard_name=self.bin_path.name,
            file_names=(f"{key}.tokens",),
        )

        if full_entry_name:
            return token_bytes, source_info

        return FilteredSample(
            __key__=key,
            __shard__=self.bin_path.name,
            __restore_key__=("Webdataset", idx),
            __sources__=(source_info,),
            tokens=token_bytes,
        )

    def list_all_samples(self) -> Generator[Tuple[str, int, int], None, None]:
        for i in range(len(self._idx_reader)):
            yield str(i), self._idx_reader.length(i) * self._idx_reader.dtype_size, 0

    def list_all_sample_parts(self) -> Generator[Tuple[str, int, int], None, None]:
        for i in range(len(self._idx_reader)):
            yield f"{i}.bin", self._idx_reader.length(i) * self._idx_reader.dtype_size, 0

    def list_sample_parts(self, sample_key: str) -> Generator[Tuple[str, int, int], None, None]:
        try:
            idx = int(sample_key)
        except ValueError:
            raise ValueError(f"Invalid bin-idx sample key: {sample_key}")
        yield f"{sample_key}.bin", self._idx_reader.length(idx) * self._idx_reader.dtype_size, 0

    def get_total_size(self) -> int:
        return self._bin_size

    def close(self):
        self._bin_file.close()

    @staticmethod
    def count_samples(bin_path: EPath | str) -> int:
        """Read only the sequence count from the .idx header without full mmap."""
        idx_path = _idx_path(EPath(bin_path))
        with idx_path.open("rb") as f:
            header = f.read(9)
            assert header == _INDEX_HEADER, f"Bad header in {idx_path}"
            version = struct.unpack("<Q", f.read(8))[0]
            assert version == 1, f"Unsupported version {version} in {idx_path}"
            _dtype_code = f.read(1)  # skip dtype
            sequence_count = struct.unpack("<Q", f.read(8))[0]
        return sequence_count

    @staticmethod
    def read_dtype(bin_path: EPath | str) -> type:
        """Read the dtype from the .idx header."""
        idx_path = _idx_path(EPath(bin_path))
        with idx_path.open("rb") as f:
            header = f.read(9)
            assert header == _INDEX_HEADER, f"Bad header in {idx_path}"
            _version = f.read(8)  # skip version
            dtype_code = struct.unpack("<B", f.read(1))[0]
        return _DTYPE_FROM_CODE[dtype_code]
