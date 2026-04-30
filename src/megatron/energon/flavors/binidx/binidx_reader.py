# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import os
import struct
from typing import Callable, Generator, Optional, Tuple

import numpy

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.structs import FilteredSample
from megatron.energon.source_info import SourceInfo

_INDEX_HEADER = b"MMIDIDX\x00\x00"

# Dtype codes matching megatron.core.datasets.indexed_dataset.DType
_DTYPE_FROM_CODE = {
    1: numpy.uint8,
    2: numpy.int8,
    3: numpy.int16,
    4: numpy.int32,
    5: numpy.int64,
    6: numpy.float64,
    7: numpy.float32,
    8: numpy.uint16,
}


class IBinIdxReader:
    """
    Reader for Megatron-LM pre-tokenized binary dataset files (.bin + .idx).

    Satisfies the same duck-typed contract as IJsonlReader:
    - __len__() -> int
    - __getitem__(idx) -> FilteredSample | None
    - close()

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

        idx_path = str(self.bin_path).removesuffix(".bin") + ".idx"

        # Parse the .idx header
        with open(idx_path, "rb") as f:
            header = f.read(9)
            assert header == _INDEX_HEADER, f"Bad header in {idx_path}"

            version = struct.unpack("<Q", f.read(8))[0]
            assert version == 1, f"Unsupported version {version} in {idx_path}"

            dtype_code = struct.unpack("<B", f.read(1))[0]
            self.dtype = _DTYPE_FROM_CODE[dtype_code]
            self.dtype_size = numpy.dtype(self.dtype).itemsize

            self.sequence_count = struct.unpack("<Q", f.read(8))[0]
            self.document_count = struct.unpack("<Q", f.read(8))[0]

            data_offset = f.tell()

        # Read .idx arrays into memory (small: ~12 bytes per sequence)
        with open(idx_path, "rb") as f:
            f.seek(data_offset)
            lengths_bytes = f.read(self.sequence_count * 4)  # int32
            pointers_bytes = f.read(self.sequence_count * 8)  # int64

        self.sequence_lengths = numpy.frombuffer(lengths_bytes, dtype=numpy.int32)
        self.sequence_pointers = numpy.frombuffer(pointers_bytes, dtype=numpy.int64)

        # Keep .bin file open for seek+read per sample (no mmap needed)
        self._bin_file = open(str(self.bin_path), "rb")
        self._bin_size = os.path.getsize(str(self.bin_path))

    def __len__(self) -> int:
        return self.sequence_count

    def __str__(self) -> str:
        return f"IBinIdxReader(bin_path={self.bin_path})"

    def __getitem__(self, idx: int | str) -> FilteredSample | tuple[bytes, SourceInfo] | None:
        full_entry_name = False
        if isinstance(idx, str):
            if idx.endswith(".bin"):
                num_idx = idx.removesuffix(".bin")
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
        byte_offset = int(self.sequence_pointers[idx])
        length = int(self.sequence_lengths[idx])
        byte_length = length * self.dtype_size

        self._bin_file.seek(byte_offset)
        token_bytes = self._bin_file.read(byte_length)

        source_info = SourceInfo(
            dataset_path=str(self.bin_path),
            index=idx,
            shard_name=self.bin_path.name,
            file_names=(f"{key}.bin",),
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
        for i in range(self.sequence_count):
            yield str(i), int(self.sequence_lengths[i]) * self.dtype_size, 0

    def list_all_sample_parts(self) -> Generator[Tuple[str, int, int], None, None]:
        for i in range(self.sequence_count):
            yield f"{i}.bin", int(self.sequence_lengths[i]) * self.dtype_size, 0

    def list_sample_parts(self, sample_key: str) -> Generator[Tuple[str, int, int], None, None]:
        try:
            idx = int(sample_key)
        except ValueError:
            raise ValueError(f"Invalid bin-idx sample key: {sample_key}")
        yield f"{sample_key}.bin", int(self.sequence_lengths[idx]) * self.dtype_size, 0

    def get_total_size(self) -> int:
        return self._bin_size

    def close(self):
        self._bin_file.close()

    @staticmethod
    def count_samples(bin_path: EPath | str) -> int:
        """Read only the sequence count from the .idx header without full mmap."""
        idx_path = str(bin_path).removesuffix(".bin") + ".idx"
        with open(idx_path, "rb") as f:
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
        idx_path = str(bin_path).removesuffix(".bin") + ".idx"
        with open(idx_path, "rb") as f:
            header = f.read(9)
            assert header == _INDEX_HEADER, f"Bad header in {idx_path}"
            _version = f.read(8)  # skip version
            dtype_code = struct.unpack("<B", f.read(1))[0]
        return _DTYPE_FROM_CODE[dtype_code]
