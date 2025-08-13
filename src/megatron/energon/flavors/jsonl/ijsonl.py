# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import struct
from typing import BinaryIO, Dict, Generator, Optional, Tuple, Union

from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath

IJSONL_SUFFIX = ".jsonl.idx"


@edataclass
class IJsonlSamplePointer:
    """
    Points to a sample inside some jsonl file on disk.
    """

    # The index of the sample in the jsonl file.
    index: int

    # The byte offset of the sample in the jsonl file.
    byte_offset: int
    # The size of the sample in the jsonl file.
    byte_size: int


class IJsonlIndexReader:
    def __init__(self, jsonl_path: Union[EPath, str]):
        jsonl_path = EPath(jsonl_path)
        index_path = jsonl_path.with_suffix(IJSONL_SUFFIX)
        self._length = index_path.size() // 8
        self.ijsonl = index_path.open("rb")

    def __getitem__(self, index: int) -> int:
        if index >= self._length or index < 0:
            raise IndexError(f"Index {index} out of range")

        if self.ijsonl.tell() != 8 * index:
            self.ijsonl.seek(8 * index)

        return struct.unpack("Q", self.ijsonl.read(8))[0]

    def __iter__(self) -> Generator[int, None, None]:
        self.ijsonl.seek(0)
        while True:
            raw = self.ijsonl.read(8)
            if len(raw) == 0:
                break
            assert len(raw) == 8
            yield struct.unpack("Q", raw)[0]

    def __len__(self) -> int:
        return self._length

    def close(self):
        self.ijsonl.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def count_samples(jsonl_path: EPath | str) -> int:
        return EPath(jsonl_path).with_suffix(IJSONL_SUFFIX).size() // 8 - 1

    @staticmethod
    def size(jsonl_path: EPath) -> int:
        with IJsonlIndexReader(jsonl_path) as reader:
            return reader[len(reader) - 1]


class IJsonlIndexWriter:
    def __init__(self, jsonl_path: EPath):
        self.final_name = jsonl_path.with_suffix(IJSONL_SUFFIX)
        self.tmp_name = jsonl_path.with_suffix(IJSONL_SUFFIX + ".tmp")
        self.ijsonl = self.tmp_name.open("wb")

    def append(self, offset: int):
        self.ijsonl.write(struct.pack("Q", offset))

    def close(self, finalize: bool = True):
        self.ijsonl.close()
        if finalize:
            self.tmp_name.move(self.final_name)
        else:
            self.tmp_name.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(finalize=exc_val is None)


@edataclass
class CacheEntry:
    ijsonl_index_reader: IJsonlIndexReader
    lookahead_offset: Optional[int] = None
    lookahead_byteoffset: Optional[int] = None


class CachedIJsonlOffsetReader:
    """
    This class is a high-level wrapper around IJsonlIndexReader that caches some
    of the recent lookups for faster access. It is designed for the case when
    you need to read multiple offsets from the same jsonl file.

    Args:
        cache_size: The number of entries to keep in the cache. By default, we keep 32.
    """

    def __init__(self, jsonl_file: Union[str, EPath], cache_size: int = 32):
        # Maps current_offset -> CacheEntry
        self.ijsonl_index_reader_cache: Dict[int, CacheEntry] = {}
        self.cache_size = cache_size
        self.jsonl_file = EPath(jsonl_file)

    def close(self):
        for cache_entry in self.ijsonl_index_reader_cache.values():
            cache_entry.ijsonl_index_reader.close()
        self.ijsonl_index_reader_cache.clear()

    def _find_or_create_entry(
        self,
        sample_offset: int,
    ) -> CacheEntry:
        """
        1. If we already have a key == sample_offset, return it.
        2. Otherwise, create a new entry or reuse the oldest entry.
        """
        # Direct hit in the cache?
        if sample_offset in self.ijsonl_index_reader_cache:
            return self.ijsonl_index_reader_cache[sample_offset]

        # We didn't find an existing entry. Create a new one.
        # Evict if needed.
        if len(self.ijsonl_index_reader_cache) >= self.cache_size:
            # Reuse the oldest entry
            oldest_key = next(iter(self.ijsonl_index_reader_cache))
            cache_entry = self.ijsonl_index_reader_cache.pop(oldest_key)
        else:
            new_reader = IJsonlIndexReader(self.jsonl_file)
            cache_entry = CacheEntry(ijsonl_index_reader=new_reader)
        self.ijsonl_index_reader_cache[sample_offset] = cache_entry
        return cache_entry

    def _get_ijsonl_byte_offset_with_entry(
        self,
        cache_entry: CacheEntry,
        sample_offset: int,
    ) -> Tuple[int, int]:
        """
        Return (start_byte_offset, length_to_next),
        possibly using per-entry lookahead for speed.
        """
        ijsonl_index_reader = cache_entry.ijsonl_index_reader

        # If offset=0, define the result as byte offset=0 for convenience
        if sample_offset == 0:
            result_byte_offset = 0
        elif sample_offset == cache_entry.lookahead_offset:
            # Reuse the previously cached byte offset from the lookahead
            assert cache_entry.lookahead_byteoffset is not None, (
                "Lookahead offset matched but no lookahead byte offset found."
            )
            result_byte_offset = cache_entry.lookahead_byteoffset
        else:
            # Normal random access
            result_byte_offset = ijsonl_index_reader[sample_offset]

        # Prepare the lookahead for (sample_offset+1)
        next_offset = sample_offset + 1
        try:
            cache_entry.lookahead_byteoffset = ijsonl_index_reader[next_offset]
            cache_entry.lookahead_offset = next_offset
        except IndexError:
            cache_entry.lookahead_offset = None
            cache_entry.lookahead_byteoffset = None

        # length = difference to the next offset, or 0 if none
        if cache_entry.lookahead_byteoffset is not None:
            length = cache_entry.lookahead_byteoffset - result_byte_offset
        else:
            length = 0

        return result_byte_offset, length

    def get_ijsonl_byte_offset(
        self,
        sample_offset: int = 0,
    ) -> Tuple[int, int]:
        """
        High-level API to get the byte offset and length for the given file & sample_offset.
        """

        # Find or create the suitable CacheEntry
        entry = self._find_or_create_entry(sample_offset)

        # Use (and update) the per-entry lookahead logic
        result_byte_offset, length = self._get_ijsonl_byte_offset_with_entry(entry, sample_offset)

        # Update cache entry with the new offset
        self.ijsonl_index_reader_cache.pop(sample_offset)
        if entry.lookahead_offset is not None:
            new_key = entry.lookahead_offset
            if new_key not in self.ijsonl_index_reader_cache:
                self.ijsonl_index_reader_cache[new_key] = entry
            else:
                # Already have this entry in the cache, so we can close the reader and use the existing one
                # TODO: We may actually may want to keep multiple readers open, because they may be multiple
                # sequences to the same sequence.
                entry.ijsonl_index_reader.close()
        else:
            # No lookahead, so we can close the reader
            entry.ijsonl_index_reader.close()

        return result_byte_offset, length

    def __len__(self) -> int:
        if len(self.ijsonl_index_reader_cache) == 0:
            return IJsonlIndexReader.count_samples(self.jsonl_file)
        return len(next(iter(self.ijsonl_index_reader_cache.values())).ijsonl_index_reader) - 1

    def get_total_size(self) -> int:
        if len(self.ijsonl_index_reader_cache) == 0:
            self.ijsonl_index_reader_cache[0] = CacheEntry(
                ijsonl_index_reader=IJsonlIndexReader(self.jsonl_file)
            )
        reader = next(iter(self.ijsonl_index_reader_cache.values())).ijsonl_index_reader
        return reader[len(reader) - 1]


class IJsonlFile:
    """
    This class is a high-level wrapper around a binary file that allows for reading a jsonl file,
    with random access while keeping the file open.

    Usage:
        with open(filename, "rb") as fileobj:
            with IJsonlFile(fileobj=fileobj) as f:
                data = f.next(offset=101888, size=100)
                json.loads(data)
        # Or, if you want to read the whole file:
        with open(filename, "rb") as fileobj:
            with IJsonlFile(fileobj=fileobj) as f:
                while True:
                    data = f.next()
                    if data is None:
                        break
                    json.loads(data)
        # Or, if you want to read the whole file:
        with open(filename, "rb") as fileobj:
            with IJsonlFile(fileobj=fileobj) as f:
                for data in f:
                    json.loads(data)
    """

    def __init__(self, fileobj: BinaryIO):
        self.fileobj = fileobj

    def seek(self, offset: int):
        self.fileobj.seek(offset)

    def next(self, offset: int | None = None, size: int | None = None) -> bytes | None:
        if offset is not None and offset != self.fileobj.tell():
            self.fileobj.seek(offset)
        if size is None:
            entry = self.fileobj.readline()
            if entry == b"":
                return None
            return entry
        else:
            assert size > 0, "Size must contain at least the line terminator and a json object"
            data = self.fileobj.read(size)
            if data == b"":
                return None
            return data

    def __iter__(self) -> Generator[bytes, None, None]:
        while True:
            data = self.next()
            if data is None:
                break
            yield data

    def close(self):
        self.fileobj.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fileobj.close()
