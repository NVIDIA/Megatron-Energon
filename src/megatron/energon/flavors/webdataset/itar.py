# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import struct
import tarfile
from types import TracebackType
from typing import BinaryIO, Dict, Generator, Optional, Tuple, Type, Union

from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.retry_stream import RetryReadStream

ITAR_SUFFIX = ".tar.idx"


@edataclass
class ITarSamplePointer:
    """
    Points to a sample inside some tar file on disk.
    The tar_file_id refers to the tar_filenames in the reader.
    """

    # The index of the tar file, to be matched with the tar_filenames in the reader.
    tar_file_id: int
    # The byte offset of the sample in the tar file.
    byte_offset: int
    # The size of the sample in the tar file.
    byte_size: int


class TarIndexReader:
    def __init__(self, tar_path: Union[EPath, str]):
        tar_path = EPath(tar_path)
        self.itar = (tar_path.with_suffix(ITAR_SUFFIX)).open("rb")
        self._length = len(self)

    def __getitem__(self, index: int) -> int:
        if index >= self._length or index < 0:
            raise IndexError(f"Index {index} out of range")

        if self.itar.tell() != 8 * index:
            self.itar.seek(8 * index)

        return struct.unpack("Q", self.itar.read(8))[0]

    def __iter__(self) -> Generator[int, None, None]:
        self.itar.seek(0)
        while True:
            raw = self.itar.read(8)
            if len(raw) == 0:
                break
            assert len(raw) == 8
            yield struct.unpack("Q", raw)[0]

    def __len__(self) -> int:
        self.itar.seek(0, 2)
        return self.itar.tell() // 8

    def close(self):
        self.itar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TarIndexWriter:
    def __init__(self, tar_path: EPath):
        self.final_name = tar_path.with_suffix(ITAR_SUFFIX)
        self.tmp_name = tar_path.with_suffix(ITAR_SUFFIX + ".tmp")
        self.itar = self.tmp_name.open("wb")

    def append(self, offset: int):
        self.itar.write(struct.pack("Q", offset))

    def close(self, finalize: bool = True):
        self.itar.close()
        if finalize:
            self.tmp_name.move(self.final_name)
        else:
            self.tmp_name.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(finalize=exc_val is None)


class SubFileReader(BinaryIO):
    """A file-like object that reads a subfile (i.e. offset, size defined portion) of a larger
    file."""

    def __init__(self, stream: BinaryIO, offset: int, size: int):
        self.offset = offset
        self._pos = 0
        self.size = size
        self.stream = stream
        self.stream.seek(self.offset)

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            n = self.size - self._pos
        else:
            n = min(n, self.size - self._pos)
        if n == 0:
            return b""
        read = self.stream.read(n)
        self._pos += len(read)
        return read

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self.size + offset
        else:
            raise ValueError("Invalid whence value")
        self._pos = max(0, min(self._pos, self.size))
        self.stream.seek(self.offset + self._pos)
        return self._pos

    def tell(self) -> int:
        return self._pos

    def __enter__(self) -> BinaryIO:
        return self

    def __exit__(
        self, exc_type: Type[BaseException], exc_val: BaseException, exc_tb: TracebackType
    ) -> None:
        self.close()

    def close(self) -> None:
        self.stream.close()

    def isatty(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False


def get_itar_byte_offset(
    path: Union[str, EPath],
    sample_offset: int = 0,
) -> int:
    """Gets the byte offset from sample offsets."""
    if sample_offset == 0:
        return 0
    with TarIndexReader(path) as itar:
        return itar[sample_offset]


@edataclass
class CacheEntry:
    tar_index_reader: TarIndexReader
    lookahead_offset: Optional[int] = None
    lookahead_byteoffset: Optional[int] = None


class CachedItarOffsetReader:
    """
    This class is a high-level wrapper around TarIndexReader that caches some
    of the recent lookups for faster access. It is designed for the case when
    you need to read multiple offsets from the same tar file or from multiple
    tar files.

    Args:
        cache_size: The number of entries to keep in the cache. By default, we keep 32.
    """

    def __init__(self, cache_size: int = 32):
        # Maps (tar_file, current_offset) -> CacheEntry
        self.tar_index_reader_cache: Dict[Tuple[str, int], CacheEntry] = {}
        self.cache_size = cache_size

    def _find_or_create_entry(
        self,
        tar_file: Union[str, "EPath"],
        sample_offset: int,
    ) -> Tuple[Tuple[str, int], CacheEntry]:
        """
        1. If we already have a key == (tar_file, sample_offset), return it.
        2. Otherwise, create a new entry (and evict if necessary).
        """
        tar_file = str(tar_file)
        key = (tar_file, sample_offset)

        # Direct hit in the cache?
        if key in self.tar_index_reader_cache:
            return key, self.tar_index_reader_cache[key]

        # We didn't find an existing entry. Create a new one.
        # Evict if needed.
        if len(self.tar_index_reader_cache) >= self.cache_size:
            self._evict_one_entry()

        new_reader = TarIndexReader(tar_file)
        cache_entry = CacheEntry(tar_index_reader=new_reader)
        self.tar_index_reader_cache[key] = cache_entry
        return key, cache_entry

    def _evict_one_entry(self):
        """
        Evict the 'oldest' item in the cache. Here we just pop the first item
        returned by iter(...) in Python 3.7+ which *should* be insertion order,
        but not strictly an LRU. For true LRU, you can use OrderedDict or similar.
        """
        oldest_key = next(iter(self.tar_index_reader_cache))
        oldest_entry = self.tar_index_reader_cache.pop(oldest_key)
        oldest_entry.tar_index_reader.close()

    def _get_itar_byte_offset_with_entry(
        self,
        cache_entry: CacheEntry,
        sample_offset: int,
    ) -> Tuple[int, int]:
        """
        Return (start_byte_offset, length_to_next),
        possibly using per-entry lookahead for speed.
        """
        tar_index_reader = cache_entry.tar_index_reader

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
            result_byte_offset = tar_index_reader[sample_offset]

        # Prepare the lookahead for (sample_offset+1)
        next_offset = sample_offset + 1
        try:
            cache_entry.lookahead_byteoffset = tar_index_reader[next_offset]
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

    def get_itar_byte_offset(
        self,
        tar_file: Union[str, "EPath"],
        sample_offset: int = 0,
    ) -> Tuple[int, int]:
        """
        High-level API to get the byte offset and length for the given file & sample_offset.
        """

        # Find or create the suitable CacheEntry
        key, entry = self._find_or_create_entry(tar_file, sample_offset)

        # Use (and update) the per-entry lookahead logic
        result_byte_offset, length = self._get_itar_byte_offset_with_entry(entry, sample_offset)

        # Update cache entry with the new offset
        self.tar_index_reader_cache.pop(key)
        if entry.lookahead_offset is not None:
            new_key = (str(tar_file), entry.lookahead_offset)
            if new_key not in self.tar_index_reader_cache:
                self.tar_index_reader_cache[new_key] = entry
            else:
                # Already have this entry in the cache, so we can close the reader and use the existing one
                # TODO: We may actually may want to keep multiple readers open, because they may be multiple
                # sequences to the same sequence.
                entry.tar_index_reader.close()
        else:
            # No lookahead, so we can close the reader
            entry.tar_index_reader.close()

        return result_byte_offset, length


class ITarFile(tarfile.TarFile):
    """This class is a subclass of tarfile.TarFile that allows for reading a tarfile,
    with random access while keeping the file open.

    Usage:
        with open(filename, "rb") as fileobj:
            with ITarFile.open(fileobj=fileobj, mode="r:") as f:
                f.offset = 101888
                tarinfo = f.next()
                print(tarinfo.name)
                member_bytes = f.extractfile(tarinfo)

                # Read more offsets here ...
    """

    def __init__(self, *args, **kwargs):
        self.in_init = True
        try:
            super().__init__(*args, **kwargs)
        finally:
            self.in_init = False

    def next(self):
        if self.in_init:
            # Don't automatically read the first member
            return None

        if self.offset != self.fileobj.tell():
            # This prevents tarfile from reading the one byte before
            self.fileobj.seek(self.offset)

        return super().next()


@contextlib.contextmanager
def open_itar(path: Union[str, EPath], byte_offset: int = 0, byte_size: Optional[int] = None):
    """
    Open an indexed tarfile with offset and size.
    Args:
        path: Path to the tarfile to open
        byte_offset: Byte offset within the file
        byte_size: Size of the file to read

    Returns:
        The opened tarfile
    """
    path = EPath(path)

    # TODO: if tar file startswith(b"\x1f\x8b\x08") -> Seekable gzip file
    with path.open("rb") as f:
        if f.read(3) == b"\x1f\x8b\x08":
            # Open as seekable tgz
            raise ValueError("Seekable tgz not supported yet")

    if byte_offset != 0 or byte_size is not None:
        if byte_size is None:
            byte_size = path.size() - byte_offset
        with RetryReadStream(path) as stream:
            with SubFileReader(
                stream,
                offset=byte_offset,
                size=byte_size,
            ) as fileobj:
                with ITarFile.open(fileobj=fileobj, mode="r:") as f:
                    yield f
    else:
        with RetryReadStream(path) as fileobj:
            with ITarFile.open(fileobj=fileobj, mode="r:") as f:
                yield f
