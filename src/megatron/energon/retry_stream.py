# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from types import TracebackType
from typing import BinaryIO, Optional, Type

from megatron.energon.epathlib import EPath


class RetryReadStream(BinaryIO):
    """A stream that retries reading from a file. Only supports reading bytes."""

    _path: EPath
    _file: Optional[BinaryIO]
    _pos: int
    _size: int

    def __init__(self, path: EPath):
        """Construct a RetryReadStream. It reads only bytes from a file."""
        self._path = path
        self._file = None
        self._pos = 0
        self._size = path.size()

    def __enter__(self) -> "RetryReadStream":
        return self

    def __exit__(
        self, exc_type: Type[BaseException], exc_val: BaseException, exc_tb: TracebackType
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()

    def read(self, n: int = -1) -> bytes:
        """Read up to n bytes, preserving short reads across at most ten I/O failures."""
        buf = bytearray()
        failures = 0
        while True:
            try:
                if self._file is None:
                    self._file = self._path.open("rb")
                    self._file.seek(self._pos)
                remaining = n - len(buf) if n >= 0 else -1
                res = self._file.read(remaining)
                self._pos += len(res)
                buf.extend(res)
                if not res or (n >= 0 and len(buf) == n) or self._pos >= self._size:
                    return bytes(buf)
            except IOError:
                if self._file is not None:
                    try:
                        self._file.close()
                    except IOError:
                        pass
                self._file = None
                failures += 1
                if failures == 10:
                    raise

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            pass
        elif whence == 1:
            offset += self._pos
        elif whence == 2:
            offset += self._size
        else:
            raise ValueError(f"Invalid whence value: {whence}")
        offset = min(max(offset, 0), self._size)
        self._pos = offset
        try:
            if self._file is not None:
                self._file.seek(offset)
        except IOError:
            pass
        return self._pos

    def tell(self) -> int:
        return self._pos

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False
