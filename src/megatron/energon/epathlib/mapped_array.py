# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import BinaryIO, Protocol

import numpy as np

from megatron.energon.epathlib.epath import EPath


class EPathMappedArray(Protocol):
    """Array-like context manager over an EPath-backed file."""

    def __enter__(self) -> "EPathMappedArray": ...

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int | slice): ...

    def open(self) -> None: ...

    def close(self) -> None: ...


class _BaseEPathMappedArray:
    path: EPath
    dtype: np.dtype
    shape: tuple[int, ...]
    offset: int

    def __init__(
        self,
        path: EPath,
        *,
        dtype,
        shape: int | tuple[int, ...] | None = None,
        offset: int = 0,
    ) -> None:
        self.path = EPath(path)
        self.dtype = np.dtype(dtype)
        self.offset = offset
        if shape is None:
            assert self.path.is_file(), f"Path {self.path} is not a file"
            shape = (self.path.size() - offset) // self.dtype.itemsize
        if isinstance(shape, int):
            shape = (shape,)
        assert len(shape) == 1, "EPathMappedArray currently supports 1D arrays only"
        self.shape = shape

    def __enter__(self) -> "EPathMappedArray":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.shape[0]

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class EPathReadMappedArray(_BaseEPathMappedArray):
    """Seek/read implementation for EPath-backed array access."""

    _file: BinaryIO | None

    def __init__(
        self,
        path: EPath,
        *,
        dtype,
        shape: int | tuple[int, ...] | None = None,
        offset: int = 0,
    ) -> None:
        super().__init__(path, dtype=dtype, shape=shape, offset=offset)
        self._file = None

    def __getitem__(self, index: int | slice):
        self.open()
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if stop <= start:
                return np.empty((0,), dtype=self.dtype)
            assert self._file is not None
            self._file.seek(self.offset + start * self.dtype.itemsize)
            count = stop - start
            raw = self._file.read(count * self.dtype.itemsize)
            result = np.frombuffer(raw, dtype=self.dtype, count=count)
            if step != 1:
                result = result[::step]
            return result

        if index < 0:
            index += len(self)
        assert 0 <= index < len(self), f"Index {index} out of range [0, {len(self)})"
        assert self._file is not None
        self._file.seek(self.offset + index * self.dtype.itemsize)
        raw = self._file.read(self.dtype.itemsize)
        assert len(raw) == self.dtype.itemsize, (
            f"Short read from {self.path}: expected {self.dtype.itemsize} bytes, got {len(raw)}"
        )
        return np.frombuffer(raw, dtype=self.dtype, count=1)[0]

    def open(self) -> None:
        if self._file is not None:
            return
        self._file = self.path.open("rb")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class EPathNumpyMappedArray(_BaseEPathMappedArray):
    """NumPy memmap implementation for local EPath-backed array access."""

    _memmap: np.memmap | None

    def __init__(
        self,
        path: EPath,
        *,
        dtype,
        shape: int | tuple[int, ...] | None = None,
        offset: int = 0,
    ) -> None:
        super().__init__(path, dtype=dtype, shape=shape, offset=offset)
        self._memmap = None

    def __getitem__(self, index: int | slice):
        self.open()
        assert self._memmap is not None
        return self._memmap[index]

    def open(self) -> None:
        if self._memmap is not None:
            return
        self._memmap = np.memmap(
            self.path.local_path(),
            dtype=self.dtype,
            mode="r",
            offset=self.offset,
            shape=self.shape,
        )

    def close(self) -> None:
        if self._memmap is None:
            return
        memmap = self._memmap
        self._memmap = None
        mmap_obj = getattr(memmap, "_mmap", None)
        del memmap
        if mmap_obj is not None:
            mmap_obj.close()


def map_epath(
    path: EPath,
    *,
    dtype,
    shape: int | tuple[int, ...] | None = None,
    offset: int = 0,
    copy_to_local: bool = False,
    use_mmap: bool = True,
) -> EPathMappedArray:
    """Open an EPath-backed mapped array using mmap when possible, otherwise seek/read."""

    path = EPath(path)
    if use_mmap and (path.is_local() or copy_to_local):
        if copy_to_local and not path.is_local():
            from megatron.energon.local_copy import ensure_local_copy

            path = ensure_local_copy(path)
        result: EPathMappedArray = EPathNumpyMappedArray(
            path,
            dtype=dtype,
            shape=shape,
            offset=offset,
        )
    else:
        result = EPathReadMappedArray(
            path,
            dtype=dtype,
            shape=shape,
            offset=offset,
        )
    result.open()
    return result
