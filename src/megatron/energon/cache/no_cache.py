# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional, TypeVar

from megatron.energon.cache.base import CachePool, FileStore, Lazy
from megatron.energon.dataclass_slots import dataclass_slots

T = TypeVar("T")


@dataclass_slots
class DirectLazy(Lazy[T]):
    """
    This is not really lazy, it will just defer the dataset access to the first get().
    """

    _data: Optional[T] = None

    def get(self) -> T:
        if self._data is None:
            self._data = self.pool.get(self.ds, self.fname)
        assert self._data is not None
        return self._data

    def __hash__(self) -> int:
        """Allows usage in sets and dicts as key."""
        return hash((id(self.ds), self.fname))

    def __eq__(self, other: Any) -> bool:
        """Allows usage in sets and dicts as key. Compares the data source and the filename."""
        if not isinstance(other, Lazy):
            return False
        return self.ds is other.ds and self.fname == other.fname


class NoCachePool(CachePool):
    """A pass-through cache pool that does not cache anything."""

    def get(self, ds: FileStore, fname: str) -> Any:
        return ds[fname]

    def get_lazy(self, ds: FileStore, fname: str) -> DirectLazy:
        return DirectLazy(ds=ds, fname=fname, pool=self)

    def close(self) -> None:
        pass
