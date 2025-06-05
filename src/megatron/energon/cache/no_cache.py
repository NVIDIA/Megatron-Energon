# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional, TypeVar

from megatron.energon.cache.base import CachePool, FileStore, Lazy
from megatron.energon.edataclass import edataclass
from megatron.energon.source_info import SourceInfo, add_source_info

T = TypeVar("T")


@edataclass
class DirectLazy(Lazy[T]):
    """
    This is not really lazy, it will just defer the dataset access to the first get().
    """

    _data: Optional[tuple[T, SourceInfo]] = None

    def get(self, sample: Any = None) -> T:
        """Get the lazy data now and adds no source info to the sample."""
        if self._data is None:
            self._data = self.ds[self.fname]
        assert self._data is not None
        add_source_info(sample, self._data[1])
        return self._data[0]

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

    def get(self, ds: FileStore, fname: str, sample: Any = None) -> Any:
        """Get the data for a given file and adds the source info to the sample."""
        return ds.get(fname, sample)

    def get_lazy(self, ds: FileStore, fname: str) -> DirectLazy:
        return DirectLazy(ds=ds, fname=fname, pool=self)

    def close(self) -> None:
        pass
