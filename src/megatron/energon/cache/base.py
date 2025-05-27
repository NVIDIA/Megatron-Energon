# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

from megatron.energon.edataclass import edataclass
from megatron.energon.source_info import SourceInfo, add_source_info

T = TypeVar("T")


class FileStore(Generic[T]):
    """Base type for a dataset that can be accessed randomly by sample key."""

    @abstractmethod
    def __getitem__(self, key: str) -> tuple[T, SourceInfo]:
        """Returns the data for the given key."""
        ...

    def get(self, key: str, sample: Any = None) -> Any:
        """Returns the data for the given key and adds the source info to the sample."""
        data, source_info = self[key]
        add_source_info(sample, source_info)
        return data

    @abstractmethod
    def get_path(self) -> str:
        """Returns the path to the dataset."""
        ...


@edataclass
class Lazy(Generic[T]):
    """
    Abstract base class for lazy references to data.
    """

    ds: FileStore
    fname: str
    pool: "CachePool"

    @abstractmethod
    def get(self, sample: Any = None) -> T:
        """
        Get the lazy data now and adds the source info to the sample.
        """
        ...

    def __hash__(self) -> int:
        """Allows usage in sets and dicts as key."""
        return hash((id(self.ds), self.fname))

    def __eq__(self, other: Any) -> bool:
        """Allows usage in sets and dicts as key. Compares the data source and the filename."""
        if not isinstance(other, Lazy):
            return False
        return self.ds is other.ds and self.fname == other.fname


@edataclass
class MockLazy(Lazy[T]):
    """
    Mock object, which can be used as a Lazy. Allows the user to set the function to retrieve the
    data. May be used to create a Lazy that is initialized from a function.
    """

    ds: FileStore
    fname: str
    pool: "CachePool"

    get_fn: Callable[[str], T]

    def __init__(self, fname: str, get_fn: Callable[[str], T]):
        """
        Initialize the MockLazy object.

        Args:
            fname: The file name of the mock object (may be used by the user).
            get_fn: The function to retrieve/generate the data.
        """
        self.ds = None
        self.fname = fname
        self.pool = None
        self.get_fn = get_fn

    def get(self, sample: Any = None) -> T:
        """
        Get the lazy data now and adds no source info to the sample.
        """
        return self.get_fn(self.fname)

    def __hash__(self) -> int:
        return hash((self.fname, self.get_fn))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MockLazy):
            return False
        return self.fname == other.fname and self.get_fn == other.get_fn

    def __repr__(self) -> str:
        return f"MockLazy(fname={self.fname!r}, get_fn={self.get_fn!r})"


class CachePool(ABC):
    """
    A cache pool allows to load the needed data in the background and access it later.
    The most important example being `FileStoreCachePool` which caches data on a local SSD disk.

    To request data, use `get_lazy` to get a `Lazy` object.
    Then, call `Lazy.get()` to get the data later on.
    """

    @abstractmethod
    def get(self, ds: FileStore, fname: str, sample: Any = None) -> Any:
        """
        Get the data for a given file and adds the source info to the sample.
        """
        ...

    @abstractmethod
    def get_lazy(self, ds: FileStore, fname: str) -> Lazy:
        """
        Get a lazy reference to the data for a given file.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Close the cache pool.
        """
        ...


class FileStoreDecoder(ABC):
    """
    Abstract base class for decoders.
    """

    @abstractmethod
    def decode(self, fname: str, data: bytes) -> Any:
        """
        Decode the specified file (i.e. path/key.ext). The extension is used to select the decoder.

        Args:
            fname: The file name of the file to decode.
            raw: The raw bytes of the file to decode.

        Returns:
            The decoded field's data.
        """
        ...
