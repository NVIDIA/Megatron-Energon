# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

from megatron.energon.dataclass_slots import dataclass_slots

T = TypeVar("T")


class FileStore(Generic[T]):
    """Base type for a dataset that can be accessed randomly by sample key."""

    @abstractmethod
    def __getitem__(self, key: str) -> T:
        """Returns the data for the given key."""
        ...

    @abstractmethod
    def get_path(self) -> str:
        """Returns the path to the dataset."""
        ...


@dataclass_slots
class Lazy(Generic[T]):
    """
    Abstract base class for lazy references to data.
    """

    ds: FileStore
    fname: str
    pool: "CachePool"

    @abstractmethod
    def get(self) -> T:
        """
        Get the lazy data now.
        """
        ...


@dataclass_slots
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

    def get(self) -> T:
        """
        Get the lazy data now.
        """
        return self.get_fn(self.fname)


class CachePool(ABC):
    """
    A cache pool allows to load the needed data in the background and access it later.
    The most important example being `FileStoreCachePool` which caches data on a local SSD disk.

    To request data, use `get_lazy` to get a `Lazy` object.
    Then, call `Lazy.get()` to get the data later on.
    """

    @abstractmethod
    def get(self, ds: FileStore, fname: str) -> Any:
        """
        Get the data for a given file.
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
