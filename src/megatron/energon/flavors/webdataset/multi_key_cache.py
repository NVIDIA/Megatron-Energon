# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Generator, Generic, TypeVar, overload

T_key = TypeVar("T_key")
T_value = TypeVar("T_value")


class MultiKeyCache(Generic[T_key, T_value]):
    """A cache that can store multiple values for the same key."""

    _size: int
    _cache: dict[T_key, list[T_value]]
    _lru_keys: list[T_key]

    def __init__(self) -> None:
        self._size = 0
        self._cache = {}
        self._lru_keys = []

    @overload
    def pop(self, key: None = None) -> T_value: ...

    @overload
    def pop(self, key: T_key) -> T_value | None: ...

    def pop(self, key: T_key | None = None) -> T_value | None:
        """Pop the value for the given key from the cache.

        If no key is provided, pop the oldest key from the cache.

        Args:
            key: The key to pop from the cache. If None, pop the oldest key from the cache.

        Returns:
            The value popped from the cache.
        """
        if key is None:
            key = self._lru_keys.pop(0)
        elif key not in self._cache:
            return None
        else:
            self._lru_keys.pop(len(self._lru_keys) - 1 - self._lru_keys[::-1].index(key))

        l = self._cache[key]
        value = l.pop(0)
        if len(l) == 0:
            del self._cache[key]
        self._size -= 1
        return value

    def add(self, key: T_key, value: T_value) -> None:
        """Add a value to the cache."""
        if key not in self._cache:
            self._cache[key] = [value]
        else:
            self._cache[key].insert(0, value)

        self._lru_keys.append(key)
        self._size += 1

    def flush(self) -> Generator[T_value, None, None]:
        """Flush the cache."""
        for values in self._cache.values():
            for value in values:
                yield value
        self._cache.clear()
        self._lru_keys.clear()
        self._size = 0

    def __len__(self) -> int:
        return self._size
