# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import os
import random
import sqlite3
import threading
import time
from typing import Any, ClassVar


class ThreadLocalStorage:
    """
    A class that allows to store data in a thread-local storage.

    Example Usage:
    ```python
    class MyThreadLocalStorage(ThreadLocalStorage):
        __thread_local__ = ("my_data",)

        # This is shared across threads
        other_data: int

        # This is local per thread
        my_data: int

        def __thread_init__(self):
            # This is called when the data on a thread is initialized, which has
            # not been accessed yet on that thread to set the value of that data.
            self.my_data = 0

    ```
    """

    __thread_local__: ClassVar[tuple[str, ...]]
    _storage: object

    def __init__(self):
        self._storage = threading.local()

    def __getattribute__(self, name: str) -> Any:
        if name in ("__thread_local__", "_storage"):
            return object.__getattribute__(self, name)
        if name in self.__thread_local__:
            if not self._thread_initialized:
                self._storage.__initialized__ = True
                self.__thread_init__()

            return getattr(self._storage, name)
        return object.__getattribute__(self, name)

    def __delattr__(self, name: str) -> None:
        if name in self.__thread_local__:
            delattr(self._storage, name)
            return
        object.__delattr__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__thread_local__:
            if not self._thread_initialized:
                self._storage.__initialized__ = True
                self.__thread_init__()
            setattr(self._storage, name, value)
            return
        object.__setattr__(self, name, value)

    @property
    def _thread_initialized(self) -> bool:
        """Check if the thread has been initialized."""
        return getattr(self._storage, "__initialized__", False)

    def thread_close(self):
        """Close the thread-local storage."""
        if self._thread_initialized:
            delattr(self._storage, "__initialized__")

    def __thread_init__(self):
        """Called when the data on a thread is accessed for the first time, to
        set the initial value of that data."""
        # Copy the data from the default values
        for name in self.__thread_local__:
            try:
                default_value = object.__getattribute__(self, name)
            except AttributeError:
                pass
            else:
                setattr(self._storage, name, default_value)


class ThreadLocalSqlite(ThreadLocalStorage):
    """A class that allows to store data in a thread-local storage."""

    database: str

    __thread_local__ = ("connection", "cursor")

    connection: sqlite3.Connection
    cursor: sqlite3.Cursor

    def __init__(self, database: str):
        super().__init__()
        self.database = database

    def __thread_init__(self):
        """Initialize the connection and cursor."""
        self.connection = sqlite3.connect(self.database)
        self.cursor = self.connection.cursor()
        self.connection.execute("PRAGMA busy_timeout = 5000;")

    def select_one(self, query: str, params: tuple[Any, ...] = ()):
        """Select one row from the database."""
        self.cursor.execute(query, params)
        return self.cursor.fetchone()

    def select_all(self, query: str, params: tuple[Any, ...] = ()):
        """Select all rows from the database."""
        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def thread_close(self):
        """Close the connection and cursor."""
        if self._thread_initialized:
            self.cursor.close()
            self.connection.close()
            super().thread_close()


def main():
    """Test the ThreadLocalSqlite class."""
    import concurrent.futures

    sqlite = ThreadLocalSqlite("tmp.sqlite")
    sqlite.cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)")
    sqlite.cursor.execute("INSERT INTO test (name) VALUES (?)", ("test",))
    sqlite.connection.commit()

    def _test_thread_local(sqlite_thread_local: ThreadLocalSqlite):
        time.sleep(random.random())
        print(sqlite_thread_local.select_all("SELECT * FROM test"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for _ in range(20):
            futures.append(executor.submit(_test_thread_local, sqlite))
        for future in concurrent.futures.as_completed(futures):
            future.result()

    os.remove("tmp.sqlite")


if __name__ == "__main__":
    main()
