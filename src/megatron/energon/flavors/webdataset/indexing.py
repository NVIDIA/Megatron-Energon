# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import sqlite3
import struct
from pathlib import Path
from typing import BinaryIO, Generator, List, Optional, Tuple, Union

from numpy import int8

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.itar import ITarRawSamplePartPointer, ITarSamplePointer
from megatron.energon.flavors.webdataset.thread_local_sqlite import ThreadLocalSqlite
from megatron.energon.local_copy import ensure_local_copy


class DuplicateSampleKeyError(RuntimeError):
    """Raised when attempting to insert a duplicate sample key into the index."""

    def __init__(self, sample_key: str) -> None:
        super().__init__(f"Duplicate sample key encountered while indexing: {sample_key!r}")
        self.sample_key = sample_key


class SqliteIndexWriter:
    sqlite_path: EPath
    db: Optional[sqlite3.Connection]
    enable_sample_tables: bool
    enable_media_metadata: bool
    reset_tables: bool

    def __init__(
        self,
        sqlite_path: EPath,
        *,
        enable_sample_tables: bool = True,
        enable_media_metadata: bool = False,
        reset_tables: bool = True,
    ):
        """
        Initializes an SQLite database and sets up the samples table:
          - samples(tar_file_id INTEGER,
                    sample_key TEXT,
                    sample_index INTEGER,
                    byte_offset INTEGER,
                    byte_size INTEGER)
        and the sample_parts table:
          - sample_parts(tar_file_id INTEGER,
                         sample_index INTEGER,
                         part_name TEXT,
                         content_byte_offset INTEGER,
                         content_byte_size INTEGER)
        Also creates indexes:
          - samples(sample_key)
          - samples(tar_file_id, sample_index)
          - sample_parts(tar_file_id, sample_index, content_byte_offset)
          - sample_parts(tar_file_id, sample_index, part_name, content_byte_offset, content_byte_size)
        """

        # Final path and temporary path
        self.sqlite_path = sqlite_path
        self.enable_sample_tables = enable_sample_tables
        self.enable_media_metadata = enable_media_metadata
        self.reset_tables = reset_tables

        # Initialize SQLite connection
        path = str(self.sqlite_path)
        # Only supporting local file system, because sqlite does not support remote file systems.
        # TODO: Implement remote file systems. Maybe create locally in tmp then upload?
        assert path.startswith("/"), (
            f"SQLite path must be absolute local file system path: {self.sqlite_path}"
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA busy_timeout = 5000;")  # wait up to 5000ms when locked

        if self.enable_sample_tables:
            assert self.reset_tables, "Reset tables is required when enabling sample tables"

            self.db.execute("DROP INDEX IF EXISTS idx_samples_sample_key")
            self.db.execute("DROP INDEX IF EXISTS idx_samples_by_tar_and_idx")
            self.db.execute("DROP TABLE IF EXISTS samples")

            self.db.execute("DROP INDEX IF EXISTS idx_sample_parts_seq")
            self.db.execute("DROP INDEX IF EXISTS idx_sample_parts_full")
            self.db.execute("DROP TABLE IF EXISTS sample_parts")

            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    tar_file_id INTEGER NOT NULL,
                    sample_key TEXT NOT NULL UNIQUE,
                    sample_index INTEGER NOT NULL,
                    byte_offset INTEGER,
                    byte_size INTEGER
                )
                """
            )
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS sample_parts (
                    tar_file_id INTEGER,
                    sample_index INTEGER,
                    part_name TEXT,
                    content_byte_offset INTEGER,
                    content_byte_size INTEGER
                )
                """
            )

        if self.enable_media_metadata:
            if self.reset_tables:
                self.db.execute("DROP TABLE IF EXISTS media_metadata")
                self.db.execute("DROP TABLE IF EXISTS media_filters")
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS media_metadata (
                    entry_key TEXT PRIMARY KEY,
                    metadata_type TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS media_filters (
                    filter_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    patterns TEXT,
                    created_at_utc TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(strategy, patterns)
                )
                """
            )

    def append_sample(
        self,
        tar_file_id: int8,
        sample_key: str,
        sample_index: int,
        byte_offset: Optional[int],
        byte_size: Optional[int],
    ):
        """
        Adds a new sample row to the samples table.

        Args:
            tar_file_id: The index of the tar file in the reader.
            sample_key: The key of the sample.
            sample_index: The index of the sample in the tar file.
            byte_offset: The byte offset of the sample in the tar file.
            byte_size: The size of the sample in the tar file.
        """

        assert self.db is not None, "Database is closed"

        # Insert a row in the samples table
        try:
            self.db.execute(
                """
                INSERT INTO samples (tar_file_id, sample_key, sample_index, byte_offset, byte_size)
                VALUES (?, ?, ?, ?, ?)
                """,
                (tar_file_id, sample_key, sample_index, byte_offset, byte_size),
            )
        except sqlite3.IntegrityError as exc:  # pragma: no cover - defensive programming
            raise DuplicateSampleKeyError(sample_key) from exc

    def append_part(
        self,
        tar_file_id: int8,
        sample_index: int,
        part_name: str,
        content_byte_offset: int,
        content_byte_size: int,
    ):
        """Adds a new part row to the samples table."""

        assert self.db is not None, "Database is closed"

        # Insert a row in the sample parts table
        self.db.execute(
            """
            INSERT INTO sample_parts (tar_file_id, sample_index, part_name, content_byte_offset, content_byte_size)
            VALUES (?, ?, ?, ?, ?)
            """,
            (tar_file_id, sample_index, part_name, content_byte_offset, content_byte_size),
        )

    def append_media_metadata(
        self,
        entry_key: str,
        metadata_type: str,
        metadata_json: str,
    ) -> None:
        """Insert or update a media metadata record."""

        assert self.enable_media_metadata, "Adding media metadata, although not enabled"

        assert self.db is not None, "Database is closed"

        self.db.execute(
            """
            INSERT OR REPLACE INTO media_metadata (entry_key, metadata_type, metadata_json)
            VALUES (?, ?, ?)
            """,
            (entry_key, metadata_type, metadata_json),
        )

    def append_media_filter(self, *, strategy: str, patterns: str | None) -> None:
        assert self.db is not None, "Database is closed"
        self.db.execute(
            "INSERT OR IGNORE INTO media_filters (strategy, patterns) VALUES (?, ?)",
            (strategy, patterns),
        )

    def close(self):
        """
        Closes the DB connection. If finalize=True, the temporary database is
        renamed to the final name, overwriting if necessary.
        """
        assert self.db is not None, "Database is closed"

        if self.enable_sample_tables:
            # Create the index after adding all the samples for better speed
            # Index on sample_key for fast lookups
            self.db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_samples_sample_key ON samples(sample_key)"
            )

            # Create index on the samples table.  Help the planner if it chooses `samples` as the probe side of the join
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_by_tar_and_idx ON samples(tar_file_id, sample_index)"
            )

            # Create index on the sample_parts table for fast sequential access
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_sample_parts_seq ON sample_parts(tar_file_id, sample_index, content_byte_offset)"
            )

            # Create a full index on the sample_parts table for equality lookups and getting offsets directly from key
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_sample_parts_full ON sample_parts(tar_file_id, sample_index, part_name, content_byte_offset, content_byte_size)"
            )

        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If an exception occurred, do not finalize (so you can inspect the temp file)
        self.close()


class JoinIndexWriter:
    """Describes how one primary dataset is joined with multiple secondary datasets.

    For fast random access, this is a binary format that is memory-mapped.
    The first 16 bytes are a header with the number of columns (1 primary + N secondary).
    Each row contains (shard_idx, byte_offset, byte_size) for each column.
    """

    def __init__(self, join_index_path: EPath):
        self.join_index_path = join_index_path
        self.join_index_file = join_index_path.open("wb")
        self.num_columns = None

    def append(self, *columns: Tuple[int, int, int]):
        """Appends a new row to the join index file.

        Each row contains (shard_idx, byte_offset, byte_size) for each column.
        """

        if self.num_columns is None:
            # Write the number of columns
            self.join_index_file.write(b"JIDX0001")  # Magic bytes with version
            self.join_index_file.write(struct.pack("q", len(columns)))
            self.num_columns = len(columns)
        else:
            assert len(columns) == self.num_columns, (
                f"Inconsistent number of keys: Had {self.num_columns} before, got {len(columns)}"
            )

        # Write the columns
        for key in columns:
            assert isinstance(key, tuple) and len(key) == 3
            self.join_index_file.write(struct.pack("qqq", *key))

    def close(self):
        self.join_index_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SqliteIndexReader:
    """Reads samples from an SQLite database created by SqliteIndexWriter.

    The database contains a table with the following schema:
    - samples(tar_file_id INTEGER,
              sample_key TEXT,
              sample_index INTEGER,
              byte_offset INTEGER,
              byte_size INTEGER)
    - sample_parts(tar_file_id INTEGER,
                   sample_index INTEGER,
                   part_name TEXT,
                   content_byte_offset INTEGER,
                   content_byte_size INTEGER)
    """

    sqlite_path: EPath
    db: ThreadLocalSqlite

    def __init__(self, sqlite_path: EPath):
        """Initialize the SQLite database reader.

        Args:
            sqlite_path: Path to the SQLite database file
        """
        self.sqlite_path = ensure_local_copy(sqlite_path)

        # Initialize SQLite connection
        path = str(self.sqlite_path)
        # Only supporting local file system, because sqlite does not support remote file systems
        assert path.startswith("/"), (
            f"SQLite path must be absolute local file system path: {self.sqlite_path}"
        )

        path = f"file:{path}?mode=ro&immutable=1"

        self.db = ThreadLocalSqlite(path, is_uri=True)

    def db_has_sample_parts(self) -> bool:
        """Check if the database has a sample_parts table.

        Returns:
            True if sample_parts table exists, False otherwise.
        """
        assert self.db is not None, "Database is closed"

        db_exists = self.db.select_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sample_parts'"
        )
        self.db.thread_close()
        return db_exists is not None

    def db_has_media_metadata(self) -> bool:
        """Check if the database has a media_metadata table."""

        assert self.db is not None, "Database is closed"

        db_exists = self.db.select_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='media_metadata'"
        )
        self.db.thread_close()
        return db_exists is not None

    def list_all_samples(self) -> Generator[Tuple[str, int, int], None, None]:
        """List all sample keys in the database.

        Returns:
            Tuple of (sample_key, byte_size)
        """

        assert self.db is not None, "Database is closed"

        for row in self.db.select_all("SELECT sample_key, byte_size, tar_file_id FROM samples"):
            yield row[0], row[1], row[2]

    def list_all_sample_parts(self) -> Generator[Tuple[str, int, int], None, None]:
        """List all sample parts (i.e. individual files) in the database.

        Returns:
            Tuple of (full_key, size, tar_file_id)
        """

        assert self.db is not None, "Database is closed"

        # Select all parts (sorted by tar_file_id, sample_index) but joined with the sample_key names
        for row in self.db.select_all(
            "SELECT "
            "s.sample_key || '.' || sp.part_name AS full_key, "
            "sp.content_byte_size AS size, "
            "sp.tar_file_id AS tar_file_id "
            "FROM sample_parts AS sp "
            "JOIN samples AS s "
            "ON sp.tar_file_id  = s.tar_file_id AND sp.sample_index = s.sample_index "
            "ORDER BY sp.tar_file_id, sp.sample_index, sp.content_byte_offset"
        ):
            yield row[0], row[1], row[2]

    def list_sample_parts(self, sample_key: str) -> Generator[Tuple[str, int, int], None, None]:
        """List all sample parts (i.e. individual files) in the database.

        Args:
            sample_key: The sample key to look up

        Returns:
            Tuple of (part_name, size, tar_file_id)
        """

        assert self.db is not None, "Database is closed"

        # Select all parts (sorted by tar_file_id, sample_index) but joined with the sample_key names
        for row in self.db.select_all(
            "SELECT "
            "sp.part_name AS part_name, "
            "sp.content_byte_size AS size, "
            "sp.tar_file_id AS tar_file_id "
            "FROM sample_parts AS sp "
            "JOIN samples AS s "
            "ON sp.tar_file_id  = s.tar_file_id AND sp.sample_index = s.sample_index "
            "WHERE s.sample_key = ? "
            "ORDER BY sp.tar_file_id, sp.sample_index, sp.content_byte_offset",
            (sample_key,),
        ):
            yield row[0], row[1], row[2]

    def get_total_size(self) -> int:
        """Get the total size of all samples in the database."""
        assert self.db is not None, "Database is closed"

        count = self.db.select_one("SELECT SUM(byte_size) FROM samples")
        return count[0] if count else 0

    def get_sample_count(self) -> int:
        """Get the total number of samples in the database."""
        assert self.db is not None, "Database is closed"

        count = self.db.select_one("SELECT COUNT(*) FROM samples")
        return count[0] if count else 0

    def get_sample_part(self, key: str, part_name: str) -> ITarRawSamplePartPointer:
        """Get a sample part by its key name and part name.

        Args:
            key: The sample key to look up
            part_name: The part name to look up

        Returns:
            Pointer to the sample part raw data.
        """
        assert self.db is not None, "Database is closed"

        row = self.db.select_one(
            "SELECT sp.tar_file_id, sp.content_byte_offset, sp.content_byte_size "
            "FROM sample_parts AS sp "
            "JOIN samples AS s "
            "ON sp.tar_file_id = s.tar_file_id AND sp.sample_index = s.sample_index "
            "WHERE s.sample_key = ? AND sp.part_name = ?",
            (key, part_name),
        )
        if row is None:
            raise KeyError(
                f"Sample part not found: key={key}, part_name={part_name} in {self.sqlite_path}"
            )
        return ITarRawSamplePartPointer(
            tar_file_id=row[0],
            raw_byte_offset=row[1],
            raw_byte_size=row[2],
        )

    def get_sample_pointer_by_key(self, key: str) -> ITarSamplePointer:
        """Get a sample by its key name.

        Args:
            key: The sample key to look up

        Returns:
            Tuple of (tar_file_id, sample_key, sample_index, byte_offset, byte_size)
        """
        assert self.db is not None, "Database is closed"

        sample = self.db.select_one(
            "SELECT tar_file_id, sample_key, sample_index, byte_offset, byte_size "
            "FROM samples WHERE sample_key = ?",
            (key,),
        )

        if sample is None:
            raise KeyError(f"Sample key not found: {key}")

        return ITarSamplePointer(
            tar_file_id=sample[0],
            byte_offset=sample[3],
            byte_size=sample[4],
        )

    def get_media_metadata(self, entry_key: str) -> Tuple[str, str] | None:
        """Fetch the media metadata record for an entry, if available."""

        assert self.db is not None, "Database is closed"

        row = self.db.select_one(
            "SELECT metadata_type, metadata_json FROM media_metadata WHERE entry_key = ?",
            (entry_key,),
        )
        return (row[0], row[1]) if row is not None else None

    def close(self):
        """Close the database connection."""
        if self.db is not None:
            self.db.thread_close()
            del self.db

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class JoinIndexReader:
    """Reads a join index file in different ways.

    If a column is specified, only that column is read, otherwise the full rows.
    You can iterate over the rows, or read a specific row by index, or get the full tensor.

    Each row contains (shard_idx, byte_offset, byte_size) for each column.
    """

    join_index_path: EPath
    join_index_file: BinaryIO
    column: Optional[int]
    num_columns: int
    has_iterated: bool
    index_row_position: int

    def __init__(self, join_index_path: EPath, column: Optional[int] = None):
        self.join_index_path = join_index_path
        self.join_index_byte_size = join_index_path.size()

        self.column = column

        self.join_index_file = join_index_path.open("rb")
        self.has_iterated = False
        self.index_row_position = -1

        # Read the header
        bytes_magic = self.join_index_file.read(8)
        assert isinstance(bytes_magic, bytes)
        assert bytes_magic[:4] == b"JIDX", f"Invalid magic bytes: {bytes_magic}"
        assert bytes_magic[4:8] == b"0001", f"Unsupported version: {bytes_magic[4:8]}"

        # Read the number of columns
        bytes_seckeys = self.join_index_file.read(8)
        assert isinstance(bytes_seckeys, bytes)
        self.num_columns = struct.unpack("q", bytes_seckeys)[0]

        self.index_row_position = 0

    def get_as_tensor(self):
        """Returns the join index as a tensor with shape (N, num_columns, 3)."""

        assert not self.has_iterated, "Cannot get_as_tensor after iterating"

        import torch

        # Read the raw bytes for all N * 3 int64s.
        data = self.join_index_file.read()
        self.index_file_position = self.join_index_file.tell()
        assert len(data) % (8 * 3) == 0, (
            f"Index file reading: Expected multiple of 3 * 8 bytes, got {len(data)} bytes"
        )

        return torch.frombuffer(data, dtype=torch.int64).view(-1, self.num_columns, 3)

    def __len__(self):
        return (self.join_index_byte_size - 16) // (self.num_columns * 8 * 3)

    def __iter__(self):
        return self

    def _read_one_row(
        self, column: Optional[int] = None
    ) -> Union[None, List[Tuple[int, int, int]]]:
        row = []
        for col_idx in range(self.num_columns):
            if column is not None and col_idx != column:
                # Skip this column
                self.join_index_file.seek(8 * 3, 1)
                continue

            bytes_key = self.join_index_file.read(8 * 3)
            if not bytes_key:
                return None
            assert isinstance(bytes_key, bytes)
            key_tuple = struct.unpack("qqq", bytes_key)
            row.append(key_tuple)
        self.index_row_position += 1
        return row

    def __next__(self) -> Union[None, List[Tuple[int, int, int]]]:
        self.has_iterated = True
        return self._read_one_row(column=self.column)

    def tell_row(self) -> int:
        return self.index_row_position

    def __getitem__(self, idx: int) -> List[Tuple[int, int, int]]:
        """Reads the idx-th row of the join index file."""

        assert 0 <= idx < len(self), f"Index out of bounds: {idx} not in [0, {len(self)})"

        # Seek to the correct position

        if self.index_row_position != idx:
            self.join_index_file.seek(16 + idx * self.num_columns * 8 * 3)
            self.index_row_position = idx

        # Read the secondary keys
        row = self._read_one_row(column=self.column)
        assert row is not None, f"Failed to read row {idx}"

        return row

    def close(self):
        self.join_index_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
