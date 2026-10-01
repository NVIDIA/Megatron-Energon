# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import sqlite3
from collections.abc import Sequence
from typing import TYPE_CHECKING, Generator, Optional, Tuple

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.common.sqlite import ThreadLocalSqlite
from megatron.energon.flavors.webdataset.itar import ITarRawSamplePartPointer, ITarSamplePointer
from megatron.energon.local_copy import ensure_local_copy

if TYPE_CHECKING:
    from megatron.energon.flavors.webdataset.prepare import (
        IndexMediaMetadata,
        IndexSample,
        IndexSamplePart,
    )


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
        if enable_media_metadata is True, it also creates the media_metadata table:
          - media_metadata(entry_key TEXT PRIMARY KEY,
                           metadata_type TEXT NOT NULL,
                           metadata_json TEXT NOT NULL)
        if enable_media_metadata is True, it also creates the media_filters table:
          - media_filters(filter_id INTEGER PRIMARY KEY AUTOINCREMENT,
                          strategy TEXT NOT NULL,
                          patterns TEXT,
                          created_at_utc TEXT DEFAULT CURRENT_TIMESTAMP,
                          UNIQUE(strategy, patterns))
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
        # Only supporting local file system, because sqlite does not support remote file systems.
        path = self.sqlite_path.local_path()
        path.parent.mkdir(parents=True, exist_ok=True)
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

    def append_samples(
        self,
        rows: Sequence["IndexSample"],
    ) -> None:
        """Insert multiple sample rows efficiently."""

        assert self.db is not None, "Database is closed"

        if len(rows) == 0:
            return

        savepoint_name = "append_samples_batch"
        self.db.execute(f"SAVEPOINT {savepoint_name}")
        try:
            # One executemany() is substantially cheaper than one execute() per row.
            self.db.executemany(
                """
                INSERT INTO samples (tar_file_id, sample_key, sample_index, byte_offset, byte_size)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        row.tar_file_id,
                        row.sample_key,
                        row.sample_index,
                        row.byte_offset,
                        row.byte_size,
                    )
                    for row in rows
                ),
            )
            self.db.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        except sqlite3.IntegrityError as exc:
            # executemany() may already have inserted earlier rows in the batch
            self.db.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            self.db.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            raise DuplicateSampleKeyError(self._find_duplicate_sample_key(rows)) from exc

    def append_parts(
        self,
        rows: Sequence["IndexSamplePart"],
    ) -> None:
        """Insert multiple sample part rows efficiently."""

        assert self.db is not None, "Database is closed"

        if len(rows) == 0:
            return

        self.db.executemany(
            """
            INSERT INTO sample_parts (tar_file_id, sample_index, part_name, content_byte_offset, content_byte_size)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                (
                    row.tar_file_id,
                    row.sample_index,
                    row.part_name,
                    row.content_byte_offset,
                    row.content_byte_size,
                )
                for row in rows
            ),
        )

    def append_media_metadata_batch(
        self,
        rows: Sequence["IndexMediaMetadata"],
    ) -> None:
        """Insert or update multiple media metadata records efficiently."""

        assert self.enable_media_metadata, "Adding media metadata, although not enabled"
        assert self.db is not None, "Database is closed"

        if len(rows) == 0:
            return

        self.db.executemany(
            """
            INSERT OR REPLACE INTO media_metadata (entry_key, metadata_type, metadata_json)
            VALUES (?, ?, ?)
            """,
            ((row.entry_key, row.metadata_type, row.metadata_json) for row in rows),
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
            # sample_key uniqueness already creates an implicit SQLite index via the table schema.

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

    def _find_duplicate_sample_key(
        self,
        rows: Sequence["IndexSample"],
    ) -> str:
        """Resolve the sample key responsible for a uniqueness violation.

        sqlite3 only tells us that *some* row in the batch violated the unique
        constraint. We do a targeted follow-up lookup so the caller still gets
        the concrete duplicate key in the raised error.
        """

        assert self.db is not None, "Database is closed"

        seen_in_batch: set[str] = set()
        for row in rows:
            sample_key = row.sample_key
            if sample_key in seen_in_batch:
                return sample_key

            existing = self.db.execute(
                "SELECT 1 FROM samples WHERE sample_key = ?",
                (sample_key,),
            ).fetchone()
            if existing is not None:
                return sample_key

            seen_in_batch.add(sample_key)

        return rows[0].sample_key


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
    - media_metadata(entry_key TEXT PRIMARY KEY,
                     metadata_type TEXT NOT NULL,
                     metadata_json TEXT NOT NULL)
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
        # Only supporting local file system, because sqlite does not support remote file systems
        path = self.sqlite_path.local_path()
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
