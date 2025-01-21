# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import sqlite3
import struct
from typing import Optional, Union

from megatron.energon.epathlib import EPath


class SqliteIndexWriter:
    sqlite_path: EPath
    db: Optional[sqlite3.Connection]

    def __init__(self, sqlite_path: EPath):
        """
        Initializes an SQLite database and sets up two tables:
          - tar_files(id INTEGER PRIMARY KEY AUTOINCREMENT, tar_file_name TEXT UNIQUE)
          - samples(id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tar_file_id INTEGER,
                    sample_key TEXT,
                    sample_index INTEGER,
                    byte_offset INTEGER)
        Also creates an index on samples(sample_key).
        """

        # Final path and temporary path
        self.sqlite_path = sqlite_path

        # Initialize SQLite connection
        self.db = sqlite3.connect(str(self.sqlite_path))
        self.db.execute("PRAGMA journal_mode = WAL;")
        self.db.execute("PRAGMA busy_timeout = 5000;")  # wait up to 5000ms when locked

        # Create the tables
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS tar_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tar_file_name TEXT UNIQUE
            )
        """
        )
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tar_file_id INTEGER,
                sample_key TEXT,
                sample_index INTEGER,
                byte_offset INTEGER
            )
        """
        )
        # Index on sample_key for fast lookups
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_samples_sample_key ON samples(sample_key)")

        # We'll cache tar_file -> tar_file_id to avoid repeated lookups
        self._tar_file_cache = {}

    def append_sample(
        self, tar_file: str, sample_key: str, sample_index: int, byte_offset: Optional[int]
    ):
        """
        Adds a new sample row to the samples table, linking to the tar_files table.
        """

        assert self.db is not None, "Database is closed"

        # 1) Check if tar_file is in the cache
        if tar_file in self._tar_file_cache:
            tar_file_id = self._tar_file_cache[tar_file]
        else:
            # Insert the tar_file into tar_files if not already present
            # Using INSERT OR IGNORE, then we can fetch the rowid
            self.db.execute(
                "INSERT OR IGNORE INTO tar_files (tar_file_name) VALUES (?)", (tar_file,)
            )
            # Now fetch the ID
            cursor = self.db.execute(
                "SELECT id FROM tar_files WHERE tar_file_name = ?", (tar_file,)
            )
            tar_file_id = cursor.fetchone()[0]
            self._tar_file_cache[tar_file] = tar_file_id

        # 2) Insert a row in the samples table
        self.db.execute(
            """
            INSERT INTO samples (tar_file_id, sample_key, sample_index, byte_offset)
            VALUES (?, ?, ?, ?)
            """,
            (tar_file_id, sample_key, sample_index, byte_offset),
        )

    def close(self):
        """
        Closes the DB connection. If finalize=True, the temporary database is
        renamed to the final name, overwriting if necessary.
        """
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If an exception occurred, do not finalize (so you can inspect the temp file)
        self.close()
