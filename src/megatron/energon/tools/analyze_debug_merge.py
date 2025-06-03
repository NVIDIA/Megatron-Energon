# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import os
import re
from pathlib import Path
from typing import (
    Callable,
    Generator,
    List,
)

import click

# Regular expressions for parsing the log file efficiently
_re_ts = re.compile(rb'"ts":(\d+)')
_re_pid = re.compile(rb'"pid":(\d+)')


@click.command(name="analyze-debug-merge")
@click.argument(
    "log_paths",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
)
@click.argument(
    "output_path",
    type=click.Path(exists=False, writable=True, dir_okay=False, path_type=Path),
)
def command(
    log_paths: List[Path],
    output_path: Path,
):
    """Internal tool to merge multiple debug logs into a single file.

    The LOG_PATH should point to the folder with the debug log, or to a single log file."""

    if len(log_paths) == 0:
        raise click.ClickException("No log paths specified")
    log_files = []
    for log_path in log_paths:
        if log_path.is_dir():
            log_files.extend(sorted(log_path.glob("*.json")))
        elif log_path.is_file():
            log_files.append(log_path)
        else:
            raise click.ClickException(f"Invalid log path: {log_path}")

    if len(log_files) == 0:
        raise click.ClickException("No log files found")

    print(f"Merging {len(log_files)} log files into {output_path}")

    entry_count = 0
    with open(output_path, "wb") as f:
        f.write(b"[\n")
        for entry in merge_log_reader(log_files):
            f.write(entry + b",\n")
            entry_count += 1
        f.seek(-2, os.SEEK_END)
        f.write(b"]\n")
    print(f"Merged {len(log_files)} log files with {entry_count} entries into {output_path}")


def merge_log_reader(log_files: List[Path]) -> Generator[bytes, None, None]:
    """Merges multiple log files into a single stream of entries."""

    # Map of (file_idx, pid) to new pid
    repid_map = {}

    def get_repid(file_idx: int, pid: int) -> int:
        if (file_idx, pid) in repid_map:
            return repid_map[(file_idx, pid)]
        repid_map[(file_idx, pid)] = len(repid_map)
        return repid_map[(file_idx, pid)]

    log_readers = [
        _log_reader(log_file, functools.partial(get_repid, idx))
        for idx, log_file in enumerate(log_files)
    ]
    log_entries = []
    for idx in reversed(range(len(log_readers))):
        reader = log_readers[idx]
        try:
            while True:
                entry, ts = next(reader)
                if ts is not None:
                    log_entries.append((entry, ts))
                    break
                yield entry
        except StopIteration:
            log_readers.pop(idx)
    # Read the entries ordered by ts
    while len(log_entries) > 0:
        # Find the smallest entry, get that entry and fetch the next entry from the reader
        min_ts = log_entries[0][1]
        min_entry_idx = 0
        for entry_idx, (_, ts) in enumerate(log_entries[1:], 1):
            if ts < min_ts:
                min_ts = ts
                min_entry_idx = entry_idx
        min_entry, _ = log_entries[min_entry_idx]
        yield min_entry
        while True:
            try:
                next_entry, ts = next(log_readers[min_entry_idx])
                if ts is not None:
                    log_entries[min_entry_idx] = (next_entry, ts)
                    break
                yield next_entry
            except StopIteration:
                del log_readers[min_entry_idx]
                del log_entries[min_entry_idx]
                break


def _log_reader(
    log_file: Path, pidmap: Callable[[int], int]
) -> Generator[tuple[bytes, int | None], None, None]:
    """Reads a log file and yields a tuple of the line and the ts."""

    def pidmap_sub(match: re.Match[bytes]) -> bytes:
        return b'"pid":' + str(pidmap(int(match.group(1)))).encode()

    had_end = False
    with open(log_file, "rb") as f:
        assert f.read(2) == b"[\n", "Log file must start with a JSON array"
        for line in f:
            if not line:
                assert had_end, "Log file must end with a JSON array"
            if line.endswith(b"]\n"):
                had_end = True
            else:
                assert line.endswith(b",\n"), f"Log file must be newline-terminated: {line}"
            line = _re_pid.sub(pidmap_sub, line)
            ts = _re_ts.search(line)
            if ts is None:
                yield line[:-2], None
            else:
                yield line[:-2], int(ts.group(1))
        assert had_end, "Log file must end with a JSON array"


if __name__ == "__main__":
    command()
