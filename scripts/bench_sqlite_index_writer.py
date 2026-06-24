# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import click

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.config import INDEX_BATCH_SIZE
import megatron.energon.flavors.webdataset.indexing as indexing
from megatron.energon.flavors.webdataset.indexing import SqliteIndexReader, SqliteIndexWriter
from megatron.energon.flavors.webdataset.itar import ITarRawSamplePartPointer, ITarSamplePointer
from megatron.energon.flavors.webdataset.lmdb_index import LmdbIndexReader, LmdbIndexWriter
from megatron.energon.flavors.webdataset.prepare import IndexSample, IndexSamplePart


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    idx_type: str
    sample_count: int
    parts_per_sample: int
    batch_size: int
    sqlite_in_memory: bool
    read_lookups: int
    insert_seconds: float
    close_seconds: float
    write_total_seconds: float
    read_sample_pointer_seconds: float
    read_sample_part_seconds: float
    read_list_sample_parts_seconds: float
    read_total_seconds: float
    total_seconds: float
    db_size_bytes: int
    db_path: str
    sqlite_log_path: str


class NoopReader:
    """No-op index reader for benchmarking read-only performance."""

    def get_sample_pointer_by_key(self, key: str) -> ITarSamplePointer:
        return ITarSamplePointer(tar_file_id=0, byte_offset=0, byte_size=0)

    def get_sample_part(self, key: str, part_name: str) -> ITarRawSamplePartPointer:
        return ITarRawSamplePartPointer(tar_file_id=0, raw_byte_offset=0, raw_byte_size=0)

    def list_sample_parts(self, key: str) -> list[ITarRawSamplePartPointer]:
        return []

    def close(self) -> None:
        pass

    def __enter__(self) -> NoopReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


class NoopWriter:
    """No-op index writer for benchmarking write-only performance."""

    def append_samples(self, samples: list[IndexSample]) -> None:
        pass

    def append_parts(self, parts: list[IndexSamplePart]) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> NoopWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


def parse_count(raw_count: str) -> int:
    raw_count = raw_count.strip().replace("_", "")
    if not raw_count:
        raise click.BadParameter("Counts must not be empty.")

    suffix = raw_count[-1].lower()
    if suffix in {"k", "m", "g"}:
        multiplier = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}[suffix]
        value = raw_count[:-1]
    else:
        multiplier = 1
        value = raw_count

    try:
        return int(value) * multiplier
    except ValueError as exc:
        raise click.BadParameter(f"Invalid sample count: {raw_count!r}") from exc


def parse_counts(raw_counts: str) -> tuple[int, ...]:
    counts = tuple(parse_count(raw_count) for raw_count in raw_counts.split(","))
    if not counts:
        raise click.BadParameter("At least one count is required.")
    return counts


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def part_name(part_idx: int) -> str:
    common_names = ("jpg", "txt", "json", "cls", "png", "webp")
    if part_idx < len(common_names):
        return common_names[part_idx]
    return f"part{part_idx:03d}"


def sample_key(sample_id: int) -> str:
    return f"sample-{sample_id:012d}"


def make_batch(
    *,
    batch_start: int,
    batch_end: int,
    samples_per_shard: int,
    parts_per_sample: int,
    sample_byte_size: int,
) -> tuple[list[IndexSample], list[IndexSamplePart]]:
    samples: list[IndexSample] = []
    sample_parts: list[IndexSamplePart] = []

    for sample_id in range(batch_start, batch_end):
        tar_file_id = sample_id // samples_per_shard
        sample_index = sample_id % samples_per_shard
        byte_offset = sample_index * sample_byte_size
        key = sample_key(sample_id)

        samples.append(
            IndexSample(
                tar_file_id=tar_file_id,
                sample_key=key,
                sample_index=sample_index,
                byte_offset=byte_offset,
                byte_size=sample_byte_size,
            )
        )

        for part_idx in range(parts_per_sample):
            sample_parts.append(
                IndexSamplePart(
                    tar_file_id=tar_file_id,
                    sample_index=sample_index,
                    part_name=part_name(part_idx),
                    content_byte_offset=byte_offset + 512 + part_idx * 128,
                    content_byte_size=128 + part_idx,
                )
            )

    return samples, sample_parts


def path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def index_data_files(db_path: Path, idx_type: str) -> list[Path]:
    if idx_type == "sqlite":
        return [db_path] if db_path.is_file() else []
    if db_path.is_dir():
        return sorted(p for p in db_path.iterdir() if p.is_file())
    return []


def drop_filesystem_cache(db_path: Path, idx_type: str) -> bool:
    """Evict OS page cache for index files via posix_fadvise (Linux).

    Returns True if cache was dropped, False if unsupported or no files.
    """
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False

    files = index_data_files(db_path, idx_type)
    if not files:
        return False

    os.sync()
    for file_path in files:
        fd = os.open(file_path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return True


def open_index_reader(db_path: Path, idx_type: str) -> SqliteIndexReader | LmdbIndexReader:
    if idx_type == "sqlite":
        return SqliteIndexReader(EPath(db_path))
    if idx_type == "lmdb":
        return LmdbIndexReader(EPath(db_path))
    if idx_type == "noop":
        return NoopReader()
    raise click.BadParameter(f"Unknown index type: {idx_type!r}")


def open_index_writer(
    db_path: Path,
    idx_type: str,
    *,
    sqlite_in_memory: bool,
) -> SqliteIndexWriter | LmdbIndexWriter:
    if idx_type == "sqlite":
        return SqliteIndexWriter(EPath(db_path), in_memory=sqlite_in_memory)
    if idx_type == "lmdb":
        if sqlite_in_memory:
            raise click.BadParameter("--sqlite-in-memory can only be used with --idx-type sqlite.")
        return LmdbIndexWriter(EPath(db_path))
    if idx_type == "noop":
        if sqlite_in_memory:
            raise click.BadParameter("--sqlite-in-memory can only be used with --idx-type sqlite.")
        return NoopWriter()
    raise click.BadParameter(f"Unknown index type: {idx_type!r}")


def lookup_sample_ids(*, sample_count: int, read_lookups: int, rng: random.Random) -> list[int]:
    if read_lookups >= sample_count:
        return list(range(sample_count))
    return [rng.randrange(sample_count) for _ in range(read_lookups)]


def run_write_benchmark(
    *,
    writer: SqliteIndexWriter | LmdbIndexWriter,
    sample_count: int,
    parts_per_sample: int,
    batch_size: int,
    samples_per_shard: int,
    sample_byte_size: int,
) -> tuple[float, float]:
    insert_start = time.perf_counter()
    for batch_start in range(0, sample_count, batch_size):
        batch_end = min(batch_start + batch_size, sample_count)
        samples, sample_parts = make_batch(
            batch_start=batch_start,
            batch_end=batch_end,
            samples_per_shard=samples_per_shard,
            parts_per_sample=parts_per_sample,
            sample_byte_size=sample_byte_size,
        )
        writer.append_samples(samples)
        writer.append_parts(sample_parts)
    insert_seconds = time.perf_counter() - insert_start

    close_start = time.perf_counter()
    writer.close()
    close_seconds = time.perf_counter() - close_start
    return insert_seconds, close_seconds


def run_read_benchmark(
    *,
    reader: SqliteIndexReader | LmdbIndexReader,
    sample_count: int,
    parts_per_sample: int,
    read_lookups: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    sample_ids = lookup_sample_ids(
        sample_count=sample_count,
        read_lookups=read_lookups,
        rng=rng,
    )
    first_part = part_name(0)

    pointer_start = time.perf_counter()
    for sample_id in sample_ids:
        reader.get_sample_pointer_by_key(sample_key(sample_id))
    read_sample_pointer_seconds = time.perf_counter() - pointer_start

    part_start = time.perf_counter()
    for sample_id in sample_ids:
        reader.get_sample_part(sample_key(sample_id), first_part)
    read_sample_part_seconds = time.perf_counter() - part_start

    list_parts_start = time.perf_counter()
    for sample_id in sample_ids:
        list(reader.list_sample_parts(sample_key(sample_id)))
    read_list_sample_parts_seconds = time.perf_counter() - list_parts_start

    return (
        read_sample_pointer_seconds,
        read_sample_part_seconds,
        read_list_sample_parts_seconds,
    )


def run_benchmark(
    *,
    db_path: Path,
    idx_type: str,
    sample_count: int,
    parts_per_sample: int,
    batch_size: int,
    samples_per_shard: int,
    sample_byte_size: int,
    read_lookups: int,
    skip_read: bool,
    drop_read_cache: bool,
    sqlite_in_memory: bool,
    sqlite_log_path: Path | None,
    rng: random.Random,
) -> BenchmarkResult:
    if sqlite_log_path is not None:
        indexing.DEBUG_SQLITE_LOG = str(sqlite_log_path)
        indexing.DEBUG_SQLITE_LOG_FILE = open(sqlite_log_path, "w")
        indexing.DEBUG_SQLITE_LOG_FILE.write("table,len(rows),time(s),time(s)/len(rows)\n")

    try:
        writer = open_index_writer(db_path, idx_type, sqlite_in_memory=sqlite_in_memory)
        insert_seconds, close_seconds = run_write_benchmark(
            writer=writer,
            sample_count=sample_count,
            parts_per_sample=parts_per_sample,
            batch_size=batch_size,
            samples_per_shard=samples_per_shard,
            sample_byte_size=sample_byte_size,
        )
        write_total_seconds = insert_seconds + close_seconds

        read_sample_pointer_seconds = 0.0
        read_sample_part_seconds = 0.0
        read_list_sample_parts_seconds = 0.0
        if not skip_read:
            effective_lookups = min(read_lookups, sample_count)
            if drop_read_cache:
                drop_filesystem_cache(db_path, idx_type)
            reader = open_index_reader(db_path, idx_type)
            try:
                (
                    read_sample_pointer_seconds,
                    read_sample_part_seconds,
                    read_list_sample_parts_seconds,
                ) = run_read_benchmark(
                    reader=reader,
                    sample_count=sample_count,
                    parts_per_sample=parts_per_sample,
                    read_lookups=effective_lookups,
                    rng=rng,
                )
            finally:
                reader.close()

        read_total_seconds = (
            read_sample_pointer_seconds
            + read_sample_part_seconds
            + read_list_sample_parts_seconds
        )

        return BenchmarkResult(
            idx_type=idx_type,
            sample_count=sample_count,
            parts_per_sample=parts_per_sample,
            batch_size=batch_size,
            sqlite_in_memory=sqlite_in_memory,
            read_lookups=min(read_lookups, sample_count) if not skip_read else 0,
            insert_seconds=insert_seconds,
            close_seconds=close_seconds,
            write_total_seconds=write_total_seconds,
            read_sample_pointer_seconds=read_sample_pointer_seconds,
            read_sample_part_seconds=read_sample_part_seconds,
            read_list_sample_parts_seconds=read_list_sample_parts_seconds,
            read_total_seconds=read_total_seconds,
            total_seconds=write_total_seconds + read_total_seconds,
            db_size_bytes=path_size(db_path),
            db_path=str(db_path),
            sqlite_log_path=str(sqlite_log_path) if sqlite_log_path is not None else "",
        )
    finally:
        if sqlite_log_path is not None and indexing.DEBUG_SQLITE_LOG_FILE is not None:
            indexing.DEBUG_SQLITE_LOG_FILE.close()
            indexing.DEBUG_SQLITE_LOG_FILE = None


def write_result(csv_path: Path, result: BenchmarkResult) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=tuple(asdict(result).keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(result))


@click.command()
@click.option(
    "--counts",
    default="1M,10M,100M",
    show_default=True,
    help="Comma-separated sample counts. Supports k, m, and g suffixes.",
)
@click.option("--parts-per-sample", default=2, show_default=True, type=int)
@click.option("--batch-size", default=INDEX_BATCH_SIZE, show_default=True, type=int)
@click.option("--samples-per-shard", default=10_000, show_default=True, type=int)
@click.option("--sample-byte-size", default=4096, show_default=True, type=int)
@click.option(
    "--idx-type",
    default="sqlite",
    show_default=True,
    type=click.Choice(["sqlite", "lmdb", "noop"]),
)
@click.option(
    "--sqlite-in-memory/--sqlite-on-disk",
    default=False,
    show_default=True,
    help="Build SQLite indexes in :memory: and save them to disk on close.",
)
@click.option(
    "--read-lookups",
    default=10_000,
    show_default=True,
    type=int,
    help="Number of random sample lookups for read benchmarks (capped at sample count).",
)
@click.option(
    "--skip-read/--no-skip-read",
    default=False,
    show_default=True,
    help="Only benchmark index write (insert + close/finalize).",
)
@click.option("--seed", default=0, show_default=True, type=int, help="RNG seed for read lookups.")
@click.option(
    "--drop-read-cache/--keep-read-cache",
    default=True,
    show_default=True,
    help=(
        "Before read benchmarks, evict OS page cache for index files (Linux posix_fadvise). "
        "Makes sqlite/lmdb read timings comparable after a warm write/close path."
    ),
)
@click.option(
    "--output-dir",
    default=Path("benchmark_outputs/sqlite_index_writer"),
    show_default=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--keep-db/--delete-db",
    default=False,
    show_default=True,
    help="Keep generated index files after each run.",
)
def main(
    *,
    counts: str,
    idx_type: str,
    parts_per_sample: int,
    batch_size: int,
    samples_per_shard: int,
    sample_byte_size: int,
    sqlite_in_memory: bool,
    read_lookups: int,
    skip_read: bool,
    drop_read_cache: bool,
    seed: int,
    output_dir: Path,
    keep_db: bool,
) -> None:
    """Benchmark webdataset index write and read performance."""

    parsed_counts = parse_counts(counts)
    if parts_per_sample < 0:
        raise click.BadParameter("--parts-per-sample must be >= 0.")
    if batch_size <= 0:
        raise click.BadParameter("--batch-size must be > 0.")
    if samples_per_shard <= 0:
        raise click.BadParameter("--samples-per-shard must be > 0.")
    if sample_byte_size <= 0:
        raise click.BadParameter("--sample-byte-size must be > 0.")
    if sqlite_in_memory and idx_type != "sqlite":
        raise click.BadParameter("--sqlite-in-memory can only be used with --idx-type sqlite.")
    if read_lookups <= 0 and not skip_read:
        raise click.BadParameter("--read-lookups must be > 0 unless --skip-read is set.")

    rng = random.Random(seed)
    run_dir = output_dir / timestamp()
    sqlite_log_dir = run_dir / "sqlite_logs"
    run_dir.mkdir(parents=True, exist_ok=False)
    sqlite_log_dir.mkdir()
    results_path = run_dir / "results.csv"

    click.echo(f"Writing benchmark output to {run_dir}")
    if not skip_read:
        if drop_read_cache:
            if hasattr(os, "posix_fadvise"):
                click.echo("Dropping OS page cache for index files before each read benchmark.")
            else:
                click.echo(
                    "Warning: posix_fadvise unavailable on this platform; "
                    "read cache is not dropped.",
                    err=True,
                )
        click.echo(
            "Read benchmarks: get_sample_pointer_by_key, get_sample_part, list_sample_parts "
            f"({read_lookups} lookups each, seed={seed})"
        )
    click.echo(",".join(field.name for field in fields(BenchmarkResult)))

    for sample_count in parsed_counts:
        if idx_type == "sqlite":
            db_path = run_dir / f"index-{idx_type}-{sample_count}-samples.sqlite"
        else:
            db_path = run_dir / f"index-{idx_type}-{sample_count}-samples"
            db_path.mkdir(parents=True, exist_ok=True)
        sqlite_log_path = (
            sqlite_log_dir / f"sqlite-{sample_count}-samples-{timestamp()}.csv"
            if idx_type == "sqlite"
            else None
        )
        if sqlite_log_path is not None:
            click.echo(f"Writing SQLite log to {sqlite_log_path}")

        result = run_benchmark(
            db_path=db_path,
            idx_type=idx_type,
            sample_count=sample_count,
            parts_per_sample=parts_per_sample,
            batch_size=batch_size,
            samples_per_shard=samples_per_shard,
            sample_byte_size=sample_byte_size,
            read_lookups=read_lookups,
            skip_read=skip_read,
            drop_read_cache=drop_read_cache,
            sqlite_in_memory=sqlite_in_memory,
            sqlite_log_path=sqlite_log_path,
            rng=rng,
        )
        write_result(results_path, result)
        click.echo(",".join(str(value) for value in asdict(result).values()))

        if not keep_db:
            if db_path.is_dir():
                shutil.rmtree(db_path, ignore_errors=True)
            elif db_path.exists():
                db_path.unlink()

    if not keep_db and not any(run_dir.iterdir()):
        shutil.rmtree(run_dir)


if __name__ == "__main__":
    main()
