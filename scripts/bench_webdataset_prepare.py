# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Iterator

import click

import megatron.energon.flavors.webdataset.indexing as indexing
from megatron.energon.flavors import BaseWebdatasetFactory
from megatron.energon.flavors.webdataset.config import INFO_JSON_FILENAME, MAIN_FOLDER_NAME


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    sample_count: int
    shard_count: int
    workers: int
    repeat: int
    prepare_seconds: float
    samples_per_second: float
    dataset_size_bytes: int
    meta_size_bytes: int
    sqlite_in_memory: bool
    sqlite_log_path: str
    dataset_path: str


def path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def progress_none(elements: Iterator[object], length: int | None = None) -> Iterator[object]:
    return elements


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def discover_shards(dataset_dir: Path) -> list[str]:
    shard_paths = sorted(
        path
        for pattern in ("**/*.tar", "**/*.tgz")
        for path in dataset_dir.glob(pattern)
        if MAIN_FOLDER_NAME not in path.relative_to(dataset_dir).parts
    )
    return [str(path.relative_to(dataset_dir)) for path in shard_paths]


def read_prepared_sample_count(dataset_dir: Path) -> int:
    info_path = dataset_dir / MAIN_FOLDER_NAME / INFO_JSON_FILENAME
    with info_path.open("r") as info_file:
        info = json.load(info_file)
    return sum(info["shard_counts"].values())


def remove_prepared_metadata(dataset_dir: Path) -> None:
    shutil.rmtree(dataset_dir / MAIN_FOLDER_NAME, ignore_errors=True)


def backup_prepared_metadata(dataset_dir: Path, backup_dir: Path) -> bool:
    metadata_dir = dataset_dir / MAIN_FOLDER_NAME
    if not metadata_dir.exists():
        return False

    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(metadata_dir), str(backup_dir))
    return True


def restore_prepared_metadata(dataset_dir: Path, backup_dir: Path) -> None:
    remove_prepared_metadata(dataset_dir)
    if backup_dir.exists():
        shutil.move(str(backup_dir), str(dataset_dir / MAIN_FOLDER_NAME))


def run_prepare_benchmark(
    dataset_dir: Path,
    shard_names: list[str],
    *,
    workers: int,
    write_log: Path | None,
    sqlite_in_memory: bool,
) -> tuple[float, int]:
    if write_log is not None:
        indexing.DEBUG_SQLITE_LOG = str(write_log)
        indexing.DEBUG_SQLITE_LOG_FILE = open(write_log, "w")
        indexing.DEBUG_SQLITE_LOG_FILE.write("table,len(rows),time(s),time(s)/len(rows)\n")

    try:
        remove_prepared_metadata(dataset_dir)
        start = time.perf_counter()
        BaseWebdatasetFactory.prepare_dataset(
            dataset_dir,
            shard_names,
            split_parts_ratio=[("train", 1.0)],
            shuffle_seed=None,
            progress_fn=progress_none,
            workers=workers,
            sqlite_in_memory=sqlite_in_memory,
        )
        prepare_seconds = time.perf_counter() - start
        return prepare_seconds, read_prepared_sample_count(dataset_dir)
    finally:
        if write_log is not None and indexing.DEBUG_SQLITE_LOG_FILE is not None:
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
@click.argument(
    "dataset_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option("--workers", default=16, show_default=True, type=int)
@click.option("--repeats", default=1, show_default=True, type=int)
@click.option(
    "--sqlite-in-memory/--sqlite-on-disk",
    default=False,
    show_default=True,
    help="Build the SQLite index in :memory: and save it to disk on close.",
)
@click.option(
    "--output-dir",
    default=Path("benchmark_outputs/webdataset_prepare"),
    show_default=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--keep-prepared-metadata/--delete-prepared-metadata",
    default=False,
    show_default=True,
    help="Keep generated .nv-meta after the benchmark if the dataset was not already prepared.",
)
def main(
    *,
    dataset_dir: Path,
    workers: int,
    repeats: int,
    sqlite_in_memory: bool,
    output_dir: Path,
    keep_prepared_metadata: bool,
) -> None:
    """Benchmark end-to-end WebDataset preparation on an existing dataset."""

    if workers <= 0:
        raise click.BadParameter("--workers must be > 0.")
    if repeats <= 0:
        raise click.BadParameter("--repeats must be > 0.")

    dataset_dir = dataset_dir.resolve()
    shard_names = discover_shards(dataset_dir)
    if not shard_names:
        raise click.ClickException(f"No .tar or .tgz shards found in {dataset_dir}.")

    run_dir = output_dir / timestamp()
    backup_dir = run_dir / "original_nv_meta"
    results_path = run_dir / "results.csv"
    sqlite_log_dir = run_dir / "sqlite_logs"
    run_dir.mkdir(parents=True, exist_ok=False)
    sqlite_log_dir.mkdir()

    click.echo(f"Writing benchmark output to {run_dir}")
    click.echo(f"Benchmarking {len(shard_names)} shards from {dataset_dir}")
    click.echo(",".join(field.name for field in fields(BenchmarkResult)))

    had_original_metadata = backup_prepared_metadata(dataset_dir, backup_dir)
    dataset_size_bytes = path_size(dataset_dir)

    try:
        for repeat in range(repeats):
            write_log = sqlite_log_dir / f"sqlite-repeat-{repeat:03d}-{timestamp()}.csv"
            click.echo(f"Writing SQLite log to {write_log}")
            prepare_seconds, sample_count = run_prepare_benchmark(
                dataset_dir,
                shard_names,
                workers=workers,
                write_log=write_log,
                sqlite_in_memory=sqlite_in_memory,
            )
            result = BenchmarkResult(
                sample_count=sample_count,
                shard_count=len(shard_names),
                workers=workers,
                repeat=repeat,
                prepare_seconds=prepare_seconds,
                samples_per_second=sample_count / prepare_seconds,
                dataset_size_bytes=dataset_size_bytes,
                meta_size_bytes=path_size(dataset_dir / MAIN_FOLDER_NAME),
                sqlite_in_memory=sqlite_in_memory,
                sqlite_log_path=str(write_log),
                dataset_path=str(dataset_dir),
            )
            write_result(results_path, result)
            click.echo(",".join(str(value) for value in asdict(result).values()))
    finally:
        if had_original_metadata:
            restore_prepared_metadata(dataset_dir, backup_dir)
        elif not keep_prepared_metadata:
            remove_prepared_metadata(dataset_dir)


if __name__ == "__main__":
    main()
