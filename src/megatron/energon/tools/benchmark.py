# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Optional

import click
import psutil
import tqdm
import yaml

from megatron.energon import WorkerConfig, get_loader, get_train_dataset
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset import (
    base_webdataset,
    fast_read,
    itar,
    itar_reader,
    random_access,
)
from megatron.energon.flavors.webdataset.empty_dataset_error import EmptyDatasetError
from megatron.energon.flavors.webdataset.itar import TarIndexReader


def check_index(dataset_path: EPath):
    if not dataset_path.is_dir():
        click.echo(f"{dataset_path} is not a directory, therefore the index will not be checked")
        return

    ok = True

    # Get info file
    info_file = dataset_path / ".nv-meta/.info.yaml"
    info = yaml.safe_load(info_file.read_text())

    click.echo("Checking the index files...")
    shards = info["shard_counts"]
    for shard_file, length in shards.items():
        with TarIndexReader(dataset_path / shard_file) as itar:
            l = len(itar)
            if l - 1 != length:
                ok = False
                print(
                    f"Error in shard {shard_file}: Shard length in Info file {length} != {l - 1} (length in index)"
                )
            shard_size = (dataset_path / shard_file).size()
            # Check that the last byte offset is the file size
            # start, end = itar.get_start_end(length - 1)
            # if end != (dataset_path / shard_file).size():
            #     ok = False
            #     print(f"Error in shard {shard_file}: Last byte offset {end} != file size {(dataset_path / shard_file).size()}")
            last_start = -1
            for i in range(length):
                start = itar[i]
                # if i == 0 or i == length - 1:
                #     print(f"shard_file: {shard_file} [{i}], start: {start}, end: {end}")
                if not (last_start <= start <= shard_size):
                    ok = False
                    print(
                        f"Error in shard {shard_file} [size={shard_size}]: Byte range {last_start}-{start} is not positive for sample {i}"
                    )
                last_start = start

    return ok


class BenchmarkWriter:
    def __init__(self, path: EPath):
        print(f"Writing benchmark to {path}")
        self.file = open(str(path), "wb")
        self.cur_sample = {}
        self.idx = 0
        self.proc = psutil.Process()

    def _write(self, message: dict):
        self.file.write(json.dumps(message, ensure_ascii=False).encode("utf-8") + b"\n")

    def sample_start(self):
        self.cur_sample["i"] = self.idx
        self.cur_sample["start_ns"] = time.perf_counter_ns()
        self.idx += 1

        self.io_counters = self.proc.io_counters()
        self.cpu_counters = self.proc.cpu_times()

    def sample_end(self):
        self.cur_sample["end_ns"] = time.perf_counter_ns()
        io_counters = self.proc.io_counters()
        cpu_counters = self.proc.cpu_times()
        self.proc.ionice()
        self.cur_sample["dur"] = (self.cur_sample["end_ns"] - self.cur_sample["start_ns"]) / 1e9
        self.cur_sample["n_opens"] = itar.STATS_NUMBER_OF_OPENS
        self.cur_sample["n_reads"] = itar.STATS_NUMBER_OF_READS
        self.cur_sample["n_seeks"] = itar.STATS_NUMBER_OF_SEEKS
        self.cur_sample["bytes_read"] = itar.STATS_BYTES_READ
        self.cur_sample["read_time_ns"] = itar.STATS_READ_TIME_NS
        self.cur_sample["open_time_ns"] = itar.STATS_OPEN_TIME_NS + fast_read.OPEN_TIME_NS
        self.cur_sample["decode_time_ns"] = (
            base_webdataset.STATS_DECODE_TIME_NS + random_access.STATS_DECODE_TIME_NS
        )
        self.cur_sample["ps_io_read_bytes"] = io_counters.read_bytes - self.io_counters.read_bytes
        self.cur_sample["ps_io_read_count"] = io_counters.read_count - self.io_counters.read_count
        self.cur_sample["ps_io_write_bytes"] = (
            io_counters.write_bytes - self.io_counters.write_bytes
        )
        self.cur_sample["ps_io_write_count"] = (
            io_counters.write_count - self.io_counters.write_count
        )
        self.cur_sample["ps_io_wait"] = cpu_counters.iowait - self.cpu_counters.iowait
        self.cur_sample["ps_cpu_user_time"] = cpu_counters.user - self.cpu_counters.user
        self.cur_sample["ps_cpu_system_time"] = cpu_counters.system - self.cpu_counters.system
        self._write(self.cur_sample)
        self.cur_sample.clear()
        if self.idx % 1000 == 0:
            self.file.flush()

    def close(self):
        self.file.close()


@click.command(name="benchmark")
@click.argument(
    "path",
    type=click.Path(path_type=EPath),
)
@click.option("--split-part", default="train", help="The split to run on", show_default=True)
@click.option(
    "--dataset-config", default="dataset.yaml", help="Dataset config file name", show_default=True
)
@click.option(
    "--split-config", default="split.yaml", help="Split config file name", show_default=True
)
@click.option(
    "--parallel", default=1, help="Number of parallel workers", show_default=True, type=int
)
@click.option(
    "--msps", default=1, help="Maximum number of samples per sequence", show_default=True, type=int
)
@click.option(
    "--limit", default=None, help="Number of samples to load", show_default=True, type=int
)
@click.option(
    "--output",
    default=None,
    help="Output file name",
    show_default=True,
    type=click.Path(path_type=EPath),
)
@click.option(
    "--stats-output",
    default=None,
    help="Output file name for stats",
    show_default=True,
    type=click.Path(path_type=Path),
)
def command(
    path: EPath,
    split_part: str,
    dataset_config: str,
    split_config: str,
    parallel: int,
    msps: int,
    limit: Optional[int],
    output: Optional[EPath],
    stats_output: Optional[Path],
):
    """Check energon dataset for errors.

    The PATH should point to the folder with the dataset.
    The dataset must comply with the energon dataset format. See README.md for more details."""

    # Check the tar file index
    # if not check_index(path):
    #     raise click.ClickException("Validation failed with errors, see logs for details.")

    # Check the dataset
    failed = False

    ignore_list = []

    def handler(exc: Exception, key: Optional[str] = None) -> None:
        nonlocal failed
        failed = True
        logging.exception(str(exc))
        if key is not None:
            ignore_list.append(key)

    kwargs = {}
    if dataset_config != "dataset.yaml":
        kwargs["dataset_config"] = dataset_config
    if split_config != "split.yaml":
        kwargs["split_config"] = split_config

    worker_config = WorkerConfig(rank=0, world_size=1, num_workers=parallel)

    proc = psutil.Process()

    try:
        dataset = get_train_dataset(
            EPath(path),
            split_part=split_part,
            worker_config=worker_config,
            batch_size=1,
            handler=handler,
            shuffle_buffer_size=0,
            max_samples_per_sequence=msps,
            **kwargs,
        )
    except EmptyDatasetError:
        click.echo(f"Skipping empty split part {split_part}")
        raise

    try:
        loader = get_loader(dataset, gc_collect_every_n_steps=10000000)
        if output is None:
            writer = None
        else:
            writer = BenchmarkWriter(output)
        start = time.perf_counter_ns()
        i = 0
        cpu_times = None
        io_counters = None
        if writer is not None:
            writer.sample_start()
        for _ in tqdm.tqdm(loader):
            if writer is not None:
                writer.sample_end()
            if i == 0:
                print(f"Time taken to load first sample: {time.perf_counter_ns() - start}")
                start = time.perf_counter_ns()
                cpu_times = proc.cpu_times()
                io_counters = proc.io_counters()
                fast_read.NUMBER_OF_OPENS = 0
                fast_read.NUMBER_OF_READS = 0
                fast_read.READ_BYTES = 0
                fast_read.READ_TIME_NS = 0
                fast_read.OPEN_TIME_NS = 0
                itar.STATS_NUMBER_OF_OPENS = 0
                itar.STATS_NUMBER_OF_READS = 0
                itar.STATS_NUMBER_OF_SEEKS = 0
                itar.STATS_READ_TIME_NS = 0
                itar.STATS_BYTES_READ = 0
                itar.STATS_OPEN_TIME_NS = 0
                itar_reader.STATS_TOTAL_READ_TIME_NS = 0
                itar_reader.STATS_TOTAL_FILES_COUNT = 0
                itar_reader.STATS_TOTAL_CONTENT_BYTES_READ = 0
                base_webdataset.STATS_DECODE_TIME_NS = 0
                random_access.STATS_READ_TIME_NS = 0
                random_access.STATS_DECODE_TIME_NS = 0
            if writer is not None:
                writer.sample_start()
            if limit is not None and i == limit + 1:
                break
            i += 1
        if writer is not None:
            writer.sample_end()
            writer.close()
        total_time_ns = time.perf_counter_ns() - start
        print(f"Time taken to load {i - 1} samples: {total_time_ns / 1e9}")

        # print("Other stats:")
        # print(f"Number of opens: {fast_read.NUMBER_OF_OPENS}")
        # print(f"Number of reads: {fast_read.NUMBER_OF_READS}")
        # print(f"Bytes read: {fast_read.READ_BYTES}")
        # print(f"Read time: {fast_read.READ_TIME_NS / 1e9}")
        # print(f"Number of opens: {itar.STATS_NUMBER_OF_OPENS}")
        # print(f"Number of reads: {itar.STATS_NUMBER_OF_READS}")
        # print(f"Number of seeks: {itar.STATS_NUMBER_OF_SEEKS}")
        # print(f"Bytes read: {itar.STATS_BYTES_READ}")
        # print(f"Read time: {itar.STATS_READ_TIME_NS / 1e9}")
        # print(f"Open time: {itar.STATS_OPEN_TIME_NS / 1e9}")
        # print(f"Decode time: {base_webdataset.STATS_DECODE_TIME_NS / 1e9}")
    except InterruptedError:
        raise
    except BaseException:
        traceback.print_exc()
        raise click.ClickException("Validation failed with errors, see logs for details.")
    if stats_output is not None:
        cpu_now = proc.cpu_times()
        io_counters_now = proc.io_counters()
        with open(stats_output, "ab") as f:
            if f.tell() == 0:
                entry = (
                    "path",
                    "dataset_config",
                    "split_config",
                    "split_part",
                    "msps",
                    "parallel",
                    "limit",
                    "clear_os_fs_cache",
                    "itar_cache_with_offset",
                    "fast_read_enabled",
                    "disable_file_handle_cache",
                    "n_opens",
                    "n_reads",
                    "n_seeks",
                    "io_wait (ms)",
                    "io_read_count",
                    "io_read_bytes (KiB)",
                    # "total_bytes_read (MiB)",
                    "avg_bytes_read (KiB)",
                    # "total_time (s)",
                    # "total_io_time (s)",
                    # "total_read_time (s)",
                    # "total_open_time (s)",
                    # "total_decode_time (s)",
                    "avg_time (ms)",
                    "avg_io_time (ms)",
                    "avg_io_read_time (ms)",
                    "avg_io_open_time (ms)",
                    "avg_decode_time (ms)",
                )
                f.write(f"{','.join(entry)}\n".encode("utf-8"))
            io_wait = cpu_now.iowait - cpu_times.iowait
            io_read_bytes = io_counters_now.read_bytes - io_counters.read_bytes
            io_read_count = io_counters_now.read_count - io_counters.read_count
            entry = (
                path.name,
                dataset_config,
                split_config,
                split_part,
                f"{msps}",
                f"{parallel}",
                f"{limit}",
                f"{itar_reader.CLEAR_OS_FS_CACHE}",
                f"{itar_reader.ITAR_CACHE_WITH_OFFSET}",
                f"{itar_reader.FAST_READ_ENABLED}",
                f"{itar_reader.DISABLE_FILE_HANDLE_CACHE}",
                f"{fast_read.NUMBER_OF_OPENS or itar.STATS_NUMBER_OF_OPENS}",
                f"{fast_read.NUMBER_OF_READS or itar.STATS_NUMBER_OF_READS}",
                f"{itar.STATS_NUMBER_OF_SEEKS}",
                f"{io_wait / limit * 1e3}",
                f"{io_read_count / limit:.3f}",
                f"{io_read_bytes / 1024 / limit:.3f}",
                # f"{itar_reader.STATS_TOTAL_CONTENT_BYTES_READ / 1024 / 1024:.3f}",
                f"{itar_reader.STATS_TOTAL_CONTENT_BYTES_READ / 1024 / limit:.3f}",
                # f"{total_time_ns / 1e9:.3f}",
                # f"{(itar_reader.STATS_TOTAL_READ_TIME_NS) / 1e9:.3f}",
                # f"{(fast_read.READ_TIME_NS or itar.STATS_READ_TIME_NS) / 1e9:.3f}",
                # f'{(fast_read.OPEN_TIME_NS or itar.STATS_OPEN_TIME_NS) / 1e9:.3f}',
                # f"{decode_time_ns / 1e9:.3f}",
                f"{total_time_ns / limit / 1e6:.3f}",
                f"{(itar_reader.STATS_TOTAL_READ_TIME_NS + random_access.STATS_READ_TIME_NS) / limit / 1e6:.3f}",
                f"{(fast_read.READ_TIME_NS or itar.STATS_READ_TIME_NS) / limit / 1e6:.3f}",
                f"{(fast_read.OPEN_TIME_NS or itar.STATS_OPEN_TIME_NS) / limit / 1e6:.3f}",
                f"{(base_webdataset.STATS_DECODE_TIME_NS - random_access.STATS_READ_TIME_NS) / limit / 1e6:.3f}",
            )
            f.write(f"{','.join(entry)}\n".encode("utf-8"))

    if failed:
        click.echo(
            "The following shards/samples failed (maybe set as dataset.yaml:ignore_list):", err=True
        )
        for item in ignore_list:
            click.echo(f"- {item}", err=True)
        raise click.ClickException("Validation failed with errors, see logs for details.")


if __name__ == "__main__":
    command()
