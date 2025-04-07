# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from concurrent.futures import Future
import json
import logging
import time
import traceback
from typing import Optional

import click
import tqdm
import yaml

from megatron.energon import WorkerConfig, get_loader, get_train_dataset
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.empty_dataset_error import EmptyDatasetError
from megatron.energon.flavors.webdataset.itar import TarIndexReader2
from megatron.energon.flavors.webdataset import fast_read
from megatron.energon.flavors.webdataset import sample_loader
from megatron.energon.flavors.webdataset import base_webdataset
from megatron.energon.flavors.webdataset import random_access
from megatron.energon.wrappers import map_dataset


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
        with TarIndexReader2(dataset_path / shard_file) as itar:
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

    def _write(self, message: dict):
        self.file.write(json.dumps(message, ensure_ascii=False).encode("utf-8") + b"\n")

    def sample_start(self):
        self.cur_sample["i"] = self.idx
        self.cur_sample["start_ns"] = time.perf_counter_ns()
        self.idx += 1

    def sample_end(self):
        self.cur_sample["end_ns"] = time.perf_counter_ns()
        self.cur_sample["dur"] = (self.cur_sample["end_ns"] - self.cur_sample["start_ns"]) / 1e9
        self.cur_sample["n_opens"] = fast_read.NUMBER_OF_OPENS
        self.cur_sample["n_reads"] = fast_read.NUMBER_OF_READS
        self.cur_sample["bytes_read"] = fast_read.READ_BYTES
        self.cur_sample["read_time_ns"] = fast_read.READ_TIME_NS
        self.cur_sample["queue_size"] = sample_loader.STATS_CURRENT_QUEUE_SIZE
        self.cur_sample["queue_size2"] = map_dataset.STATS_PREFETCH_PIPELINE_SIZE
        self.cur_sample["queue_size3"] = random_access.STATS_LOAD_QUEUE_SIZE
        self.cur_sample["queue_size4"] = random_access.STATS_DECODE_QUEUE_SIZE
        # self.cur_sample["decode_time_ns"] = base_webdataset.STATS_DECODE_TIME_NS
        self.cur_sample["decode_time_ns"] = base_webdataset.STATS_DECODE_TIME_NS + random_access.STATS_DECODE_TIME_NS
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
@click.option(
    "--split-parts", default="train,val,test", help="The splits to verify", show_default=True
)
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
def command(
    path: EPath,
    split_parts: str,
    dataset_config: str,
    split_config: str,
    parallel: int,
    msps: int,
    limit: Optional[int],
    output: Optional[EPath],
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

    for split_part in split_parts.split(","):
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
            continue

        try:
            if output is None:
                writer = None
            else:
                writer = BenchmarkWriter(output)
            start = time.time()
            i = 0
            last_sample = None
            if writer is not None:
                writer.sample_start()
            for sample_batch in tqdm.tqdm(get_loader(dataset)):
                if last_sample is not None:
                    for packed_sample_batch in last_sample.samples:
                        for packed_sample in packed_sample_batch:
                            for img in packed_sample.images:
                                if isinstance(img, Future):
                                    img.result()
                    last_sample = sample_batch
                if writer is not None:
                    writer.sample_end()
                if i == 0:
                    print(f"Time taken to load first sample: {time.time() - start}")
                    start = time.time()
                if writer is not None:
                    writer.sample_start()
                if limit is not None and i == limit + 1:
                    break
                i += 1
            if writer is not None:
                writer.sample_end()
                writer.close()
            print(f"Time taken to load {i - 1} samples: {time.time() - start}")

            print("Other stats:")
            print(f"Number of opens: {fast_read.NUMBER_OF_OPENS}")
            print(f"Number of reads: {fast_read.NUMBER_OF_READS}")
            print(f"Bytes read: {fast_read.READ_BYTES}")
            print(f"Read time: {fast_read.READ_TIME_NS / 1e9}")
        except InterruptedError:
            raise
        except BaseException:
            traceback.print_exc()
            raise click.ClickException("Validation failed with errors, see logs for details.")

    if failed:
        click.echo(
            "The following shards/samples failed (maybe set as dataset.yaml:ignore_list):", err=True
        )
        for item in ignore_list:
            click.echo(f"- {item}", err=True)
        raise click.ClickException("Validation failed with errors, see logs for details.")


if __name__ == "__main__":
    command()
