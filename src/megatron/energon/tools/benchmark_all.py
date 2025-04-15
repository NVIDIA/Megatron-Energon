# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import click


@click.command(name="benchmark-all")
@click.argument(
    "paths",
    type=click.Path(path_type=Path),
    nargs=-1,
)
@click.option(
    "--split-parts",
    default="train",
    help="The splits to verify (comma-separated list)",
    show_default=True,
    type=str,
    callback=lambda ctx, param, value: value.split(","),
)
@click.option(
    "--dataset-configs",
    default="dataset.yaml",
    help="Dataset config file name (comma-separated list)",
    show_default=True,
    type=str,
    callback=lambda ctx, param, value: value.split(","),
)
@click.option(
    "--split-configs",
    default="split.yaml",
    help="Split config file name (comma-separated list)",
    show_default=True,
    type=str,
    callback=lambda ctx, param, value: value.split(","),
)
@click.option(
    "--parallel", default=1, help="Number of parallel workers", show_default=True, type=int
)
@click.option(
    "--msps",
    default="1,10,100",
    help="Maximum number of samples per sequence (comma-separated list)",
    show_default=True,
    type=str,
    callback=lambda ctx, param, value: [int(x) for x in value.split(",")],
)
@click.option(
    "--limit", default=None, help="Number of samples to load", show_default=True, type=int
)
@click.option(
    "--output",
    default=None,
    help="Output file name",
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option(
    "--stats-output",
    default=None,
    help="Output file name for stats",
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option(
    "--iter-cache-with-offset",
    default="1,0",
    help="Iter cache with offset",
    show_default=True,
    type=str,
    callback=lambda ctx, param, value: value.split(","),
)
@click.option(
    "--fast-read-enabled",
    default="1,0",
    help="Fast read enabled",
    show_default=True,
    type=str,
    callback=lambda ctx, param, value: value.split(","),
)
@click.option(
    "--disable-file-handle-cache",
    default="1,0",
    help="Disable file handle cache",
    show_default=True,
    callback=lambda ctx, param, value: value.split(","),
)
def command(
    paths: List[Path],
    split_parts: List[str],
    dataset_configs: List[str],
    split_configs: List[str],
    parallel: int,
    msps: List[int],
    limit: Optional[int],
    output: Optional[Path],
    stats_output: Optional[Path],
    iter_cache_with_offset: List[str],
    fast_read_enabled: List[str],
    disable_file_handle_cache: List[str],
):
    """Check energon dataset for errors.

    The PATH should point to the folder with the dataset.
    The dataset must comply with the energon dataset format. See README.md for more details."""
    current_output = None

    for path in paths:
        for dataset_config in dataset_configs:
            for split_config in split_configs:
                for split_part in split_parts:
                    for msp in msps:
                        for run_iter_cache_with_offset in iter_cache_with_offset:
                            for run_fast_read_enabled in fast_read_enabled:
                                for run_disable_file_handle_cache in disable_file_handle_cache:
                                    print(
                                        f"Benchmarking {path} with dataset_config={dataset_config} and split_config={split_config} and split_part={split_part} and msp={msp} "
                                        f"iter_cache_with_offset={run_iter_cache_with_offset} and fast_read_enabled={run_fast_read_enabled} and disable_file_handle_cache={run_disable_file_handle_cache}"
                                    )
                                    # Create a unique output file for this benchmark if output is specified
                                    if output is not None:
                                        current_output = output.with_name(
                                            f"{output.stem}_{path.name}_{dataset_config.replace('.yaml', '')}_{split_config.replace('.yaml', '')}_{split_part}_{msp}.{output.suffix}"
                                        )

                                    # Build the command
                                    cmd = [
                                        sys.executable,
                                        "-m",
                                        "megatron.energon.cli.main",
                                        "benchmark",
                                        str(path),
                                        "--split-part",
                                        split_part,
                                        "--dataset-config",
                                        dataset_config,
                                        "--split-config",
                                        split_config,
                                        "--parallel",
                                        str(parallel),
                                        "--msps",
                                        str(msp),
                                    ]

                                    if limit is not None:
                                        cmd.extend(["--limit", str(limit)])

                                    if current_output is not None:
                                        cmd.extend(["--output", str(current_output)])

                                    if stats_output is not None:
                                        cmd.extend(["--stats-output", str(stats_output)])

                                    # Execute the benchmark command
                                    subprocess.run(
                                        cmd,
                                        check=True,
                                        env={
                                            **os.environ,
                                            "CLEAR_OS_FS_CACHE": "1",
                                            "ITAR_CACHE_WITH_OFFSET": run_iter_cache_with_offset,
                                            "FAST_READ_ENABLED": run_fast_read_enabled,
                                            "DISABLE_FILE_HANDLE_CACHE": run_disable_file_handle_cache,
                                        },
                                    )


if __name__ == "__main__":
    command()
