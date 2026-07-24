# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause


from pathlib import Path
from typing import Optional

import click

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.metadata import (
    EnergonDatasetType,
    get_dataset_type,
)
from megatron.energon.flavors.webdataset.prepare import WebdatasetPreparator
from megatron.energon.media.extractor import MediaFilterConfig
from megatron.energon.media.filesystem_prepare import prepare_filesystem_dataset


@click.command(name="prepare-media")
@click.argument(
    "path",
    type=click.Path(path_type=EPath),
)
@click.option(
    "--progress/--no-progress",
    default=True,
)
@click.option(
    "--num-workers",
    type=int,
    default=16,
    help="Number of workers to use to scan files",
)
@click.option(
    "--media-metadata-by-glob",
    type=str,
    help="Media detection by using one or more glob patterns such as '*.jpg'. Separate multiple patterns by commas.",
)
@click.option(
    "--media-metadata-by-header",
    is_flag=True,
    help="Media detection by binary file header.",
)
@click.option(
    "--media-metadata-by-extension",
    is_flag=True,
    help="Media detection by standard file extensions.",
)
@click.option(
    "--tmp-path",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "When PATH is remote, directory where index.sqlite is built locally before upload. "
        "Default: a new directory under /tmp (removed after success)."
    ),
)
def command(
    path: EPath,
    progress: bool,
    num_workers: int,
    media_metadata_by_glob: str | None,
    media_metadata_by_header: bool,
    media_metadata_by_extension: bool,
    tmp_path: Optional[Path],
):
    """Prepare a filesystem dataset by collecting media metadata."""

    media_filter_config = MediaFilterConfig.parse(
        media_metadata_by_glob, media_metadata_by_header, media_metadata_by_extension
    )

    if tmp_path is not None:
        index_sqlite_tmp_path = tmp_path / "index.sqlite"
    else:
        index_sqlite_tmp_path = None

    ds_type = get_dataset_type(path)
    if ds_type == EnergonDatasetType.WEBDATASET:
        click.echo("Preparing webdataset and computing media metadata...")

        if progress:

            def progress_fn(els, length=None):
                with click.progressbar(
                    els,
                    label="Processing shards",
                    show_pos=True,
                    length=length,
                ) as bar:
                    yield from bar

        else:

            def progress_fn(els, length=None):
                return els

        count = WebdatasetPreparator.add_media_metadata(
            path,
            media_filter=media_filter_config,
            workers=num_workers,
            progress_fn=progress_fn,
            index_sqlite_tmp_path=index_sqlite_tmp_path,
        )

        click.echo(f"Done. Stored metadata for {count} files.")
        return
    elif ds_type not in (EnergonDatasetType.FILESYSTEM, EnergonDatasetType.INVALID):
        raise click.ClickException(
            f"'prepare-media' only supports WebDatasets or filesystem datasets, but this path is a '{ds_type}' dataset"
        )

    click.echo("Preparing filesystem dataset and computing media metadata...")
    stored = prepare_filesystem_dataset(
        path,
        media_filter_config,
        progress=progress,
        workers=num_workers,
        index_sqlite_tmp_path=index_sqlite_tmp_path,
    )
    click.echo(f"Done. Stored metadata for {stored} files.")


if __name__ == "__main__":
    command()
