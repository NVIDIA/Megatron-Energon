# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import io
import math
import shutil
import tarfile
from tqdm import tqdm
from pathlib import Path

import click
from PIL import Image


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


def path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def text_payload(sample_id: int, text_bytes: int) -> bytes:
    prefix = f"sample {sample_id:012d}\n".encode("utf-8")
    if len(prefix) >= text_bytes:
        return prefix[:text_bytes]
    return prefix + (b"x" * (text_bytes - len(prefix)))


def image_payload(sample_id: int, *, width: int, height: int, image_format: str) -> bytes:
    pillow_format = "JPEG" if image_format == "jpg" else image_format.upper()
    image = Image.new(
        "RGB",
        (width, height),
        color=((sample_id * 29) % 256, (sample_id * 53) % 256, (sample_id * 97) % 256),
    )
    output = io.BytesIO()
    image.save(output, format=pillow_format)
    return output.getvalue()


def add_tar_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    member = tarfile.TarInfo(name=name)
    member.size = len(data)
    member.mtime = 0
    member.mode = 0o644
    tar.addfile(member, io.BytesIO(data))


def create_dummy_webdataset(
    dataset_dir: Path,
    *,
    sample_count: int,
    samples_per_shard: int,
    text_bytes: int,
    images_per_sample: int = 0,
    image_width: int = 64,
    image_height: int = 64,
    image_format: str = "jpg",
) -> list[str]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    shard_count = math.ceil(sample_count / samples_per_shard)
    shard_names: list[str] = []

    for shard_idx in tqdm(range(shard_count)):
        shard_name = f"shard-{shard_idx:06d}.tar"
        shard_names.append(shard_name)
        shard_path = dataset_dir / shard_name
        shard_start = shard_idx * samples_per_shard
        shard_end = min(shard_start + samples_per_shard, sample_count)

        with tarfile.open(shard_path, mode="w") as tar:
            for sample_id in range(shard_start, shard_end):
                add_tar_member(
                    tar,
                    name=f"{sample_id:012d}.txt",
                    data=text_payload(sample_id, text_bytes),
                )
                for image_idx in range(images_per_sample):
                    part_name = (
                        image_format
                        if images_per_sample == 1
                        else f"image_{image_idx}.{image_format}"
                    )
                    add_tar_member(
                        tar,
                        name=f"{sample_id:012d}.{part_name}",
                        data=image_payload(
                            sample_id + image_idx,
                            width=image_width,
                            height=image_height,
                            image_format=image_format,
                        ),
                    )

    return shard_names


@click.command()
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--samples",
    default="100k",
    show_default=True,
    help="Number of text samples to generate. Supports k, m, and g suffixes.",
)
@click.option("--samples-per-shard", default=10_000, show_default=True, type=int)
@click.option("--text-bytes", default=128, show_default=True, type=int)
@click.option(
    "--images-per-sample",
    default=0,
    show_default=True,
    type=int,
    help="Number of image parts to add to each sample.",
)
@click.option("--image-width", default=64, show_default=True, type=int)
@click.option("--image-height", default=64, show_default=True, type=int)
@click.option(
    "--image-format",
    default="jpg",
    show_default=True,
    type=click.Choice(["jpg", "png"]),
    help="Image encoding and WebDataset part extension.",
)
@click.option(
    "--force-overwrite",
    is_flag=True,
    help="Delete OUTPUT_DIR first if it already exists.",
)
def main(
    *,
    output_dir: Path,
    samples: str,
    samples_per_shard: int,
    text_bytes: int,
    images_per_sample: int,
    image_width: int,
    image_height: int,
    image_format: str,
    force_overwrite: bool,
) -> None:
    """Create a dummy text and image WebDataset for prepare benchmarks."""

    sample_count = parse_count(samples)
    if sample_count <= 0:
        raise click.BadParameter("--samples must be > 0.")
    if samples_per_shard <= 0:
        raise click.BadParameter("--samples-per-shard must be > 0.")
    if text_bytes <= 0:
        raise click.BadParameter("--text-bytes must be > 0.")
    if images_per_sample < 0:
        raise click.BadParameter("--images-per-sample must be >= 0.")
    if image_width <= 0:
        raise click.BadParameter("--image-width must be > 0.")
    if image_height <= 0:
        raise click.BadParameter("--image-height must be > 0.")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not force_overwrite:
            raise click.ClickException(
                f"{output_dir} already exists and is not empty. Use --force-overwrite."
            )
        shutil.rmtree(output_dir)

    shard_names = create_dummy_webdataset(
        output_dir,
        sample_count=sample_count,
        samples_per_shard=samples_per_shard,
        text_bytes=text_bytes,
        images_per_sample=images_per_sample,
        image_width=image_width,
        image_height=image_height,
        image_format=image_format,
    )
    click.echo(
        f"Generated {sample_count} samples across {len(shard_names)} shards "
        f"({path_size(output_dir)} bytes) in {output_dir}."
    )


if __name__ == "__main__":
    main()
