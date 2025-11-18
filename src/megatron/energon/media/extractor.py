# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Union

import filetype
from PIL import Image, UnidentifiedImageError

from megatron.energon.av import AVDecoder
from megatron.energon.media.metadata import ImageMetadata, MediaMetadataBase, MediaMetadataType

logger = logging.getLogger(__name__)


SourceData = Union[bytes, Path, io.IOBase]


class MediaFilterStrategy(str, Enum):
    """Strategy used to decide whether an entry should be treated as media."""

    EXT = "EXT"  # by file extension
    TYPE = "TYPE"  # by filetype (detected using filetype package)
    PATTERN = "PATTERN"  # by glob pattern (e.g. '*.jpg')


@dataclass(frozen=True)
class MediaFilterConfig:
    """Configuration for media detection during dataset preparation."""

    strategy: MediaFilterStrategy
    pattern: str | None = None

    @classmethod
    def parse(cls, value: str | None) -> "MediaFilterConfig":
        if value is None:
            value = "EXT"

        value_upper = value.upper()

        if value_upper == MediaFilterStrategy.EXT.value:
            return cls(strategy=MediaFilterStrategy.EXT)

        if value_upper == MediaFilterStrategy.TYPE.value:
            return cls(strategy=MediaFilterStrategy.TYPE)

        return cls(strategy=MediaFilterStrategy.PATTERN, pattern=value)


_IMAGE_EXTENSIONS: set[str] = {
    "bmp",
    "gif",
    "heic",
    "jpeg",
    "jpg",
    "png",
    "tif",
    "tiff",
    "webp",
}

_AV_EXTENSIONS: set[str] = {
    "aac",
    "flac",
    "m4a",
    "m4v",
    "mkv",
    "mov",
    "mp3",
    "mp4",
    "ogg",
    "wav",
    "webm",
}

_FILETYPE_PROBE_SIZE = 262


def should_consider_media(name: str, config: MediaFilterConfig) -> bool:
    """Check whether a file name qualifies for metadata extraction under the filter."""

    lower_name = name.lower()

    if config.strategy == MediaFilterStrategy.TYPE:
        # TYPE detection relies on file content, hence it always requires inspection.
        return True

    if config.strategy == MediaFilterStrategy.EXT:
        return _guess_type_from_extension(lower_name) is not None

    assert config.pattern is not None, "Pattern strategy requires a glob expression"
    return fnmatch(lower_name, config.pattern.lower())


def extract_metadata(
    source: SourceData,
    config: MediaFilterConfig,
    filename: str | None = None,
) -> MediaMetadataBase | None:
    if isinstance(source, (bytes, bytearray, io.IOBase)):
        assert filename is not None, (
            "Filename is required when extracting metadata from bytes or IOBase"
        )
    else:
        assert filename is None, "Filename is not allowed when extracting metadata from path"
        filename = source.name

    media_type = _detect_media_type(filename, config, source)

    print(f"Detecting media type for {filename} with config {config}: {media_type}")

    if media_type is None:
        return None

    metadata = _build_metadata(media_type, source)
    if metadata is None:
        return None
    return metadata


def _detect_media_type(
    name: str,
    config: MediaFilterConfig,
    source: SourceData,
) -> MediaMetadataType | None:
    lower_name = name.lower()
    extension_guess = _guess_type_from_extension(lower_name)

    if config.strategy == MediaFilterStrategy.EXT:
        return extension_guess

    if config.strategy == MediaFilterStrategy.TYPE:
        detected = _guess_type_from_filetype(source)
        return detected if detected is not None else extension_guess

    assert config.pattern is not None
    if not fnmatch(lower_name, config.pattern.lower()):
        return None

    if extension_guess is not None:
        return extension_guess

    return _guess_type_from_filetype(source)


def _guess_type_from_extension(name: str) -> MediaMetadataType | None:
    suffix = Path(name).suffix.lstrip(".").lower()
    if suffix in _IMAGE_EXTENSIONS:
        return MediaMetadataType.IMAGE
    if suffix in _AV_EXTENSIONS:
        return MediaMetadataType.AV
    return None


def _guess_type_from_filetype(source: SourceData) -> MediaMetadataType | None:
    kind = filetype.guess(source)

    if kind is None or kind.mime is None:
        return None
    mime = kind.mime
    if mime.startswith("image/"):
        return MediaMetadataType.IMAGE
    if mime.startswith("video/") or mime.startswith("audio/"):
        return MediaMetadataType.AV
    return None


def _build_metadata(
    media_type: MediaMetadataType,
    source: SourceData,
) -> MediaMetadataBase | None:
    if media_type is MediaMetadataType.IMAGE:
        return _build_image_metadata(source)
    if media_type is MediaMetadataType.AV:
        return _build_av_metadata(source)
    return None


def _build_image_metadata(source: SourceData) -> ImageMetadata | None:
    try:
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)

        with Image.open(source) as image:
            image.load()
            return ImageMetadata(
                width=image.width,
                height=image.height,
                format=image.format or "UNKNOWN",
                mode=image.mode or "UNKNOWN",
            )
    except UnidentifiedImageError:
        logger.debug("Failed to parse image metadata", exc_info=True)
        return None


def _build_av_metadata(source: SourceData) -> MediaMetadataBase | None:
    try:
        if isinstance(source, (bytes, bytearray)):
            return AVDecoder(io.BytesIO(source)).get_metadata()
        elif isinstance(source, io.IOBase):
            return AVDecoder(source).get_metadata()
        else:
            with source.open("rb") as stream:
                return AVDecoder(stream).get_metadata()
    except Exception:  # pragma: no cover - depends on external libs/media support
        logger.debug("Failed to parse AV metadata", exc_info=True)
        return None
