# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import BinaryIO, Union

import filetype
from PIL import Image, UnidentifiedImageError

from megatron.energon.av import AVDecoder
from megatron.energon.media.metadata import ImageMetadata, MediaMetadataBase, MediaMetadataType

logger = logging.getLogger(__name__)


SourceData = Union[bytes, Path, BinaryIO]


class MediaFilterStrategy(str, Enum):
    """Strategy used to decide whether an entry should be treated as media."""

    EXTENSION = "EXTENSION"  # by file extension
    HEADER = "HEADER"  # by header filetype (detected using filetype package)
    GLOB = "GLOB"  # by one or more glob patterns (e.g. '*.jpg')


@dataclass(frozen=True)
class MediaFilterConfig:
    """Configuration for media detection during dataset preparation."""

    strategy: MediaFilterStrategy
    patterns: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, glob: str | None, header: bool, extension: bool) -> "MediaFilterConfig":
        # Check that exactly one of the strategies is enabled
        strategy_count = sum(bool(s) for s in [glob, header, extension])

        if strategy_count != 1:
            raise ValueError(
                "Exactly one of GLOB, HEADER, or EXTENSION media filters must be enabled. "
                "You can use multiple glob patterns by separating them by commas."
            )

        if glob:
            if "," in glob:
                patterns = glob.split(",")
            else:
                patterns = [glob]
            return cls(strategy=MediaFilterStrategy.GLOB, patterns=patterns)
        if header:
            return cls(strategy=MediaFilterStrategy.HEADER)
        if extension:
            return cls(strategy=MediaFilterStrategy.EXTENSION)
        assert False, "Internal error: Should not be reached"

    def should_consider_all(self) -> bool:
        """Check whether all files need to be considered for metadata extraction.
        This is the case, if we need to inspect the file content to determine the media type."""

        return self.strategy == MediaFilterStrategy.HEADER

    def should_consider_media(self, name: str) -> bool:
        """Check whether a file name qualifies for metadata extraction under the filter.
        This is a first stage check to avoid loading the file content into memory if possible."""

        lower_name = name.lower()

        if self.strategy == MediaFilterStrategy.HEADER:
            # TYPE detection relies on file content, hence it always requires inspection.
            return True

        if self.strategy == MediaFilterStrategy.EXTENSION:
            return _guess_type_from_extension(lower_name) is not None

        assert self.patterns is not None, "Pattern strategy requires a glob expression"
        return any(fnmatch(lower_name, pattern) for pattern in self.patterns)

    def extract_metadata(
        self,
        source: SourceData,
        filename: str | None = None,
    ) -> MediaMetadataBase | None:
        """Extract media metadata from the source, if the file is a media file according to the filter.
        If the file is found not to be a media file, None is returned.

        Args:
            source: The source data to extract metadata from. This can be a bytes, Path, or an open file.
            filename: The filename of the source data. This is required when extracting metadata from bytes or an open file.

        Returns:
            The media metadata, if the file is a media file according to the filter. None otherwise.
        """

        if isinstance(source, (bytes, bytearray, io.IOBase)):
            assert filename is not None, (
                "Filename is required when extracting metadata from bytes or IOBase"
            )
        else:
            assert filename is None, "Filename is not allowed when extracting metadata from path"
            filename = source.name

        media_type = _detect_media_type(filename, self, source)

        if media_type is None:
            return None

        metadata = _build_metadata(media_type, source)
        if metadata is None:
            return None
        return metadata


_IMAGE_EXTENSIONS: set[str] = {
    "bmp",
    "gif",
    "ico",
    "j2k",
    "jp2",
    "jpx",
    "jpeg",
    "jpg",
    "png",
    "tif",
    "tiff",
    "webp",
}

_AV_EXTENSIONS: set[str] = {
    "aac",
    "avi",
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


def _detect_media_type(
    name: str,
    config: MediaFilterConfig,
    source: SourceData,
) -> MediaMetadataType | None:
    # Case 1: GLOB strategy
    if config.strategy == MediaFilterStrategy.GLOB:
        if not any(fnmatch(name, pattern) for pattern in config.patterns):
            return None

    extension_guess = _guess_type_from_extension(name)

    # Case 2: EXTENSION strategy
    if config.strategy == MediaFilterStrategy.EXTENSION:
        return extension_guess

    # Case 3: HEADER strategy
    assert config.strategy == MediaFilterStrategy.HEADER, (
        "Internal error: Unexpected media filter strategy"
    )

    detected = _guess_type_from_filetype(source)
    return detected if detected is not None else extension_guess


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
            return AVDecoder(io.BytesIO(source)).get_metadata(get_audio_num_samples=True)
        elif isinstance(source, io.IOBase):
            return AVDecoder(source).get_metadata(get_audio_num_samples=True)
        else:
            with source.open("rb") as stream:
                return AVDecoder(stream).get_metadata(get_audio_num_samples=True)
    except Exception:  # pragma: no cover - depends on external libs/media support
        logger.debug("Failed to parse AV metadata", exc_info=True)
        return None
