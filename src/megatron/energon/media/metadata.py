# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from typing import Any, ClassVar, Dict, Mapping, Type, TypeVar

from megatron.energon.edataclass import edataclass


class MediaMetadataType(str, Enum):
    """Enumerates the supported media metadata payload kinds."""

    AV = "av"
    IMAGE = "image"

    @classmethod
    def from_string(cls, value: str) -> "MediaMetadataType":
        try:
            return cls(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported media metadata type: {value!r}") from exc


TMetadata = TypeVar("TMetadata", bound="MediaMetadataBase")


@edataclass
class MediaMetadataBase:
    """Base class for metadata payloads to support typed JSON storage."""

    metadata_type: ClassVar[MediaMetadataType]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable mapping representation."""

        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls: Type[TMetadata], payload: Mapping[str, Any]) -> TMetadata:
        """Construct the metadata object from its JSON representation."""

        return cls(**payload)


@edataclass
class AVMetadata(MediaMetadataBase):
    """Metadata of a video or audio asset."""

    metadata_type: ClassVar[MediaMetadataType] = MediaMetadataType.AV

    video_duration: float | None = None
    video_num_frames: int | None = None
    video_fps: float | None = None
    video_width: int | None = None
    video_height: int | None = None

    audio_duration: float | None = None
    audio_channels: int | None = None
    audio_sample_rate: int | None = None
    audio_num_samples: int | None = None


@edataclass
class ImageMetadata(MediaMetadataBase):
    """Metadata for an encoded image file."""

    metadata_type: ClassVar[MediaMetadataType] = MediaMetadataType.IMAGE

    width: int
    height: int
    format: str
    mode: str


_MEDIA_METADATA_REGISTRY: Dict[MediaMetadataType, Type[MediaMetadataBase]] = {
    MediaMetadataType.AV: AVMetadata,
    MediaMetadataType.IMAGE: ImageMetadata,
}


def serialize_media_metadata(metadata: MediaMetadataBase) -> tuple[MediaMetadataType, str]:
    """Serialise the metadata to a tuple of (type, json.dumps(payload))."""

    payload_json = json.dumps(metadata.to_dict(), separators=(",", ":"))
    return metadata.metadata_type, payload_json


def deserialize_media_metadata(
    metadata_type: str | MediaMetadataType,
    metadata_json: str,
) -> MediaMetadataBase:
    """Deserialize a metadata record from stored SQLite values."""

    if not isinstance(metadata_type, MediaMetadataType):
        metadata_type = MediaMetadataType.from_string(metadata_type)

    try:
        payload_cls = _MEDIA_METADATA_REGISTRY[metadata_type]
    except KeyError as exc:  # pragma: no cover - future proofing
        raise ValueError(f"Unsupported media metadata type: {metadata_type}") from exc

    payload_dict = json.loads(metadata_json) if metadata_json else {}
    return payload_cls.from_dict(payload_dict)
