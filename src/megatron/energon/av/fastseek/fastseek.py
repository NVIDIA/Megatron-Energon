# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from bisect import bisect_right
from typing import Optional

import filetype
from bitstring.bits import BitsType

from .containers.matroska import parse_matroska
from .containers.probe import parse_probe
from .keyframeinfo import KeyframeInfo


class StreamInfo:
    """Keyframe information about a video stream."""

    __slots__ = ("id", "keyframe_pts", "keyframe_indexes", "keyframes")

    #: The stream id.
    id: int
    #: The keyframes, sorted by frame index.
    keyframes: list[KeyframeInfo]
    #: The PTS times of the keyframes, sorted by PTS.
    keyframe_pts: list[int] | None
    #: The frame indexes of the keyframes if access by keyframes is allowed.
    keyframe_indexes: list[int] | None

    def __init__(self, id: int, keyframes: list[KeyframeInfo]) -> None:
        """Initialize the StreamInfo object. Given an unsorted list of KeyframeInfos, it provides attributes to access those sorted by PTS or by frame index.

        Args:
            id: The stream id.
            keyframes: The list of KeyframeInfos.
            has_frame_index: Whether frame indexes are supported for this container.
        """
        self.id = id
        self.keyframes = sorted(keyframes, key=lambda x: (x.index or -1, x.pts or -1))
        if all(x.pts is None for x in keyframes):
            self.keyframe_pts = None
        else:
            self.keyframe_pts = sorted(x.pts for x in keyframes if x.pts is not None)
        if all(x.index is None for x in keyframes):
            self.keyframe_indexes = None
        else:
            self.keyframe_indexes = [x.index for x in self.keyframes if x.index is not None]

    def __repr__(self) -> str:
        return f"StreamInfo(id={self.id}, keyframe_pts={self.keyframe_pts}, keyframe_indexes={self.keyframe_indexes}, keyframes={self.keyframes})"


class Fastseek:
    """
    Gathers information from the video container file (e.g. metadata which requires minimal decoding)
    to find keyframes in the video for fast seeking.

    Information is returned in the form of KeyframeInfo structures which can be used by a decoding loop
    to make informed decisions about the best seeking behavior

    Currently supports:
    - Matroska/WebM: frames are indexed by time and inter-frame duration must be accounted for to get to the right frame. Use force_probe=True to use pyav to get frame-accurate keyframes.
    - All other formats: Use pyav to find keyframes.

    Frames will be indexed by number. This is not as fast as using a supported container but is still
    significantly faster than sequential decoding.
    """

    #: Keyframe info by stream id
    keyframes: dict[int, StreamInfo]

    #: List of stream ids for indexed access.
    streams: list[int]
    #: Whether frame indexes are supported for this container.
    #: If True, supports frame index access.
    frame_index_supported: bool
    #: Whether PTS are supported for this container.
    #: If True, supports PTS access.
    pts_supported: bool
    #: MIME type of the container.
    mime: str

    def __init__(self, file: BitsType, force_probe: bool = False) -> None:
        """Initialize the Fastseek object.

        Args:
            file: The video file data as a bitstring BitsType object. This should contain the raw bytes of the video file.
            force_probe: If True, use ffmpeg to probe the stream without decoding. This may be slower but works with any container format.
                   If False (default), attempt to parse the container format directly (only optimized for matroska).
        """
        if force_probe:
            keyframes = parse_probe(file)
        else:
            ftype = filetype.guess(file)

            if ftype is None:
                raise ValueError(
                    "Unable to determine file type (hint: try passing probe=True to the Fastseek constructor)"
                )

            self.mime = ftype.mime

            if ftype.mime in ("video/x-matroska", "video/webm"):
                keyframes = parse_matroska(file)
            else:
                keyframes = parse_probe(file)

        if len(keyframes) == 0:
            raise ValueError(
                f"The parser for {ftype.mime} was unable to find any streams (hint: try passing probe=True to the Fastseek constructor)"
            )

        if all(len(kf) == 0 for kf in keyframes.values()):
            raise ValueError(
                f"The parser for {ftype.mime} was unable to find any keyframes (hint: try passing probe=True to the Fastseek constructor)"
            )

        self.keyframes = {k: StreamInfo(k, keyframes) for k, keyframes in keyframes.items()}
        self.frame_index_supported = any(
            stream.keyframe_indexes is not None for stream in self.keyframes.values()
        )
        self.pts_supported = any(
            stream.keyframe_pts is not None for stream in self.keyframes.values()
        )
        self.streams = list(self.keyframes.keys())

    def should_seek_by_frame(
        self, current_frame_index: int, target_frame_index: int, stream: int = 0
    ) -> Optional[KeyframeInfo]:
        """Determine if seeking to a keyframe is necessary to reach the target frame.

        This method helps optimize video seeking by determining whether a seek operation
        is needed to reach the target frame. It returns information about the nearest
        keyframe only if seeking would be beneficial (i.e., if sequential decoding from
        the current position would be less efficient).

        Args:
            current_frame_index: The current frame number
            target_frame_index: The desired frame number to seek to
            stream: The video stream index to use. Defaults to 0.

        Returns:
            Information about the nearest keyframe if seeking would be beneficial,
            or None if sequential decoding from current position is more efficient.
        """
        nearest_iframe: KeyframeInfo = self.nearest_keyframe_by_frame(target_frame_index, stream)
        return (
            nearest_iframe
            if (current_frame_index < nearest_iframe.index <= target_frame_index)
            or (target_frame_index < current_frame_index)
            else None
        )

    def nearest_keyframe_by_frame(self, target_frame_index: int, stream: int = 0) -> KeyframeInfo:
        """Find the nearest keyframe that comes before the target frame.

        This method performs a binary search to find the keyframe that is closest to,
        but not after, the target frame position. This is useful for determining the
        optimal starting point for decoding to reach a specific frame.

        Args:
            target_frame_index: The target frame number to find the nearest keyframe for.
            stream: The video stream index to use. Defaults to 0.
                Used when the container has multiple video streams.

        Returns:
            Information about the nearest keyframe before the target position.
            Contains details like the keyframe's position, timestamp, and file offset.

        Note:
            The implementation currently uses a list-based approach for stream selection
            as some video containers don't report track IDs correctly. This is a temporary
            workaround and may be updated in the future.
        """

        if stream >= len(self.streams):
            raise ValueError(f"No stream with index {stream}")

        assert self.frame_index_supported, "Frame indexes are not supported for this container"

        stream_id = self.streams[stream]
        stream_info = self.keyframes[stream_id]

        if len(stream_info.keyframes) == 0:
            raise ValueError(f"No keyframes found for stream {stream}")
        assert stream_info.keyframe_indexes is not None, (
            "Frame indexes are not supported for this container"
        )

        # bisect_right returns the rightmost insertion point, so subtracting 1 gives
        # us the index of the last keyframe with index <= target
        nearest_iframe_to_target_index: int = (
            bisect_right(stream_info.keyframe_indexes, target_frame_index) - 1
        )
        return stream_info.keyframes[max(0, nearest_iframe_to_target_index)]

    def should_seek_by_pts(self, current_pts: int, target_pts: int, stream: int = 0) -> int | None:
        """
        Find the nearest keyframe that comes before the target frame index and return its PTS.

        Args:
            current_pts: The PTS of the current frame to be decoded.
            target_pts: The target PTS to seek to.
            stream: The video stream index to use. Defaults to 0.
                Used when the container has multiple video streams.

        Returns:
            The PTS of the nearest keyframe at or before the target PTS.
            If no such keyframe is found, returns None.
        """
        if stream >= len(self.streams):
            raise ValueError(f"No stream with index {stream}")

        assert self.pts_supported, "PTS are not supported for this container"

        stream_id = self.streams[stream]
        stream_info = self.keyframes[stream_id]

        if stream_info.keyframe_pts is None or len(stream_info.keyframe_pts) == 0:
            raise ValueError(f"No keyframes found for stream {stream}")

        pos = bisect_right(stream_info.keyframe_pts, target_pts) - 1
        keyframe_pts = stream_info.keyframe_pts[max(0, pos)]

        return (
            keyframe_pts
            if (current_pts < keyframe_pts <= target_pts) or (target_pts < current_pts)
            else None
        )
