# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from bisect import bisect_right
from typing import Optional

import filetype
from bitstring.bits import BitsType

from .containers.matroska import parse_matroska
from .containers.probe import parse_probe
from .keyframeinfo import KeyframeInfo


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

    keyframe_pts: dict[int, list[int]]
    keyframes_by_frames: dict[int, tuple[list[KeyframeInfo], list[int]]]
    streams: list[int]
    frames_supported: bool
    mime: str

    def __init__(self, file: BitsType, force_probe: bool = False) -> None:
        """Initialize the Fastseek object.

        Args:
            file: The video file data as a bitstring BitsType object. This should contain the raw bytes of the video file.
            force_probe: If True, use ffmpeg to probe the stream without decoding. This is slower but works with any container format.
                   If False (default), attempt to parse the container format directly. Only works with MP4/MOV and Matroska/WebM.

        Raises:
            ValueError: If the file type cannot be determined or if the container format is not supported (when probe=False).
        """
        if force_probe:
            keyframes = parse_probe(file)
            self.frames_supported = True
        else:
            ftype = filetype.guess(file)

            if ftype is None:
                raise ValueError(
                    "Unable to determine file type (hint: try passing probe=True to the Fastseek constructor)"
                )

            self.mime = ftype.mime

            if ftype.mime in ["video/x-matroska", "video/webm"]:
                keyframes = parse_matroska(file)
                self.frames_supported = False
            else:
                keyframes = parse_probe(file)
                self.frames_supported = True

        if len(keyframes) == 0:
            raise ValueError(
                f"The parser for {ftype.mime} was unable to find any streams (hint: try passing probe=True to the Fastseek constructor)"
            )

        if all(len(kf) == 0 for kf in keyframes.values()):
            raise ValueError(
                f"The parser for {ftype.mime} was unable to find any keyframes (hint: try passing probe=True to the Fastseek constructor)"
            )

        self.keyframe_pts = {k: sorted(x.pts for x in v) for k, v in keyframes.items()}
        if self.frames_supported:
            self.keyframes_by_frames = {
                k: (l := sorted(v, key=lambda x: x.index), [x.index for x in l])
                for k, v in keyframes.items()
            }
        self.streams = list(keyframes.keys())

    def should_seek(
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
        nearest_iframe: KeyframeInfo = self.nearest_keyframe(target_frame_index, stream)
        return (
            nearest_iframe
            if (current_frame_index < nearest_iframe.index <= target_frame_index)
            or (target_frame_index < current_frame_index)
            else None
        )

    def nearest_keyframe(self, target_frame_index: int, stream: int = 0) -> KeyframeInfo:
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

        if stream >= len(self.keyframes_by_frames):
            raise ValueError(f"No stream with index {stream}")

        stream_id = self.streams[stream]
        keyframes, frames = self.keyframes_by_frames[stream_id]

        if len(keyframes) == 0:
            raise ValueError(f"No keyframes found for stream {stream}")

        # bisect_right returns the rightmost insertion point, so subtracting 1 gives
        # us the index of the last keyframe with index <= target
        nearest_iframe_to_target_index: int = bisect_right(frames, target_frame_index) - 1
        return keyframes[max(0, nearest_iframe_to_target_index)]

    def get_next_keyframe_pts(self, current_frame_pts: int, stream: int = 0) -> int:
        """Find the PTS of the next keyframe after the current frame.

        Args:
            current_frame_pts: The PTS of the current frame.
            stream: The video stream index to use. Defaults to 0.
                Used when the container has multiple video streams.

        Returns:
            The PTS of the next keyframe after the current frame.
        """

        if stream >= len(self.streams):
            raise ValueError(f"No stream with index {stream}")

        pts = self.keyframe_pts[self.streams[stream]]

        if len(pts) == 0:
            raise ValueError(f"No keyframes found for stream {stream}")

        pos = bisect_right(pts, current_frame_pts)
        if pos == len(pts):
            return float("inf")
        return pts[pos]
