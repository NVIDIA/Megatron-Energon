# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Literal, Optional

import filetype
from bitstring.bits import BitsType
from sortedcontainers import SortedList

from .containers.matroska import parse_matroska
from .containers.mpeg import parse_mpeg
from .containers.probe import parse_probe
from .keyframeinfo import KeyframeInfo


class Fastseek:
    """
    Gathers information from the video container file (e.g. metadata which requires minimal decoding)
    to find keyframes in the video for fast seeking.

    Information is returned in the form of KeyframeInfo structures which can be used by a decoding loop
    to make informed decisions about the best seeking behavior

    Currently supports:
    - MP4/MOV: frames are indexed by number and frame counting can be used to get the exact frame
    - Matroska/WebM: frames are indexed by time and inter-frame duration must be accounted for to get to the right frame

    If your container is not listed above, pass "probe=True" to the constructor, this will use ffmpeg to parse the stream
    without decoding it. Frames will be indexed by number. This is not as fast as using a supported container but is still
    significantly faster than sequential decoding.
    """

    keyframes: dict[int, SortedList[KeyframeInfo]]
    unit: Literal["frames", "pts"]
    mime: str

    def __init__(self, file: BitsType, probe: bool = False) -> None:
        """Initialize the Fastseek object.

        Args:
            file: The video file data as a bitstring BitsType object. This should contain the raw bytes of the video file.
            probe: If True, use ffmpeg to probe the stream without decoding. This is slower but works with any container format.
                   If False (default), attempt to parse the container format directly. Only works with MP4/MOV and Matroska/WebM.

        Raises:
            ValueError: If the file type cannot be determined or if the container format is not supported (when probe=False).
        """
        if probe:
            self.keyframes = parse_probe(file)
            self.unit = "frames"
        else:
            ftype = filetype.guess(file)

            if ftype is None:
                raise ValueError(
                    "Unable to determine file type (hint: try passing probe=True to the Fastseek constructor)"
                )

            self.mime = ftype.mime

            if ftype.mime in ["video/mp4", "video/quicktime"]:
                self.keyframes = parse_mpeg(file)
                self.unit = "frames"
            elif ftype.mime in ["video/x-matroska", "video/webm"]:
                self.keyframes = parse_matroska(file)
                self.unit = "pts"
            else:
                raise ValueError(
                    f"Unsupported container: {ftype.mime} (hint: try passing probe=True to the Fastseek constructor)"
                )

            if len(self.keyframes) == 0:
                raise ValueError(
                    f"The parser for {ftype.mime} was unable to find any streams (hint: try passing probe=True to the Fastseek constructor)"
                )

            if all(len(kf) == 0 for kf in self.keyframes.values()):
                raise ValueError(
                    f"The parser for {ftype.mime} was unable to find any keyframes (hint: try passing probe=True to the Fastseek constructor)"
                )

    def should_seek(self, current: int, target: int, stream: int = 0) -> Optional[KeyframeInfo]:
        """Determine if seeking to a keyframe is necessary to reach the target frame.

        This method helps optimize video seeking by determining whether a seek operation
        is needed to reach the target frame. It returns information about the nearest
        keyframe only if seeking would be beneficial (i.e., if sequential decoding from
        the current position would be less efficient).

        Args:
            current: The current frame number or timestamp (depending on container format)
            target: The desired frame number or timestamp to seek to
            stream: The video stream index to use. Defaults to 0.

        Returns:
            Information about the nearest keyframe if seeking would be beneficial,
            or None if sequential decoding from current position is more efficient.
            The KeyframeInfo contains the keyframe's position and timing information.

        Note:
            The units for current and target depend on the container format:
            - For MP4/MOV: frame numbers (count-based)
            - For Matroska/WebM: timestamps (time-based)
        """
        nearest_iframe: KeyframeInfo = self.nearest_keyframe(target, stream)
        return (
            nearest_iframe
            if (current < nearest_iframe.index <= target) or (target < current)
            else None
        )

    def nearest_keyframe(self, target: int, stream: int = 0) -> KeyframeInfo:
        """Find the nearest keyframe that comes before the target frame.

        This method performs a binary search to find the keyframe that is closest to,
        but not after, the target frame position. This is useful for determining the
        optimal starting point for decoding to reach a specific frame.

        Args:
            target: The target frame number or timestamp to find the nearest keyframe for.
                The unit (frame count or timestamp) depends on the container format.
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

        if stream >= len(self.keyframes):
            raise ValueError(f"No stream with index {stream}")

        stream_id = list(self.keyframes.keys())[stream]

        if len(self.keyframes[stream_id]) == 0:
            raise ValueError(f"No keyframes found for stream {stream}")

        nearest_iframe_to_target_index: int = self.keyframes[stream_id].bisect_left(target) - 1
        return self.keyframes[stream_id][max(0, nearest_iframe_to_target_index)]
