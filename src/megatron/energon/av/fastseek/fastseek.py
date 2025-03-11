# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import filetype
from bitstring.bits import BitsType

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

    If your container is not list above, pass "probe=True" to the constructor, this will use ffmpeg to parse the stream
    without decoding it. Frames will be indexed by number. This is not as fast as using a supported container but is still
    significantly faster than sequential decoding.
    """

    def __init__(self, file: BitsType, probe: bool = False) -> None:
        """Initialize the Fastseek object.

        Args:
            file (BitsType): The video file data as a bitstring BitsType object. This should contain the raw bytes of the video file.
            probe (bool, optional): If True, use ffmpeg to probe the stream without decoding. This is slower but works with any container format.
                                  If False (default), attempt to parse the container format directly. Only works with MP4/MOV and Matroska/WebM.

        Raises:
            ValueError: If the file type cannot be determined or if the container format is not supported (when probe=False).
        """
        if probe:
            self.keyframes = parse_probe(file)
            self.unit = "count"
        else:
            ftype = filetype.guess(file)

            if ftype is None:
                raise ValueError("Unable to determine file type")

            self.mime = ftype.mime

            if ftype.mime in ["video/mp4", "video/quicktime"]:
                self.keyframes = parse_mpeg(file)
                self.unit = "count"
            elif ftype.mime in ["video/x-matroska", "video/webm"]:
                self.keyframes = parse_matroska(file)
                self.unit = "time"
            else:
                raise ValueError(
                    f"Unsupported container: {ftype.mime} (hint: try passing probe=True to the Fastseek constructor)"
                )

    def should_seek(self, current: int, target: int, stream: int = 0) -> KeyframeInfo | None:
        """Determine if seeking to a keyframe is necessary to reach the target frame.

        This method helps optimize video seeking by determining whether a seek operation
        is needed to reach the target frame. It returns information about the nearest
        keyframe only if seeking would be beneficial (i.e., if sequential decoding from
        the current position would be less efficient).

        Args:
            current (int): The current frame number or timestamp (depending on container format)
            target (int): The desired frame number or timestamp to seek to
            stream (int, optional): The video stream index to use. Defaults to 0.

        Returns:
            KeyframeInfo | None: Information about the nearest keyframe if seeking would be beneficial,
                               or None if sequential decoding from current position is more efficient.
                               The KeyframeInfo contains the keyframe's position and timing information.

        Note:
            The units for current and target depend on the container format:
            - For MP4/MOV: frame numbers (count-based)
            - For Matroska/WebM: timestamps (time-based)
        """
        nearest_iframe: int = self.nearest_keyframe(target, stream)
        return nearest_iframe if current < nearest_iframe.index <= target else None

    def nearest_keyframe(self, target: int, stream: int = 0) -> KeyframeInfo:
        """Find the nearest keyframe that comes before the target frame.

        This method performs a binary search to find the keyframe that is closest to,
        but not after, the target frame position. This is useful for determining the
        optimal starting point for decoding to reach a specific frame.

        Args:
            target (int): The target frame number or timestamp to find the nearest keyframe for.
                         The unit (frame count or timestamp) depends on the container format.
            stream (int, optional): The video stream index to use. Defaults to 0.
                                  Used when the container has multiple video streams.

        Returns:
            KeyframeInfo: Information about the nearest keyframe before the target position.
                         Contains details like the keyframe's position, timestamp, and file offset.

        Note:
            The implementation currently uses a list-based approach for stream selection
            as some video containers don't report track IDs correctly. This is a temporary
            workaround and may be updated in the future.
        """
        # HACK some videos don't report track ID correctly, so just use a list for now
        nearest_iframe_to_target_index: int = (
            list(self.keyframes.values())[stream].bisect_left(target) - 1
        )
        return list(self.keyframes.values())[stream][max(0, nearest_iframe_to_target_index)]
