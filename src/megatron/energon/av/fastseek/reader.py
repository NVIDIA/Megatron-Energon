# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from abc import ABC, abstractmethod
from collections.abc import Generator

import av

from .fastseek import Fastseek


class FastseekReader(ABC):
    """A class that provides a interface for reading video frames from a video stream for frame range extraction."""

    def __init__(
        self, seeker: Fastseek, input_container: av.container.InputContainer, stream_idx: int = 0
    ) -> None:
        """Initialize the fastseek reader.

        Args:
            seeker: The Fastseek object to use for seeking.
            input_container: The pyav input container to read from.
            stream_idx: The index of the video stream to read from.
        """
        self.seeker = seeker
        self.input_container = input_container
        self.frame_iterator = input_container.decode(video=stream_idx)
        self.stream = input_container.streams.video[stream_idx]
        self.skipped = 0

    @abstractmethod
    def seek_read(
        self,
        range_start: int,
        range_end: int | None,
    ) -> Generator[av.VideoFrame, None, None]:
        """
        Read video frames from the video stream for the given range.
        Subsequent calls to this method must be called with increasing range_start/range_end values,
        i.e. range_start(call i) <= range_end(call i) <= range_start(call i+1).

        Args:
            range_start: The start of the range to read from. The type of the range is defined by the subclass.
            range_end: The end of the range to read from. The type of the range is defined by the subclass.

        Returns:
            A generator of video frames.
        """
        ...


class FastseekReaderByFrames(FastseekReader):
    """A video frame reader that seeks by frame index."""

    next_frame_index: int = 0
    last_frame: av.VideoFrame | None = None

    def seek_read(
        self, range_start: int, range_end: int | None
    ) -> Generator[av.VideoFrame, None, None]:
        frame_info = self.seeker.should_seek(self.next_frame_index, range_start)
        if frame_info is not None:
            self.input_container.seek(frame_info.pts, stream=self.stream)
            self.next_frame_index = frame_info.index
        next_idx = self.next_frame_index
        for next_idx, frame in zip(
            range(self.next_frame_index + 1, range_start + 1), self.frame_iterator
        ):
            # print(f"Skip frame {next_idx - 1} at {frame.pts} +{frame.duration}")
            pass
        self.skipped += next_idx - self.next_frame_index
        self.next_frame_index = next_idx

        frame = self.last_frame
        if frame is not None and self.next_frame_index == range_start:
            # Repeat the previous frame. This is allowed.
            yield frame
            if range_end is not None and self.next_frame_index > range_end:
                # Special case: User requested the last frame, not more.
                return
        for frame in self.frame_iterator:
            self.next_frame_index += 1
            yield frame
            if range_end is not None and self.next_frame_index > range_end:
                break
        self.last_frame = frame


class FastseekReaderByPts(FastseekReader):
    """A video frame reader that seeks by PTS."""

    next_frame_pts: int = 0
    next_keyframe_pts: int = 0
    last_frame: av.VideoFrame | None = None

    def seek_read(
        self, range_start: int, range_end: int | None
    ) -> Generator[av.VideoFrame, None, None]:
        if self.next_keyframe_pts < range_start:
            # print(f"Seeking to frame {self.next_keyframe_pts} for {range_start} from {self.next_frame_pts}")
            self.input_container.seek(range_start, stream=self.stream)
        skipped = 0
        frame = self.last_frame
        if frame is not None and range_start < (frame.pts + frame.duration):
            # Repeat the previous frame. This is allowed.
            yield frame
            if range_end is not None and (frame.pts + frame.duration) >= range_end:
                # Special case: User requested the last frame, not more.
                return
        for frame in self.frame_iterator:
            if range_start >= (frame.pts + frame.duration):
                skipped += 1
                continue
            yield frame
            if range_end is not None and (frame.pts + frame.duration) >= range_end:
                break
        self.last_frame = frame
        self.next_frame_pts = frame.pts + frame.duration
        self.next_keyframe_pts = self.seeker.get_next_keyframe_pts(frame.pts + frame.duration)
        # print(f"Next keyframe for {frame.pts + frame.duration} is {self.pts[max(0, bs-1): bs+2]} at {bs}")
        self.skipped += skipped
