# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Iterator

import av
import av.container
import av.stream

from .fastseek import Fastseek


class FastseekReader(ABC):
    """A class that provides a interface for reading video frames from a video stream for frame range extraction."""

    #: The Fastseek object to use for seeking.
    seeker: Fastseek
    #: The input container to read from.
    input_container: av.container.InputContainer
    #: The iterator over the video frames.
    frame_iterator: Iterator[av.VideoFrame]
    #: The video stream to read from.
    stream: av.stream.Stream
    #: Number of frames skipped by the reader. For statistical purposes.
    skipped: int

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
        `range_start <= range_end` must hold. If `range_start == range_end` and within the video range, one frame is returned.

        Args:
            range_start: The start of the range to read from. The type of the range is defined by the subclass.
            range_end: The end of the range to read from. The type of the range is defined by the subclass. If None, the range goes to the end of the video. This is inclusive!

        Returns:
            A generator of video frames.
        """
        ...


class FastseekReaderByFrames(FastseekReader):
    """A video frame reader that seeks by frame index."""

    #: The next frame index that would be returned by the iterator.
    _next_frame_index: int = 0
    #: The previous frame that was returned by the iterator.
    _previous_frame: av.VideoFrame | None = None

    def seek_read(
        self, range_start: int, range_end: int | None
    ) -> Generator[av.VideoFrame, None, None]:
        if (
            keyframe_info := self.seeker.should_seek_by_frame(
                self._next_frame_index - 1, range_start
            )
        ) is not None:
            seek_pts = keyframe_info.pts
            if seek_pts is None:
                # input_container.seek should seek to the nearest keyframe, which should be the requested one.
                seek_pts = range_start
            self.input_container.seek(seek_pts, stream=self.stream)
            assert keyframe_info.index is not None, "Frame index is required for this container"
            self._next_frame_index = keyframe_info.index
            frame = None
        else:
            frame = self._previous_frame
        # Skip the frames between the keyframe / previous frame and the requested range_start.
        next_idx = self._next_frame_index
        for next_idx, frame in zip(
            range(self._next_frame_index + 1, range_start + 1), self.frame_iterator
        ):
            pass
        self.skipped += next_idx - self._next_frame_index
        self._next_frame_index = next_idx

        if frame is not None and self._next_frame_index - 1 == range_start:
            # Repeat the previous frame.
            yield frame
            if range_end == range_start:
                # Special case: User requested the last frame again, not more.
                return
        for frame in self.frame_iterator:
            self._next_frame_index += 1
            self._previous_frame = frame
            yield frame
            if range_end is not None and self._next_frame_index > range_end:
                break


class FastseekReaderByPts(FastseekReader):
    """A video frame reader that seeks by PTS."""

    #: The PTS of the next frame that would be returned by the iterator.
    _next_frame_pts: int = 0
    #: The previous frame that was returned by the iterator.
    _previous_frame: av.VideoFrame | None = None

    def seek_read(
        self, range_start: int, range_end: int | None
    ) -> Generator[av.VideoFrame, None, None]:
        assert range_end is None or range_start <= range_end, (
            f"Range start {range_start} must be less or equal than range end {range_end}"
        )
        if (
            seek_keyframe_pts := self.seeker.should_seek_by_pts(self._next_frame_pts, range_start)
        ) is not None:
            # Seeking backward or forward beyond the next keyframe
            # print(f"Seeking to frame {self.next_keyframe_pts} for {range_start} from {self.next_frame_pts}")
            self.input_container.seek(seek_keyframe_pts, stream=self.stream)
            self._next_frame_pts = seek_keyframe_pts
            frame = self._previous_frame = None
        else:
            frame = self._previous_frame
        # Skip frames before start
        if frame is None or range_start >= (frame.pts + frame.duration):
            skipped = 0
            for frame in self.frame_iterator:
                if range_start < (frame.pts + frame.duration):
                    break
                skipped += 1
            else:
                # Out of the end of the video
                frame = None
            self.skipped += skipped
            if frame is not None:
                self._next_frame_pts = frame.pts
                self._previous_frame = frame
        if frame is None:
            # No frame available -> at the end of the video
            # Just keep the next_frame_pts as it was
            return
        # Yield at least the current frame. It's after the start.
        yield frame
        while range_end is None or (frame.pts + frame.duration) <= range_end:
            try:
                frame = next(self.frame_iterator)
            except StopIteration:
                self._previous_frame = None
                break
            # Store the current frame's PTS, because we can still access that frame!
            self._next_frame_pts = frame.pts
            self._previous_frame = frame
            yield frame
