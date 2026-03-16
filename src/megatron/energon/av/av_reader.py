# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Generator
from dataclasses import dataclass
from typing import Iterator, Union

import av
import av.container
import av.indexentries
import av.stream


@dataclass(slots=True)
class AVProbeIndexEntry:
    """A single entry in a ProbeIndex."""

    timestamp: int
    is_keyframe: bool


class AVProbeIndex:
    """An index built by demuxing the container to find keyframe positions.

    Used as a fallback for containers (e.g. AVI) where stream.index_entries
    is not populated on open.
    """

    _entries: list[AVProbeIndexEntry]
    _keyframe_timestamps: list[int]

    def __init__(self, container: av.container.InputContainer, stream_idx: int = 0) -> None:
        self._entries = []
        self._keyframe_timestamps = []

        for packet in container.demux(video=stream_idx):
            if packet.pts is None:
                continue
            entry = AVProbeIndexEntry(timestamp=packet.pts, is_keyframe=packet.is_keyframe)
            self._entries.append(entry)
            if packet.is_keyframe:
                self._keyframe_timestamps.append(packet.pts)

        container.seek(0)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> AVProbeIndexEntry:
        return self._entries[idx]

    def search_timestamp(
        self, timestamp: int, *, backward: bool = True, any_frame: bool = False
    ) -> int:
        """Find the index of the nearest keyframe to the given timestamp.

        Args:
            timestamp: The timestamp to search for.
            backward: If True, find the keyframe at or before the timestamp.
            any_frame: If True, search all frames, not just keyframes.

        Returns:
            The index into this ProbeIndex, or -1 if not found.
        """
        if any_frame:
            timestamps = [e.timestamp for e in self._entries]
        else:
            timestamps = self._keyframe_timestamps

        if not timestamps:
            return -1

        if backward:
            pos = bisect_right(timestamps, timestamp) - 1
            if pos < 0:
                return 0 if not any_frame else 0
            found_ts = timestamps[pos]
        else:
            pos = bisect_right(timestamps, timestamp - 1)
            if pos >= len(timestamps):
                return -1
            found_ts = timestamps[pos]

        # Map back to entry index if searching keyframes only
        if not any_frame:
            for i, e in enumerate(self._entries):
                if e.timestamp == found_ts and e.is_keyframe:
                    return i
            return -1
        else:
            return pos


class AVReader(ABC):
    """A class that provides an interface for reading video frames from a video stream for frame range extraction."""

    #: The input container to read from.
    input_container: av.container.InputContainer
    #: The iterator over the video frames.
    frame_iterator: Iterator[av.VideoFrame]
    #: The video stream to read from.
    stream: av.stream.Stream
    #: The index to use for seeking (either stream.index_entries or a ProbeIndex).
    index: Union[av.indexentries.IndexEntries, "AVProbeIndex"]
    #: Number of frames skipped by the reader. For statistical purposes.
    skipped: int

    def __init__(
        self,
        input_container: av.container.InputContainer,
        stream_idx: int = 0,
        index: Union[av.indexentries.IndexEntries, "AVProbeIndex", None] = None,
    ) -> None:
        """Initialize the reader.

        Args:
            input_container: The pyav input container to read from.
            stream_idx: The index of the video stream to read from.
            index: Optional index to use for seeking. If None, uses stream.index_entries.
        """
        self.input_container = input_container
        self.frame_iterator = input_container.decode(video=stream_idx)
        self.stream = input_container.streams.video[stream_idx]
        self.index = index if index is not None else self.stream.index_entries
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


class AVReaderByFrames(AVReader):
    """A video frame reader that seeks by frame index."""

    #: The next frame index that would be returned by the iterator.
    _next_frame_index: int = 0
    #: The previous frame that was returned by the iterator.
    _previous_frame: av.VideoFrame | None = None

    def seek_read(
        self, range_start: int, range_end: int | None
    ) -> Generator[av.VideoFrame, None, None]:
        assert range_end is None or range_start <= range_end, (
            f"Range start {range_start} must be less or equal than range end {range_end}"
        )

        target_ts = self.index[range_start].timestamp
        keyframe_frame_num = self.index.search_timestamp(target_ts)
        keyframe_ts = self.index[keyframe_frame_num].timestamp
        if (self._next_frame_index < keyframe_frame_num) or (range_start < self._next_frame_index):
            self.input_container.seek(keyframe_ts, stream=self.stream)
            self._next_frame_index = keyframe_frame_num
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


class AVReaderByPts(AVReader):
    """A video frame reader that seeks by PTS.

      NOTE: this assumes the container index uses PTS (mp4 uses DTS for example).
    """

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

        keyframe_frame_num = self.index.search_timestamp(range_start)
        keyframe_ts = self.index[keyframe_frame_num].timestamp

        if (self._next_frame_pts < keyframe_ts) or (range_start < self._next_frame_pts):
            self.input_container.seek(keyframe_ts, stream=self.stream)
            self._next_frame_pts = keyframe_ts
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
