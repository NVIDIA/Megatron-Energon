# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import io
from collections.abc import Iterator
from fractions import Fraction
from typing import BinaryIO, Literal, Optional, Sequence, Union, overload

import numpy as np
import torch

# Try importing optional dependencies
try:
    import av
    import av.container
    import filetype

    from megatron.energon.av.fastseek import Fastseek

    AV_DECODE_AVAILABLE = True
except ImportError as e:
    AV_DECODE_AVAILABLE = False
    MISSING_DEPENDENCY = str(e)

from megatron.energon.av.av_data import AVData


class AVDecoder:
    """A class that provides a flexible interface for decoding audio and video data.

    This class allows users to control decoding parameters at runtime rather than having them fixed
    during initialization. It's particularly useful for cases where different samples may need different
    decoding parameters.
    """

    _seeker: Optional["Fastseek"]

    def __init__(self, stream: BinaryIO) -> None:
        if not AV_DECODE_AVAILABLE:
            raise ImportError(
                f"AV decoding is not available. Please install the required dependencies with:\n"
                f"pip install megatron-energon[av_decode]\n"
                f"Missing dependency: {MISSING_DEPENDENCY}. Install megatron-energon[av_decode] to use AVDecoder."
            )
        self.stream = stream
        self._seeker = None

    @property
    def seeker(self) -> Fastseek:
        if self._seeker is None:
            try:
                self._seeker = Fastseek(self.stream)
            except ValueError:
                self._seeker = Fastseek(self.stream, probe=True)
        return self._seeker

    def get_video(self) -> AVData:
        """Get the entire video data from the stream (without audio)."""

        video_clips, video_timestamps = self.get_video_clips(video_clip_ranges=[(0, float("inf"))])
        return AVData(
            video_clips=video_clips,
            video_timestamps=video_timestamps,
            audio_clips=[],
            audio_timestamps=[],
        )

    def get_video_clips(
        self,
        video_clip_ranges: Sequence[tuple[float, float]],
        video_unit: Literal["frames", "seconds"] = "seconds",
        video_out_frame_size: Optional[tuple[int, int]] = None,
    ) -> tuple[list[torch.Tensor], list[tuple[float, float]]]:
        """Get video clips from the video stream.

        Args:
            video_clip_ranges: List of video clip start and end positions in the given unit (see video_unit)
            video_unit: Unit of the video clip positions ("frames" for frame number, "seconds" for timestamp)
            video_out_frame_size: Output size for video frames (width, height), or None to use the original frame size

        Returns:
            A tuple containing:
                - video_clips: List of video clips
                - video_clips_timestamps: List of timestamps for each video clip start and end in seconds
        """

        assert video_unit in ("frames", "seconds")

        self.stream.seek(0)  # Reset the video stream so that pyav can read the entire container

        with av.open(self.stream) as input_container:
            assert len(input_container.streams.video) > 0, (
                "No video stream found, but video_clips are requested"
            )

            video_stream = input_container.streams.video[0]

            # Enable multi-threaded decode for video
            # TODO: This causes a bug which leads to a deadlock in ffmpeg when deallocating the object.
            # Thus, disable for now.
            # video_stream.thread_type = 3
            video_stream.thread_type = 0

            # Pre-calculate timing info for video
            average_rate: Fraction = video_stream.average_rate  # Frames per second
            assert average_rate, "Video stream has no FPS."

            time_base: Fraction = video_stream.time_base  # Seconds per PTS unit
            average_frame_duration: int = int(1 / average_rate / time_base)  # PTS units per frame

            if video_clip_ranges is not None and video_unit != self.seeker.unit:
                # Convert video_clip_ranges to video_unit
                if video_unit == "frames":
                    # Convert from frames to seconds
                    video_clip_ranges = [
                        (
                            clip[0] / average_rate,
                            clip[1] / average_rate,
                        )
                        for clip in video_clip_ranges
                    ]
                else:
                    # Convert from seconds to frames
                    video_clip_ranges = [
                        (
                            clip[0] * average_rate,
                            clip[1] * average_rate,
                        )
                        for clip in video_clip_ranges
                    ]

            frame_iterator: Iterator[av.VideoFrame] = input_container.decode(video=0)
            previous_frame_index: int = 0

            video_clips_frames: list[list[torch.Tensor]] = []
            video_clips_timestamps: list[tuple[float, float]] = []

            for video_clip_range in video_clip_ranges:
                start_frame_index, end_frame_index = video_clip_range

                # Convert to int if possible, set end to None if infinite
                start_frame_index = int(start_frame_index)
                end_frame_index = int(end_frame_index) if end_frame_index != float("inf") else None

                clip_frames: list[torch.Tensor] = []
                clip_timestamp_start = None
                clip_timestamp_end = None

                # Find start frame
                if (
                    iframe_info := self.seeker.should_seek(previous_frame_index, start_frame_index)
                ) is not None:
                    input_container.seek(iframe_info.pts, stream=input_container.streams.video[0])
                    previous_frame_index = iframe_info.index

                for i, frame in enumerate(frame_iterator):
                    take_frame = False
                    last_frame = False

                    # Container uses frame counts, we can find the exact target frame by counting from the iframe which is at a known offset
                    if self.seeker.unit == "frames":
                        if previous_frame_index + i >= start_frame_index:
                            take_frame = True
                        if (
                            end_frame_index is not None
                            and previous_frame_index + i >= end_frame_index
                        ):
                            last_frame = True

                    # Container uses time, the target frame might not correspond exactly to any metadata but the desired timestamp should
                    # fall within a frames display period
                    if self.seeker.unit == "seconds":
                        if start_frame_index <= frame.pts + average_frame_duration:
                            take_frame = True
                        if (
                            end_frame_index is not None
                            and end_frame_index <= frame.pts + average_frame_duration
                        ):
                            last_frame = True

                    if take_frame:
                        if video_out_frame_size is not None:
                            frame = frame.reformat(
                                width=video_out_frame_size[0],
                                height=video_out_frame_size[1],
                                format="rgb24",
                                interpolation="BILINEAR",
                            )
                        else:
                            frame = frame.reformat(format="rgb24")

                        clip_frames.append(torch.from_numpy(frame.to_ndarray()))
                        if clip_timestamp_start is None:
                            clip_timestamp_start = float(frame.pts * frame.time_base)

                        clip_timestamp_end = float(
                            (frame.pts + average_frame_duration) * frame.time_base
                        )

                    if last_frame:
                        break

                # Add the number of frames we iterated over to the previous frame index
                previous_frame_index += i + 1

                if clip_timestamp_start is not None and clip_timestamp_end is not None:
                    video_clips_frames.append(clip_frames)
                    video_clips_timestamps.append((clip_timestamp_start, clip_timestamp_end))

        # Stack frames within each clip
        out_video_clips = [
            torch.stack(clip_frames).permute((0, 3, 1, 2)) for clip_frames in video_clips_frames
        ]
        return out_video_clips, video_clips_timestamps

    def get_audio(self) -> AVData:
        """Get the entire audio data from the stream."""
        audio_clips, audio_timestamps = self.get_audio_clips(audio_clip_ranges=[(0, float("inf"))])
        return AVData(
            video_clips=[],
            video_timestamps=[],
            audio_clips=audio_clips,
            audio_timestamps=audio_timestamps,
        )

    def get_audio_clips(
        self,
        audio_clip_ranges: Sequence[tuple[float, float]],
        audio_unit: Literal["samples", "seconds"] = "seconds",
    ) -> tuple[list[torch.Tensor], list[tuple[float, float]]]:
        """Get audio clips from the audio stream.

        Args:
            audio_clip_ranges: List of audio clip start and end positions in the given unit (see audio_unit)
            audio_unit: Unit of the audio clip positions ("samples" for sample number, "seconds" for timestamp)

        Returns:
            A tuple containing:
                - audio_clips: List of audio clips
                - audio_clips_timestamps: List of timestamps for each audio clip start and end in seconds
        """

        assert audio_unit in ("samples", "seconds")

        self.stream.seek(0)  # Reset the video stream so that pyav can read the entire container

        with av.open(self.stream) as input_container:
            assert len(input_container.streams.audio) > 0, (
                "No audio stream found, but audio_clips are requested"
            )

            audio_stream = input_container.streams.audio[0]
            audio_sample_rate = audio_stream.sample_rate

            assert audio_sample_rate, "Audio streams without sample rate are not supported"

            if audio_unit == "samples":
                # Convert from samples to seconds
                audio_clip_ranges = [
                    (
                        float(clip[0] / audio_sample_rate),
                        float(clip[1] / audio_sample_rate),
                    )
                    for clip in audio_clip_ranges
                ]

            out_audio_clips: list[torch.Tensor] = []
            out_audio_clips_timestamps: list[tuple[float, float]] = []

            def audio_frame_array(frame: av.AudioFrame) -> np.ndarray:
                if frame.format.is_planar:
                    arr_processed = frame.to_ndarray()  # Already (channels, samples)
                else:
                    # Calculate the number of channels and samples
                    channels = int(frame.layout.nb_channels)
                    samples = int(frame.samples)
                    # Reshape the interleaved data to (samples, channels), then transpose to (channels, samples)
                    arr_processed = np.reshape(frame.to_ndarray(), (samples, channels)).transpose(
                        1, 0
                    )
                return arr_processed

            for start_time, end_time in audio_clip_ranges:
                # Seek near start time, but rounded down to the nearest frame
                input_container.seek(int(start_time * av.time_base))

                clip_start_time = None
                clip_end_time = None

                decoded_samples = []
                previous_frame = None
                for frame in input_container.decode(audio=0):
                    assert frame.pts is not None, "Audio frame has no PTS timestamp"
                    cur_frame_time = float(frame.pts * frame.time_base)
                    cur_frame_duration = float(frame.samples / audio_sample_rate)

                    if cur_frame_time < start_time:
                        # Skip frames before the start time
                        previous_frame = frame
                        continue

                    if clip_start_time is None:
                        # This is our first matching frame
                        if previous_frame is not None:
                            # We have a previous frame that we need to crop to the start time
                            prev_start_time = float(previous_frame.pts * previous_frame.time_base)
                            prev_frame_array = audio_frame_array(previous_frame)
                            prev_frame_array = prev_frame_array[
                                :, int((start_time - prev_start_time) * audio_sample_rate + 0.5) :
                            ]
                            decoded_samples.append(prev_frame_array)
                            clip_start_time = start_time
                            clip_end_time = prev_start_time + cur_frame_duration
                        else:
                            clip_start_time = cur_frame_time

                    # Stop decoding if the end of the frame is past the end time
                    if cur_frame_time + cur_frame_duration >= end_time:
                        # Crop the last frame to the end time
                        last_frame_array = audio_frame_array(frame)
                        last_frame_array = last_frame_array[
                            :, : int((end_time - cur_frame_time) * audio_sample_rate + 0.5)
                        ]
                        decoded_samples.append(last_frame_array)
                        clip_end_time = end_time
                        break

                    frame_nd = audio_frame_array(frame)  # (channels, samples)
                    decoded_samples.append(frame_nd)
                    clip_end_time = cur_frame_time + cur_frame_duration

                if decoded_samples:
                    # Combine all channels/samples along samples axis
                    clip_all = np.concatenate(decoded_samples, axis=-1)  # (channels, total_samples)
                    if clip_start_time is not None and clip_end_time is not None:
                        out_audio_clips.append(torch.from_numpy(clip_all))
                        out_audio_clips_timestamps.append((clip_start_time, clip_end_time))

        return out_audio_clips, out_audio_clips_timestamps

    def get_video_with_audio(self) -> AVData:
        """Get the entire video and audio data from the stream."""
        return self.get_clips(
            video_clip_ranges=[(0, float("inf"))],
            audio_clip_ranges=[(0, float("inf"))],
            video_unit="seconds",
            audio_unit="seconds",
        )

    def get_clips(
        self,
        video_clip_ranges: Optional[Sequence[tuple[float, float]]] = None,
        audio_clip_ranges: Optional[Sequence[tuple[float, float]]] = None,
        video_unit: Literal["frames", "seconds"] = "seconds",
        audio_unit: Literal["samples", "seconds"] = "seconds",
        video_out_frame_size: Optional[tuple[int, int]] = None,
    ) -> AVData:
        """Get clips from the video and/or audio streams.
        Given a list of (start, end) tuples, this method will decode the video and/or audio clips
        at the specified start and end times. The units of the start and end times are specified by
        the `video_unit` and `audio_unit` arguments.

        Args:
            video_clip_ranges: List of video clip start and end positions in the given unit (see video_unit)
            audio_clip_ranges: List of audio clip start and end positions in the given unit (see audio_unit)
            video_unit: Unit of the video clip positions ("frames" for frame number, "seconds" for timestamp)
            audio_unit: Unit of the audio clip positions ("samples" for sample number, "seconds" for timestamp)
            video_out_frame_size: Output size for video frames (width, height), or None to use the original frame size

        Returns:
            AVData containing the decoded video and audio clips
        """
        if video_clip_ranges is not None:
            ret_video_clips, ret_video_clips_timestamps = self.get_video_clips(
                video_clip_ranges, video_unit, video_out_frame_size
            )
        else:
            ret_video_clips = []
            ret_video_clips_timestamps = []

        if audio_clip_ranges is not None:
            ret_audio_clips, ret_audio_clips_timestamps = self.get_audio_clips(
                audio_clip_ranges, audio_unit
            )
        else:
            ret_audio_clips = []
            ret_audio_clips_timestamps = []

        return AVData(
            video_clips=ret_video_clips,
            video_timestamps=ret_video_clips_timestamps,
            audio_clips=ret_audio_clips,
            audio_timestamps=ret_audio_clips_timestamps,
        )

    def get_frames(
        self,
        video_decode_audio: bool = False,
    ) -> Optional[AVData]:
        """Decode the audio/video data with the specified parameters.

        Args:
            audio_clip_duration: Duration of each audio clip in seconds
            audio_num_clips: Number of audio clips to extract (-1 for all)
            video_decode_audio: Whether to decode audio from video
            video_num_frames: Number of video frames to extract
            video_out_frame_size: Output size for video frames (width, height)

        Returns:
            VideoData containing the decoded frames and metadata, or None if decoding failed
            The video tensor is in the shape (frames, channels, height, width)
            The audio tensor is in the shape (channels, samples)
        """
        extension = self._get_extension()
        if extension in ("mov", "mp4", "webm", "mkv"):
            if video_decode_audio:
                return self.get_video_with_audio()
            else:
                return self.get_video()
        elif extension in ("flac", "mp3", "wav"):
            return self.get_audio()
        else:
            return None

    def _get_extension(self) -> Optional[str]:
        """Get the file extension from the raw data."""
        # Try to guess the file type using the first few bytes
        self.stream.seek(0)  # Reset stream position before guessing
        ftype = filetype.guess(self.stream)
        if ftype is None:
            return None
        return ftype.extension

    def get_video_fps(self) -> float:
        """Get the FPS of the video stream."""
        self.stream.seek(0)
        with av.open(self.stream) as input_container:
            video_stream = input_container.streams.video[0]
            assert video_stream.average_rate is not None
            return float(video_stream.average_rate)

    def get_audio_samples_per_second(self) -> int:
        """Get the number of samples per second of the audio stream."""
        self.stream.seek(0)
        with av.open(self.stream) as input_container:
            audio_stream = input_container.streams.audio[0]
            assert audio_stream.sample_rate is not None
            return int(audio_stream.sample_rate)

    @overload
    def get_duration(self, get_frame_count: Literal[True]) -> tuple[float, int]: ...

    @overload
    def get_duration(self, get_frame_count: bool = False) -> tuple[float, Optional[int]]: ...

    def get_duration(self, get_frame_count: bool = False) -> tuple[float, Optional[int]]:
        """Get the duration of the video and/or audio stream. If both lengths are found, the maximum is returned.

        Args:
            get_frame_count: Whether to return the number of frames in the video. This is a more costly operation.

        Returns:
            A tuple containing the duration in seconds, and the number of frames in the video
        """

        video_duration = None
        audio_duration = None
        num_frames = None

        self.stream.seek(0)  # Reset the video stream so that pyav can read the entire container

        with av.open(self.stream) as input_container:
            if input_container.streams.video:
                video_stream = input_container.streams.video[0]
                assert video_stream.time_base is not None

                video_start_pts = video_stream.start_time
                if video_start_pts is None:
                    video_start_pts = 0.0

                video_duration = video_stream.duration

            if input_container.streams.audio:
                audio_time_base = input_container.streams.audio[0].time_base
                audio_start_pts = input_container.streams.audio[0].start_time
                if audio_start_pts is None:
                    audio_start_pts = 0.0

                audio_duration = input_container.streams.audio[0].duration

            if audio_duration is None and video_duration is None:
                # If duration isn't found in header the whole video is decoded to
                # determine the duration.
                packets = [p for p in input_container.demux(video=0) if p.pts is not None]
                num_frames = len(packets)
                video_duration = packets[-1].pts + packets[-1].duration

            # Take the largest duration of either video or audio duration
            if audio_duration is not None and video_duration is not None:
                duration = max(
                    int(video_duration - video_start_pts) * video_stream.time_base,
                    int(audio_duration - audio_start_pts) * audio_time_base,
                )
            elif video_duration is not None:
                duration = int(video_duration - video_start_pts) * video_stream.time_base
            elif audio_duration is not None:
                duration = int(audio_duration - audio_start_pts) * audio_time_base

            if get_frame_count and num_frames is None:
                packets = [p for p in input_container.demux(video=0) if p.pts is not None]
                num_frames = len(packets)

        return float(duration), num_frames


class AVWebdatasetDecoder:
    """A decoder class for audio and video data that provides a consistent interface for decoding media files.

    This class encapsulates the decoding parameters and provides a callable interface that can be used
    with webdataset or other data loading pipelines. It supports both video and audio decoding with
    configurable parameters for frame extraction, resizing, and audio clip extraction.

    Args:
        video_decode_audio: Whether to decode audio from video files. If True, audio will be
            extracted alongside video frames.
        av_decode: If "AVDecoder", returns an AVDecoder instance for flexible decoding. If "torch",
            returns decoded VideoData.

    Example:
        >>> decoder = AVWebdatasetDecoder(
        ...     video_decode_audio=True,
        ...     av_decode="AVDecoder"
        ... )
        >>> result = decoder("video.mp4", video_bytes)
    """

    def __init__(
        self,
        video_decode_audio: bool,
        av_decode: Literal["torch", "AVDecoder", "pyav"] = "AVDecoder",
    ) -> None:
        self.video_decode_audio = video_decode_audio
        self.av_decode = av_decode

    def read_av_data(self, key: str, data: bytes) -> AVDecoder:
        """Decoder function that returns an AVData object for flexible decoding.

        Args:
            key: The file extension or key
            data: The raw bytes of the media file

        Returns:
            AVData object that can be used to decode the media with custom parameters
        """
        return AVDecoder(io.BytesIO(data))

    def __call__(
        self, key: str, data: bytes
    ) -> Optional[
        Union[AVData, AVDecoder, "av.container.InputContainer", "av.container.OutputContainer"]
    ]:
        """
        Extract the video or audio data from default media extensions.

        Args:
            key: media file extension
            data: raw media bytes

        Returns:
            If av_decode is "torch", returns VideoData containing the decoded frames and metadata.
            If av_decode is "AVDecoder", returns an AVDecoder instance for flexible decoding.
            If av_decode is "pyav", returns an av.container.InputContainer or av.container.OutputContainer instance.
            Returns None if decoding failed or file type is not supported.
        """
        if not any(
            key == ext or key.endswith("." + ext)
            for ext in ("mp4", "mov", "webm", "mkv", "flac", "mp3", "wav")
        ):
            return None

        av_decoder = self.read_av_data(key, data)

        if self.av_decode == "AVDecoder":
            return av_decoder
        elif self.av_decode == "pyav":
            return av.open(av_decoder.stream)
        elif self.av_decode == "torch":
            return av_decoder.get_frames(
                video_decode_audio=self.video_decode_audio,
            )
        else:
            raise ValueError(f"Invalid av_decode value: {self.av_decode}")
