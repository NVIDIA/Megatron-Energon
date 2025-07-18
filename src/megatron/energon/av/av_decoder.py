# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import io
import warnings
from collections.abc import Iterator
from fractions import Fraction
from typing import BinaryIO, Literal, Optional, Sequence, Union, overload

import numpy as np
import torch

from megatron.energon.edataclass import edataclass

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

    seeker: "Fastseek"
    stream: BinaryIO
    suppress_warnings: bool

    def __init__(self, stream: BinaryIO, suppress_warnings: bool = False) -> None:
        if not AV_DECODE_AVAILABLE:
            raise ImportError(
                f"AV decoding is not available. Please install the required dependencies with:\n"
                f"pip install megatron-energon[av_decode]\n"
                f"Missing dependency: {MISSING_DEPENDENCY}. Install megatron-energon[av_decode] to use AVDecoder."
            )
        self.stream = stream
        self.suppress_warnings = suppress_warnings

        try:
            self.seeker = Fastseek(self.stream)
        except ValueError:
            self.stream.seek(0)
            self.seeker = Fastseek(self.stream, probe=True)

        self.stream.seek(0)

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

        with av.open(self.stream, "r") as input_container:
            initialize_av_container(input_container)

            assert len(input_container.streams.video) > 0, (
                "No video stream found, but video_clips are requested"
            )

            video_stream = input_container.streams.video[0]

            # Pre-calculate timing info for video
            average_rate: Fraction = video_stream.average_rate  # Frames per second
            assert average_rate, "Video stream has no FPS."

            time_base: Fraction = video_stream.time_base  # Seconds per PTS unit

            if video_clip_ranges is not None:
                # Convert video_clip_ranges to seeker unit
                if video_unit == "frames" and self.seeker.unit == "pts":
                    # Convert from frames to pts units
                    video_clip_ranges = [
                        (
                            clip[0] / average_rate / time_base,
                            clip[1] / average_rate / time_base,
                        )
                        for clip in video_clip_ranges
                    ]

                    if not self.suppress_warnings:
                        warnings.warn(
                            "Video container unit is frames, but seeking in time units. The resulting frames may be slightly off.",
                            RuntimeWarning,
                        )
                elif video_unit == "seconds" and self.seeker.unit == "frames":
                    # Convert from seconds to frames
                    video_clip_ranges = [
                        (
                            clip[0] * average_rate,
                            clip[1] * average_rate,
                        )
                        for clip in video_clip_ranges
                    ]
                    if not self.suppress_warnings:
                        warnings.warn(
                            "Video container unit is time units, but seeking using frame number. The resulting frames may be slightly off.",
                            RuntimeWarning,
                        )
                elif video_unit == "seconds" and self.seeker.unit == "pts":
                    # Convert from seconds to pts units
                    video_clip_ranges = [
                        (clip[0] / time_base, clip[1] / time_base) for clip in video_clip_ranges
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

                for frame in frame_iterator:
                    take_frame = False
                    last_frame = False

                    # Container uses frame counts, we can find the exact target frame by counting from the iframe which is at a known offset
                    if self.seeker.unit == "frames":
                        if previous_frame_index >= start_frame_index:
                            take_frame = True
                        if end_frame_index is not None and previous_frame_index >= end_frame_index:
                            last_frame = True

                    # Container uses time, the target frame might not correspond exactly to any metadata but the desired timestamp should
                    # fall within a frames display period
                    if self.seeker.unit == "pts":
                        if start_frame_index <= (frame.pts + frame.duration):
                            take_frame = True
                        if end_frame_index is not None and end_frame_index <= (
                            frame.pts + frame.duration
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

                        clip_timestamp_end = float((frame.pts + frame.duration) * frame.time_base)

                    previous_frame_index += 1

                    if last_frame:
                        break

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

        with av.open(self.stream, "r") as input_container:
            initialize_av_container(input_container)
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

                if end_time != float("inf"):
                    desired_duration = end_time - start_time
                    desired_sample_count = int(desired_duration * audio_sample_rate + 0.5)
                else:
                    desired_sample_count = None

                clip_start_time = None
                clip_end_time = None

                decoded_samples = []
                decoded_sample_count = 0

                previous_frame = None
                for frame in input_container.decode(audio=0):
                    assert frame.pts is not None, "Audio frame has no PTS timestamp"
                    cur_frame_time = float(frame.pts * frame.time_base)
                    cur_frame_duration = float(frame.duration * frame.time_base)

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
                            decoded_sample_count += prev_frame_array.shape[1]
                            clip_start_time = start_time
                            clip_end_time = prev_start_time + cur_frame_duration
                        else:
                            clip_start_time = cur_frame_time

                    # Stop decoding if the end of the frame is past the end time
                    if cur_frame_time + cur_frame_duration >= end_time:
                        # Crop the last frame to the end time
                        last_frame_array = audio_frame_array(frame)
                        additional_samples = int(
                            (end_time - cur_frame_time) * audio_sample_rate + 0.5
                        )
                        projected_total_samples = decoded_sample_count + additional_samples
                        projected_total_samples = decoded_sample_count + additional_samples

                        if (
                            desired_sample_count is not None
                            and 0 < abs(projected_total_samples - desired_sample_count) < 2
                        ):
                            # We are within 2 samples of the desired duration, let's adjust
                            # the last frame so that we get the desired duration
                            additional_samples = desired_sample_count - decoded_sample_count

                        last_frame_array = last_frame_array[:, :additional_samples]
                        decoded_samples.append(last_frame_array)
                        decoded_sample_count += last_frame_array.shape[1]
                        clip_end_time = end_time
                        break

                    frame_nd = audio_frame_array(frame)  # (channels, samples)
                    decoded_samples.append(frame_nd)
                    decoded_sample_count += frame_nd.shape[1]
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
        """Decode the entire audio/video data and return an AVData object.

        Args:
            video_decode_audio: Whether to decode audio from video

        Returns:
            VideoData containing the decoded frames and metadata, or None if decoding failed
            The video tensor is in the shape (frames, channels, height, width)
            The audio tensor is in the shape (channels, samples)
        """
        extension = self._get_extension()
        if extension is not None:
            extension = extension.lower()

        if extension in ("mov", "mp4", "webm", "mkv", "avi", "m4v"):
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
        metadata = self.get_metadata(
            get_video=True,
            get_video_duration=False,
            get_video_frame_count=False,
            get_video_frame_size=False,
            get_audio=False,
        )
        assert metadata.video_fps is not None
        return metadata.video_fps

    def get_audio_samples_per_second(self) -> int:
        """Get the number of samples per second of the audio stream."""
        metadata = self.get_metadata(
            get_video=False,
            get_audio=True,
            get_audio_duration=False,
        )
        assert metadata.audio_sample_rate is not None
        return metadata.audio_sample_rate

    def has_audio_stream(self) -> bool:
        """Check if the stream has an audio stream."""
        self.stream.seek(0)
        with av.open(self.stream, "r") as input_container:
            initialize_av_container(input_container)
            return len(input_container.streams.audio) > 0

    def has_video_stream(self) -> bool:
        """Check if the stream has a video stream."""
        self.stream.seek(0)
        with av.open(self.stream, "r") as input_container:
            initialize_av_container(input_container)
            return len(input_container.streams.video) > 0

    def get_audio_duration(self) -> Optional[float]:
        """Get the duration of the audio stream.

        Returns:
            The duration of the audio stream in seconds
        """

        metadata = self.get_metadata(
            get_video=False,
            get_audio=True,
            get_audio_duration=True,
        )
        return metadata.audio_duration

    @overload
    def get_video_duration(self, get_frame_count: Literal[True]) -> tuple[Optional[float], int]: ...

    @overload
    def get_video_duration(
        self, get_frame_count: bool = False
    ) -> tuple[Optional[float], Optional[int]]: ...

    def get_video_duration(
        self, get_frame_count: bool = False
    ) -> tuple[Optional[float], Optional[int]]:
        """Get the duration of the video stream.

        Args:
            get_frame_count: Whether to return the number of frames in the video. This is a more costly operation.

        Returns:
            A tuple containing the duration in seconds, and the number of frames in the video
        """

        metadata = self.get_metadata(
            get_video=True,
            get_video_duration=True,
            get_video_frame_count=get_frame_count,
            get_video_frame_size=False,
            get_audio=False,
            get_audio_duration=False,
        )
        return metadata.video_duration, metadata.video_num_frames

    def get_metadata(
        self,
        get_video: bool = True,
        get_video_duration: bool = True,
        get_video_frame_count: bool = True,
        get_video_frame_size: bool = True,
        get_audio: bool = True,
        get_audio_duration: bool = True,
    ) -> "AVMetadata":
        """Get the metadata of the media object.

        Args:
            get_video: Compute video metadata.
            get_video_duration: Compute video duration if not found in header.
            get_video_frame_count: Compute video frame count if not found in header.
            get_video_frame_size: Compute video frame size if not found in header.
            get_audio: Compute audio metadata.
            get_audio_duration: Compute audio duration if not found in header.
        """
        self.stream.seek(0)
        with av.open(self.stream, "r") as input_container:
            initialize_av_container(input_container)

            metadata = AVMetadata()

            if get_video and input_container.streams.video:
                video_stream = input_container.streams.video[0]

                metadata.video_duration = video_stream.duration

                if get_video_duration and metadata.video_duration is None:
                    # If duration isn't found in header the whole video is decoded to
                    # determine the duration.
                    metadata.video_num_frames = 0
                    last_packet = None
                    for packet in input_container.demux(video=0):
                        if packet.pts is not None:
                            metadata.video_num_frames += 1
                            last_packet = packet

                    if last_packet is not None and last_packet.duration is not None:
                        assert last_packet.pts is not None
                        metadata.video_duration = last_packet.pts + last_packet.duration
                if metadata.video_duration is not None:
                    if video_stream.start_time is not None:
                        metadata.video_duration -= video_stream.start_time
                    if video_stream.time_base is not None:
                        metadata.video_duration *= float(video_stream.time_base)

                if get_video_frame_count and metadata.video_num_frames is None:
                    metadata.video_num_frames = sum(
                        1 for p in input_container.demux(video=0) if p.pts is not None
                    )

                if video_stream.average_rate is not None:
                    metadata.video_fps = float(video_stream.average_rate)
                elif metadata.video_num_frames is not None and metadata.video_duration is not None:
                    metadata.video_fps = metadata.video_num_frames / metadata.video_duration
                if get_video_frame_size:
                    input_container.seek(0)
                    for first_frame in input_container.decode(video=0):
                        metadata.video_width = first_frame.width
                        metadata.video_height = first_frame.height
                        break
                    else:
                        metadata.video_width = video_stream.width
                        metadata.video_height = video_stream.height

            if get_audio and input_container.streams.audio:
                audio_stream = input_container.streams.audio[0]
                metadata.audio_sample_rate = audio_stream.sample_rate
                metadata.audio_duration = audio_stream.duration
                if get_audio_duration and metadata.audio_duration is None:
                    last_packet = None
                    input_container.seek(0)
                    for packet in input_container.demux(audio=0):
                        if packet.pts is not None:
                            last_packet = packet

                    if last_packet is not None and last_packet.duration is not None:
                        assert last_packet.pts is not None
                        metadata.audio_duration = last_packet.pts + last_packet.duration

                if metadata.audio_duration is not None:
                    if audio_stream.start_time is not None:
                        metadata.audio_duration -= audio_stream.start_time
                    if audio_stream.time_base is not None:
                        metadata.audio_duration *= float(audio_stream.time_base)

                metadata.audio_channels = audio_stream.channels

        return metadata

    def __repr__(self):
        return f"AVDecoder(stream={self.stream!r})"


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

    def read_av_data(self, data: bytes) -> AVDecoder:
        """Decoder function that returns an AVData object for flexible decoding.

        Args:
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
        key = key.lower()
        if not any(
            key == ext or key.endswith("." + ext)
            for ext in ("mp4", "avi", "mov", "webm", "mkv", "flac", "mp3", "wav")
        ):
            return None

        av_decoder = self.read_av_data(data)

        if self.av_decode == "AVDecoder":
            return av_decoder
        elif self.av_decode == "pyav":
            input_container = av.open(av_decoder.stream)
            initialize_av_container(input_container)
            return input_container
        elif self.av_decode == "torch":
            return av_decoder.get_frames(
                video_decode_audio=self.video_decode_audio,
            )
        else:
            raise ValueError(f"Invalid av_decode value: {self.av_decode}")


@edataclass
class AVMetadata:
    """Metadata of the media object."""

    video_duration: Optional[float] = None
    video_num_frames: Optional[int] = None
    video_fps: Optional[float] = None
    video_width: Optional[int] = None
    video_height: Optional[int] = None

    audio_duration: Optional[float] = None
    audio_channels: Optional[int] = None
    audio_sample_rate: Optional[int] = None


def initialize_av_container(input_container: "av.container.InputContainer") -> None:
    """Every PyAV container should be initialized with this function.

    This function ensures that no additional threads are created.
    This is to avoid deadlocks in ffmpeg when when deallocating the container.
    Furthermore, we cannot have multiple threads before forking the process when
    using torch data loaders with multiple workers.
    """

    for stream in input_container.streams:
        cc = stream.codec_context

        if cc is not None:
            cc.thread_type = "NONE"
            cc.thread_count = 0
