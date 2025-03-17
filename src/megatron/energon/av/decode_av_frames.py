# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import io
from collections.abc import Iterator
from fractions import Fraction
from typing import Literal, Optional, Sequence, Union, overload

import numpy as np
import torch

# Try importing optional dependencies
try:
    import av
    import filetype
    import soundfile as sf

    from megatron.energon.av.fastseek import Fastseek

    AV_DECODE_AVAILABLE = True
except ImportError as e:
    AV_DECODE_AVAILABLE = False
    MISSING_DEPENDENCY = str(e)

from megatron.energon.av.video_data import VideoData


class AVData:
    """A class that provides a flexible interface for decoding audio and video data.

    This class allows users to control decoding parameters at runtime rather than having them fixed
    during initialization. It's particularly useful for cases where different samples may need different
    decoding parameters.
    """

    def __init__(self, stream: io.BytesIO) -> None:
        if not AV_DECODE_AVAILABLE:
            raise ImportError(
                f"AV decoding is not available. Please install the required dependencies with:\n"
                f"pip install megatron-energon[av_decode]\n"
                f"Missing dependency: {MISSING_DEPENDENCY}"
            )
        self.stream = stream

    def get_frames(
        self,
        audio_clip_duration: int = 1,
        audio_num_clips: int = -1,
        video_decode_audio: bool = False,
        video_num_frames: int = 64,
        video_out_frame_size: tuple[int, int] = (224, 224),
    ) -> Optional[VideoData]:
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
            video, audio, metadata = self.decode_video_frames(
                num_frames=video_num_frames,
                out_frame_size=video_out_frame_size,
                decode_audio=video_decode_audio,
                num_clips=audio_num_clips,
                clip_duration=audio_clip_duration,
            )
        elif extension in ("flac", "mp3", "wav"):
            video = None
            audio, metadata = self.decode_audio_samples(
                num_clips=audio_num_clips,
                clip_duration=audio_clip_duration,
                audio_format=extension,
            )
        else:
            return None

        return VideoData(
            frames=video,
            aframes=audio,
            info=metadata,
        )

    def _get_extension(self) -> Optional[str]:
        """Get the file extension from the raw data."""
        # Try to guess the file type using the first few bytes
        self.stream.seek(0)  # Reset stream position before guessing
        ftype = filetype.guess(self.stream)
        if ftype is None:
            return None
        return ftype.extension

    def get_frame_batch(
        self,
        frame_indices: Sequence[int],
        out_frame_size: Optional[tuple[int, int]] = None,
        seeker: Optional["Fastseek"] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Gets a batch of frames at the given indices from a video file.

        Args:
            frame_indices: The indices of the frames to extract.
            out_frame_size: The size of the output frames. If None, use the original frame size.
            seeker: The seeker to use for seeking the video file. Defaults to a new seeker.

        Returns:
            A tuple containing the video frames and metadata
            The video tensor is in the shape (frames, channels, height, width)

        NOTE: indices should be expressed in the correct units:
            - mp4/mov: frame number
            - mkv/webm: time (in time_base units)
            - other (probe mode): frame number
        """
        if seeker is None:
            seeker = Fastseek(self.stream)

        self.stream.seek(0)  # Reset the video stream so that pyav can read the entire container

        with av.open(self.stream) as input_container:
            # Grab video & audio streams
            video_stream = input_container.streams.video[0]
            if len(input_container.streams.audio) > 0:
                audio_stream = input_container.streams.audio[0]
                audio_fps = audio_stream.sample_rate or 0
            else:
                audio_fps = 0

            # Enable multi-threaded decode for video
            # TODO: This causes a bug which leads to a deadlock in ffmpeg when deallocating the object.
            # Thus, disable for now.
            # video_stream.thread_type = 3
            video_stream.thread_type = 0

            # Collect metadata
            video_fps = float(video_stream.average_rate) if video_stream.average_rate else 0.0

            metadata = {"video_fps": video_fps, "audio_fps": audio_fps}

            # Pre-calculate timing info for video
            average_rate: Fraction = video_stream.average_rate
            time_base: Fraction = video_stream.time_base
            average_frame_duration: int = int(1 / average_rate / time_base)

            frame_iterator: Iterator[av.VideoFrame] = input_container.decode(video=0)
            previous_frame_index: int = 0

            frames: list[torch.Tensor] = []
            for target_frame_index in frame_indices:
                if (
                    iframe_info := seeker.should_seek(previous_frame_index, target_frame_index)
                ) is not None:
                    input_container.seek(iframe_info.pts, stream=input_container.streams.video[0])
                    previous_frame_index = iframe_info.index

                for i, frame in enumerate(frame_iterator):
                    # Container uses frame counts, we can find the exact target frame by counting from the iframe which is at a known offset
                    if seeker.unit == "count" and previous_frame_index + i == target_frame_index:
                        break

                    # Container uses time, the target frame might not correspond exactly to any metadata but the desired timestamp should
                    # fall within a frames display period
                    if (
                        seeker.unit == "time"
                        and frame.pts <= target_frame_index <= frame.pts + average_frame_duration
                    ):
                        break

                if out_frame_size is not None:
                    frame = frame.reformat(
                        width=out_frame_size[0],
                        height=out_frame_size[1],
                        format="rgb24",
                        interpolation="BILINEAR",
                    )
                else:
                    frame = frame.reformat(format="rgb24")

                frames.append(torch.from_numpy(frame.to_ndarray()))

                previous_frame_index = target_frame_index + 1

        # Stack video frames along dim=0 => [frames, channels, height, width]
        video_tensor = torch.stack(frames)
        # Convert video tensor from (frames, height, width, channels) to (frames, channels, height, width)
        video_tensor = video_tensor.permute((0, 3, 1, 2))
        return video_tensor, metadata

    @overload
    def decode_video_frames(
        self,
        num_frames: int = -1,
        *,
        out_frame_size: Optional[tuple[int, int]] = None,
        decode_audio: Literal[True],
        num_clips: int = 1,
        clip_duration: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]: ...

    @overload
    def decode_video_frames(
        self,
        num_frames: int = -1,
        *,
        out_frame_size: Optional[tuple[int, int]] = None,
        decode_audio: bool = False,
        num_clips: int = 1,
        clip_duration: int = 1,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict]: ...

    def decode_video_frames(
        self,
        num_frames: int = -1,
        *,
        out_frame_size: Optional[tuple[int, int]] = None,
        decode_audio: bool = False,
        num_clips: int = 1,
        clip_duration: int = 1,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """Decode video frames and optionally audio from a video file.

        This method extracts frames from a video file at evenly spaced intervals. If requested,
        it can also extract audio clips from the video. The method supports various video
        container formats and handles both frame-based (MP4/MOV) and time-based (Matroska/WebM)
        seeking.

        Args:
            num_frames: Number of frames to extract. If -1, extracts all frames. Defaults to -1.
            out_frame_size: Desired output frame size as (width, height).
                If None, keeps original frame size. Defaults to None.
            decode_audio: Whether to decode audio from the video. Defaults to False.
            num_clips: Number of audio clips to extract if decode_audio is True.
                If -1, extracts a single clip from the entire audio. Defaults to 1.
            clip_duration: Duration of each audio clip in seconds. Only used if decode_audio is
                True. Defaults to 1.

        Returns:
            A tuple containing:
                - video_tensor: Tensor of shape [num_frames, channels, height, width] containing
                              the decoded video frames. Values are in range [0, 255].
                - audio_tensor: Tensor containing the decoded audio clips if decode_audio is True,
                              otherwise an empty tensor.
                - metadata: Dictionary containing video and audio metadata (fps, sample rate, etc.).

        Note:
            The method uses the Fastseek class to optimize frame seeking, which determines
            whether to use frame numbers or timestamps based on the container format.
        """
        seeker: "Fastseek" = Fastseek(self.stream)
        self.stream.seek(0)

        # --- First, decode video frames ---
        with av.open(self.stream) as input_container:
            if seeker.unit == "count":
                if input_container.streams.video[0].frames != 0:
                    upper_bound = input_container.streams.video[0].frames - 1
                    frame_count = input_container.streams.video[0].frames
                else:  # Need to count
                    packets = [p for p in input_container.demux(video=0) if p.pts is not None]
                    upper_bound = len(packets) - 1
                    frame_count = len(packets)
            elif seeker.unit == "time":
                if input_container.streams.video[0].duration is not None:
                    upper_bound = input_container.streams.video[0].duration
                    packets = [p for p in input_container.demux(video=0) if p.pts is not None]
                    frame_count = len(packets)
                else:
                    packets = [p for p in input_container.demux(video=0) if p.pts is not None]
                    upper_bound = packets[-1].pts + packets[-1].duration
                    frame_count = len(packets)

        if num_frames == -1:
            num_frames = frame_count

        # Pick which video frames to extract
        frame_indices = np.linspace(0, upper_bound, num_frames, dtype=int).tolist()
        video_tensor, metadata = self.get_frame_batch(frame_indices, out_frame_size, seeker)

        # --- Then, if requested, decode audio using the same clip logic as decode_audio_samples ---
        audio_tensor = None
        if decode_audio:
            # Open the container again to get sample_count and sampling_rate
            self.stream.seek(0)  # Reset stream position
            with av.open(self.stream) as input_container:
                audio_stream = input_container.streams.audio[0]
                sample_count = audio_stream.duration
                sampling_rate = audio_stream.rate
                assert sample_count is not None

            if num_clips == -1:
                # Single clip from the entire audio
                clip_indices = [(0, sample_count - 1)]
            else:
                clip_indices = self.get_clip_indices(
                    sampling_rate, sample_count, num_clips, clip_duration
                )

            # Actually read the audio clips
            self.stream.seek(0)  # Reset stream position
            audio_tensor, audio_metadata = self.get_audio_batch(clip_indices)
            # Merge any extra audio metadata
            metadata.update(audio_metadata)

        return video_tensor, audio_tensor, metadata

    def get_clip_indices(
        self,
        sampling_rate: int,
        total_samples: int,
        num_clips: int,
        clip_duration_sec: int,
    ) -> list[tuple[int, int]]:
        """Calculate indices for audio clips based on sampling rate and duration.

        Args:
            sampling_rate: The sampling rate of the audio in Hz
            total_samples: Total number of samples in the audio
            num_clips: Number of clips to extract
            clip_duration_sec: Duration of each clip in seconds

        Returns:
            List of lists containing (start_idx, end_idx) for each clip
        """
        clip_samples = int(sampling_rate * clip_duration_sec)
        clip_samples = min(clip_samples, total_samples)  # Don't exceed total length

        if num_clips == 1:
            return [(0, clip_samples - 1)]

        # If total length can accommodate all clips without overlap, space them out evenly
        if num_clips * clip_samples <= total_samples:
            spacing = total_samples // num_clips
        else:
            # Overlap: distribute clips so first starts at 0 and last ends at total_samples - clip_samples
            spacing = (total_samples - clip_samples) // (num_clips - 1)

        return [(i * spacing, i * spacing + clip_samples - 1) for i in range(num_clips)]

    def get_audio_batch(
        self,
        clip_indices: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, dict]:
        """
        Gets a batch of audio samples at the given indices from an audio file.
        Indices correspond to the original sample rate.

        Args:
            clip_indices: List of (start_idx, end_idx) pairs for each clip

        Returns:
            Tuple of (audio_tensor, metadata) where audio_tensor has shape [num_clips, channels, samples]
        """
        self.stream.seek(0)

        with av.open(self.stream) as input_container:
            audio_stream = input_container.streams.audio[0]
            orig_rate = audio_stream.sample_rate
            duration_per_sample = 1 / orig_rate
            metadata = {"audio_fps": orig_rate}

            if len(clip_indices) == 0:
                # Empty result
                return torch.zeros(0, audio_stream.channels, 0), metadata

            clips = []
            expected_samples = (
                clip_indices[0][1] - clip_indices[0][0] + 1
            )  # Expected samples per clip

            # First pass: decode all clips and find max length
            for start_idx, end_idx in clip_indices:
                start_time = start_idx * duration_per_sample
                end_time = end_idx * duration_per_sample

                # Seek near start time (convert to microseconds per PyAV docs)
                input_container.seek(int(start_time * av.time_base))

                decoded_samples = []
                for frame in input_container.decode(audio=0):
                    frame_start = frame.pts * frame.time_base
                    # Stop decoding if we've passed the end
                    if frame_start >= end_time:
                        break

                    frame_nd = frame.to_ndarray()  # (channels, samples)
                    decoded_samples.append(frame_nd)

                if decoded_samples:
                    # Combine all channels/samples into one array
                    clip_all = np.concatenate(decoded_samples, axis=-1)  # (channels, total_samples)

                    # Ensure we get exactly the expected number of samples
                    if clip_all.shape[-1] > expected_samples:
                        clip_all = clip_all[:, :expected_samples]
                    elif clip_all.shape[-1] < expected_samples:
                        # Pad with zeros if we got fewer samples than expected
                        padded = np.zeros(
                            (clip_all.shape[0], expected_samples), dtype=clip_all.dtype
                        )
                        padded[:, : clip_all.shape[-1]] = clip_all
                        clip_all = padded

                    clips.append(clip_all)

            # Convert to torch and stack
            return torch.stack([torch.from_numpy(clip) for clip in clips]), metadata

    def decode_audio_samples(
        self,
        num_clips: int = 1,
        clip_duration: int = 1,
        audio_format: str = "flac",
    ) -> tuple[torch.Tensor, dict]:
        """Decode audio samples from an audio file.

        This method extracts audio clips from various audio formats. For WAV files, it uses
        soundfile for direct decoding. For other formats (FLAC, MP3), it uses PyAV for
        decoding. The method can extract multiple clips of specified duration from the audio.

        Args:
            num_clips: Number of audio clips to extract. If -1, extracts a single clip from the entire audio. Defaults to 1.
            clip_duration: Duration of each clip in seconds. Defaults to 1.
            audio_format: Format of the input audio file. Supported formats are "wav", "flac", and "mp3". Defaults to "flac".

        Returns:
            A tuple containing:
                - video_tensor: None (since this is audio-only decoding)
                - audio_tensor: Tensor of shape [num_clips, channels, samples] containing
                              the decoded audio clips. For WAV files, a single clip is
                              returned as [channels, samples].
                - metadata: Dictionary containing audio metadata (sample rate, etc.)

        Note:
            For WAV files, the entire requested duration is read at once using soundfile.
            For other formats, the method uses get_clip_indices to determine clip positions
            and get_audio_batch to extract the clips.
        """
        if audio_format == "wav":
            with sf.SoundFile(self.stream) as f:
                sample_rate = f.samplerate
                target_length_in_samples = min(f.frames, int(clip_duration * sample_rate))

                f.seek(0)
                waveform = f.read(frames=target_length_in_samples, dtype="float32")

                metadata = {"audio_fps": f.samplerate}
                audio_tensor = torch.from_numpy(waveform)

        else:
            with av.open(self.stream) as input_container:
                sample_count = input_container.streams.audio[0].duration
                sampling_rate = input_container.streams.audio[0].rate
                assert sample_count is not None

            if num_clips == -1:
                num_clips = 1
                clip_indices = [(0, sample_count - 1)]
            else:
                clip_indices = self.get_clip_indices(
                    sampling_rate, sample_count, num_clips, clip_duration
                )

            self.stream.seek(0)  # Reset stream position
            audio_tensor, metadata = self.get_audio_batch(clip_indices)

        return audio_tensor, metadata


class AVDecoder:
    """A decoder class for audio and video data that provides a consistent interface for decoding media files.

    This class encapsulates the decoding parameters and provides a callable interface that can be used
    with webdataset or other data loading pipelines. It supports both video and audio decoding with
    configurable parameters for frame extraction, resizing, and audio clip extraction.

    Args:
        audio_clip_duration: Duration of each audio clip in seconds. Used when decoding audio from
            video files or standalone audio files.
        audio_num_clips: Number of audio clips to extract. If -1, extracts a single clip from the
            entire audio duration.
        video_decode_audio: Whether to decode audio from video files. If True, audio will be
            extracted alongside video frames.
        video_num_frames: Number of video frames to extract. If -1, extracts all frames.
        video_out_frame_size: Output size for video frames as (width, height). If None, frames
            are returned at their original resolution.
        video_decode: If "AVData", returns an AVData instance for flexible decoding. If "torch",
            returns decoded VideoData.

    Example:
        >>> decoder = AVDecoder(
        ...     audio_clip_duration=3,
        ...     audio_num_clips=5,
        ...     video_decode_audio=True,
        ...     video_num_frames=64,
        ...     video_out_frame_size=(224, 224)
        ... )
        >>> result = decoder("video.mp4", video_bytes)
    """

    def __init__(
        self,
        audio_clip_duration: int,
        audio_num_clips: int,
        video_decode_audio: bool,
        video_num_frames: int,
        video_out_frame_size: tuple[int, int],
        video_decode: Literal["torch", "AVData"] = "torch",
    ) -> None:
        if not AV_DECODE_AVAILABLE:
            raise ImportError(
                f"AV decoding is not available. Please install the required dependencies with:\n"
                f"pip install megatron-energon[av_decode]\n"
                f"Missing dependency: {MISSING_DEPENDENCY}"
            )

        self.audio_clip_duration = audio_clip_duration
        self.audio_num_clips = audio_num_clips
        self.video_decode_audio = video_decode_audio
        self.video_num_frames = video_num_frames
        self.video_out_frame_size = video_out_frame_size
        self.video_decode = video_decode

    def read_av_data(self, key: str, data: bytes) -> AVData:
        """Decoder function that returns an AVData object for flexible decoding.

        Args:
            key: The file extension or key
            data: The raw bytes of the media file

        Returns:
            AVData object that can be used to decode the media with custom parameters
        """
        return AVData(io.BytesIO(data))

    def __call__(self, key: str, data: bytes) -> Optional[Union[VideoData, AVData]]:
        """
        Extract the video or audio data from default media extensions.

        Args:
            key: media file extension
            data: raw media bytes

        Returns:
            If video_decode is "torch", returns VideoData containing the decoded frames and metadata.
            If video_decode is "AVData", returns an AVData instance for flexible decoding.
            Returns None if decoding failed or file type is not supported.
        """
        if not any(
            key == ext or key.endswith("." + ext)
            for ext in ("mp4", "mov", "webm", "mkv", "flac", "mp3", "wav")
        ):
            return None
        av_data = self.read_av_data(key, data)
        if av_data is None:
            return None
        if self.video_decode == "AVData":
            return av_data
        return av_data.get_frames(
            audio_clip_duration=self.audio_clip_duration,
            audio_num_clips=self.audio_num_clips,
            video_decode_audio=self.video_decode_audio,
            video_num_frames=self.video_num_frames,
            video_out_frame_size=self.video_out_frame_size,
        )
