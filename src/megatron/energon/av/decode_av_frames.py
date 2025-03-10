# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import io
from collections.abc import Collection, Iterator
from fractions import Fraction
from typing import Literal

# Try importing optional dependencies
try:
    import av
    import numpy as np
    import soundfile as sf
    import torch
    import filetype
    from megatron.energon.av.fastseek import Fastseek
    from megatron.energon.av.video_data import VideoData
    AV_DECODE_AVAILABLE = True
except ImportError as e:
    AV_DECODE_AVAILABLE = False
    MISSING_DEPENDENCY = str(e)


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
        ) -> VideoData | None:
        """Decode the audio/video data with the specified parameters.

        Args:
            audio_clip_duration: Duration of each audio clip in seconds
            audio_num_clips: Number of audio clips to extract (-1 for all)
            video_decode_audio: Whether to decode audio from video
            video_num_frames: Number of video frames to extract
            video_out_frame_size: Output size for video frames (width, height)

        Returns:
            VideoData containing the decoded frames and metadata, or None if decoding failed
        """
        extension = self._get_extension()
        if extension in ("mov", "mp4", "webm", "mkv"):
            media = self.decode_video_frames(
                self.stream,
                num_frames=video_num_frames,
                out_frame_size=video_out_frame_size,
                decode_audio=video_decode_audio,
                num_clips=audio_num_clips,
                clip_duration=audio_clip_duration,
            )
        elif extension in ("flac", "mp3", "wav"):
            media = self.decode_audio_samples(
                self.stream,
                num_clips=audio_num_clips,
                clip_duration=audio_clip_duration,
                audio_format=extension,
            )
        else:
            return None

        if media is not None:
            frames = media[0].permute((0, 3, 1, 2)) if media[0] is not None else None
            return VideoData(
                frames=frames,
                aframes=media[1],
                info=media[2],
            )
        return None

    def _get_extension(self) -> str | None:
        """Get the file extension from the raw data."""
        # Try to guess the file type using the first few bytes
        self.stream.seek(0)  # Reset stream position before guessing
        ftype = filetype.guess(self.stream)
        if ftype is None:
            return None
        return ftype.extension

    def get_frame_batch(
        self,
        video_file: io.BytesIO,
        frame_indices: Collection[int],
        out_frame_size: tuple = None,
        seeker: Fastseek | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Gets a batch of frames at the given indices from a video file.

        NOTE: indices should be expressed in the correct units:
            - mp4/mov: frame number
            - mkv/webm: time (in time_base units)
            - other (probe mode): frame number
        """
        if seeker is None:
            seeker: Fastseek = Fastseek(video_file)

        video_file.seek(
            0
        )  # Reset the video stream so that pyav can read the entire container

        with av.open(video_file) as input_container:
            # Grab video & audio streams
            video_stream = input_container.streams.video[0]
            audio_stream = input_container.streams.audio[0]

            # enable multi-threaded decode for video
            video_stream.thread_type = 3

            # Collect metadata
            video_fps = (
                float(video_stream.average_rate) if video_stream.average_rate else 0.0
            )
            audio_fps = audio_stream.sample_rate or 0
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
                    iframe_info := seeker.should_seek(
                        previous_frame_index, target_frame_index
                    )
                ) is not None:
                    input_container.seek(
                        iframe_info.pts, stream=input_container.streams.video[0]
                    )
                    previous_frame_index = iframe_info.index

                for i, frame in enumerate(frame_iterator):
                    # Container uses frame counts, we can find the exact target frame by counting from the iframe which is at a known offset
                    if (
                        seeker.unit == "count"
                        and previous_frame_index + i == target_frame_index
                    ):
                        break

                    # Container uses time, the target frame might not correspond exactly to any metadata but the desired timestamp should
                    # fall within a frames display period
                    if (
                        seeker.unit == "time"
                        and frame.pts
                        <= target_frame_index
                        <= frame.pts + average_frame_duration
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

        # Stack video frames along dim=0 => [batch_size, channels, height, width]
        video_tensor = torch.stack(frames)

        return video_tensor, metadata


    def decode_video_frames(
        self,
        stream: io.BytesIO,
        num_frames: int = -1,
        out_frame_size: tuple = None,
        decode_audio: bool = False,
        num_clips: int = 1,
        clip_duration: int = 1,
    ):
        seeker: Fastseek = Fastseek(stream)
        stream.seek(0)

        # --- First, decode video frames ---
        with av.open(stream) as input_container:
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
        video_tensor, metadata = self.get_frame_batch(
            stream, frame_indices, out_frame_size, seeker
        )

        # --- Then, if requested, decode audio using the same clip logic as decode_audio_samples ---
        audio_tensor = torch.empty(0)
        if decode_audio:
            # Open the container again to get sample_count and sampling_rate
            stream.seek(0)  # Reset stream position
            with av.open(stream) as input_container:
                audio_stream = input_container.streams.audio[0]
                sample_count = audio_stream.duration
                sampling_rate = audio_stream.rate

            if num_clips == -1:
                # Single clip from the entire audio
                clip_indices = [[0, sample_count - 1]]
            else:
                clip_indices = self.get_clip_indices(
                    sampling_rate, sample_count, num_clips, clip_duration
                )

            # Actually read the audio clips
            stream.seek(0)  # Reset stream position
            audio_tensor, audio_metadata = self.get_audio_batch(
                stream,
                clip_indices,
            )
            # Merge any extra audio metadata
            metadata.update(audio_metadata)

        return video_tensor, audio_tensor, metadata


    def get_clip_indices(
        self,
        sampling_rate: int,
        total_samples: int,
        num_clips: int,
        clip_duration_sec: int,
    ) -> list[list[int]]:
        """Calculate indices for audio clips based on sampling rate and duration.

        Args:
            sampling_rate: The sampling rate of the audio in Hz
            total_samples: Total number of samples in the audio
            num_clips: Number of clips to extract
            clip_duration_sec: Duration of each clip in seconds

        Returns:
            List of lists containing [start_idx, end_idx] for each clip
        """
        clip_samples = int(sampling_rate * clip_duration_sec)
        clip_samples = min(clip_samples, total_samples)  # Don't exceed total length

        if num_clips == 1:
            return [[0, clip_samples - 1]]

        # If total length can accommodate all clips without overlap, space them out evenly
        if num_clips * clip_samples <= total_samples:
            spacing = total_samples // num_clips
        else:
            # Overlap: distribute clips so first starts at 0 and last ends at total_samples - clip_samples
            spacing = (total_samples - clip_samples) // (num_clips - 1)

        return [[i * spacing, i * spacing + clip_samples - 1] for i in range(num_clips)]


    def get_audio_batch(
        self,
        audio_file: io.BytesIO,
        clip_indices: list[list[int]],
    ) -> tuple[torch.Tensor, dict]:
        """
        Gets a batch of audio samples at the given indices from an audio file.
        Indices correspond to the original sample rate.

        Args:
            audio_file: The audio file as a BytesIO stream
            clip_indices: List of [start_idx, end_idx] pairs for each clip

        Returns:
            Tuple of (audio_tensor, metadata) where audio_tensor has shape [num_clips, channels, samples]
        """
        audio_file.seek(0)

        with av.open(audio_file) as input_container:
            audio_stream = input_container.streams.audio[0]
            orig_rate = audio_stream.sample_rate
            duration_per_sample = 1 / orig_rate
            metadata = {"audio_fps": orig_rate}

            clips = []
            expected_samples = clip_indices[0][1] - clip_indices[0][0] + 1  # Expected samples per clip

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
                        padded = np.zeros((clip_all.shape[0], expected_samples), dtype=clip_all.dtype)
                        padded[:, :clip_all.shape[-1]] = clip_all
                        clip_all = padded

                    clips.append(clip_all)

            # Convert to torch and stack
            return torch.stack([torch.from_numpy(clip) for clip in clips]), metadata


    def decode_audio_samples(
            self,
            stream: io.BytesIO,
            num_clips: int = 1,
            clip_duration: int = 1,
            audio_format: str = "flac",
    ) -> tuple[torch.Tensor, dict]:
        if audio_format == "wav":
            with sf.SoundFile(stream) as f:
                sample_rate = f.samplerate
                target_length_in_samples = min(f.frames, int(clip_duration * sample_rate))

                f.seek(0)
                waveform = f.read(frames=target_length_in_samples, dtype='float32')

                metadata = {"audio_fps": f.samplerate}
                audio_tensor = torch.from_numpy(waveform)

        else:
            with av.open(stream) as input_container:
                sample_count = input_container.streams.audio[0].duration
                sampling_rate = input_container.streams.audio[0].rate

            if num_clips == -1:
                num_clips = 1
                clip_indices = [[0, sample_count - 1]]
            else:
                clip_indices = self.get_clip_indices(sampling_rate, sample_count, num_clips, clip_duration)

            stream.seek(0)  # Reset stream position
            audio_tensor, metadata = self.get_audio_batch(stream, clip_indices)

        return None, audio_tensor, metadata


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

    def __call__(self, key: str, data: bytes) -> VideoData | AVData | None:
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
        if not any(key.endswith(ext) for ext in ("mp4", "mov", "webm", "mkv", "flac", "mp3", "wav")):
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

