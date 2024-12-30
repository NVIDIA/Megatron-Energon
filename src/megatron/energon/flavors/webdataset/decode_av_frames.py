# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import io
from argparse import ArgumentParser
from collections.abc import Collection, Iterator
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch

from megatron.energon.flavors.webdataset.fastseek import Fastseek

def frame_to_ts(frame: int, average_rate: Fraction, time_base: Fraction) -> int:
    return int(frame / average_rate / time_base)


def ts_to_frame(ts: int, average_rate: Fraction, time_base: Fraction) -> int:
    return int(ts * time_base * average_rate)


def get_frame_batch(
    video_file: io.BytesIO,
    frame_indices: Collection[int],
    out_frame_size: tuple = None,
    decode_audio: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Gets a batch of frames at the given indices from a video file."""
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
        video_fps = float(video_stream.average_rate) if video_stream.average_rate else 0.0
        audio_fps = audio_stream.sample_rate or 0
        metadata = {"video_fps": video_fps, "audio_fps": audio_fps}

        # Pre-calculate timing info for video
        average_rate: Fraction = video_stream.average_rate
        time_base: Fraction = video_stream.time_base
        average_frame_duration: int = int(1 / average_rate / time_base)

        frame_iterator: Iterator[av.VideoFrame] = input_container.decode(video=0)
        previous_frame_number: int = 0

        frames: list[torch.Tensor] = []
                # Decode requested video frames
        frames = []
        for target_frame_number in frame_indices:
            if seeker.mime in ["video/x-matroska", "video/webm"]:
                # Matroska uses time rather than frame number
                prev_frame_ts = frame_to_ts(
                    previous_frame_number, average_rate, seeker.container_time_base
                )
                target_frame_ts = frame_to_ts(
                    target_frame_number, average_rate, seeker.container_time_base
                )
            else:
                prev_frame_ts = previous_frame_number
                target_frame_ts = target_frame_number

            target_pts = frame_to_ts(target_frame_number, average_rate, time_base)

            if seeker.should_seek(prev_frame_ts, target_frame_ts):
                input_container.seek(target_pts, stream=video_stream)

            for frame in frame_iterator:
                if (
                    frame.pts
                    <= target_pts + (average_frame_duration / 2)
                    <= frame.pts + average_frame_duration
                ):
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
                    break

            previous_frame_number = target_frame_number

        # Decode all audio frames (or just a subset if you prefer)
        audio_frames = []
        if decode_audio:
            audio_iterator = input_container.decode(audio=0)
            for audio_frame in audio_iterator:
                # Convert audio frame to a NumPy array (shape: channels x samples)
                audio_nd = audio_frame.to_ndarray()
                audio_frames.append(torch.from_numpy(audio_nd))

    # Stack video frames along dim=0 => [batch_size, channels, height, width]
    video_tensor = torch.stack(frames)

    # Depending on how you want to handle audio of varying lengths, you might
    # cat them into one large tensor or return them as a list. Here's a simple cat:
    if audio_frames:
        # This will produce a shape like: [num_audio_frames, channels, samples]
        audio_tensor = torch.cat([af.unsqueeze(0) for af in audio_frames], dim=0)
    else:
        audio_tensor = torch.empty(0)  # or None

    return video_tensor, audio_tensor, metadata


def decode_video_frames(data: bytes, num_frames: int = -1, out_frame_size: tuple = None, decode_audio: bool = False):

    byte_stream = io.BytesIO(data)

    with av.open(byte_stream) as input_container:
        if input_container.streams.video[0].frames != 0:
            frame_count = input_container.streams.video[0].frames
        else:  # Need to count
            frame_count = len(
                [p for p in input_container.demux(video=0) if p.pts is not None]
            )

    if num_frames == -1:
        num_frames = frame_count

    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int).tolist()
    video_tensor, audio_tensor, metadata = get_frame_batch(byte_stream, frame_indices, out_frame_size, decode_audio)

    return video_tensor, audio_tensor, metadata


def get_audio_batch(
    audio_file: io.BytesIO,
    clip_indices: Collection[Collection[int]]
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Gets a batch of audio_samples at the given indices from an audio file.

    Indices can be a list of lists of individual samples, i.e. multiple individual clips of audio
    """
    audio_file.seek(
        0
    )  # Reset the audio stream so that pyav can read the entire container

    with av.open(audio_file) as input_container:
        # Grab audio stream
        audio_stream = input_container.streams.audio[0]

        # Collect metadata
        sampling_rate = audio_stream.sample_rate or 0
        metadata = {"audio_fps": sampling_rate}

        clips = []
        duration_per_sample = 1 / sampling_rate

        for indices in clip_indices:
            # Get the start and end times of the clip in seconds
            start_time = indices[0] * duration_per_sample
            end_time = indices[-1] * duration_per_sample

            # Seek to the start time
            input_container.seek(int(start_time * av.time_base))

            # Decode audio frames for the duration of the clip
            decoded_samples = []
            current_time = start_time
            for audio_frame in input_container.decode(audio=0):
                # Check if we've gone past the end of the clip
                frame_start_time = audio_frame.pts * audio_frame.time_base
                if frame_start_time >= end_time:
                    break

                # Convert audio frame to a NumPy array (shape: channels x samples)
                audio_nd = audio_frame.to_ndarray()

                # Append only the samples that fall within the clip's range
                for channel_data in audio_nd:
                    decoded_samples.append(channel_data)

                # Update current time (not strictly necessary but can help debugging)
                current_time += audio_nd.shape[1] / sampling_rate

            # Combine samples and append to clips list
            if decoded_samples:
                clip_tensor = torch.from_numpy(np.concatenate(decoded_samples, axis=-1))
                if len(indices) > 2:
                    clips.append(clip_tensor[:len(indices)])  # Trim to exact number of samples in clip
                else:
                    clips.append(clip_tensor)

        # Stack all clips into a batched tensor
        return torch.stack(clips), metadata


def get_clip_indices(sampling_rate, total_samples, num_clips, clip_duration_sec):
    # Calculate clip length in samples
    clip_samples = int(clip_duration_sec * sampling_rate)

    if num_clips == 1:
        # Single clip, center it
        start = (total_samples - clip_samples) // 2
        return [np.arange(start, start + clip_samples)]

    # Spacing between clip centers
    spacing = (total_samples - clip_samples) / (num_clips - 1)

    # Calculate start indices for each clip
    start_indices = [int(i * spacing) for i in range(num_clips)]

    # Get the range of indices for each clip
    clip_indices = [np.arange(start, start + clip_samples) for start in start_indices]

    return clip_indices


def decode_audio_samples(data: bytes, num_clips: int = 1, clip_duration: int = 1):

    byte_stream = io.BytesIO(data)

    with av.open(byte_stream) as input_container:
        if input_container.streams.audio[0].frames != 0:
            sample_count = input_container.streams.audio[0].frames
            sampling_rate = input_container.streams.audio[0].rate
        else:  # Need to count
            sampling_rate = input_container.streams.audio[0].rate
            sample_count = input_container.streams.audio[0].duration * sampling_rate

    if num_clips == -1:
        num_clips = 1
        clip_indices = [[0, sample_count - 1]]
    else:
        clip_indices = get_clip_indices(sampling_rate, sample_count, num_clips, clip_duration)

    audio_tensor, metadata = get_audio_batch(byte_stream, clip_indices)

    return None, audio_tensor, metadata
