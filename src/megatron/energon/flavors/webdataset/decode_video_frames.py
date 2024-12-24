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
    out_frame_size: tuple,
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
                    frame = frame.reformat(
                        width=out_frame_size[0],
                        height=out_frame_size[1],
                        format="rgb24",
                        interpolation="BILINEAR",
                    )
                    frames.append(torch.from_numpy(frame.to_ndarray()))
                    break

            previous_frame_number = target_frame_number

        # Decode all audio frames (or just a subset if you prefer)
        audio_frames = []
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


def decode_video_frames(data: bytes, frames: int, out_frame_size: tuple):

    byte_stream = io.BytesIO(data)

    with av.open(byte_stream) as input_container:
        if input_container.streams.video[0].frames != 0:
            frame_count = input_container.streams.video[0].frames
        else:  # Need to count
            frame_count = len(
                [p for p in input_container.demux(video=0) if p.pts is not None]
            )

    frame_indices = np.linspace(0, frame_count - 1, frames, dtype=int).tolist()
    video_tensor, audio_tensor, metadata = get_frame_batch(byte_stream, frame_indices, out_frame_size)

    return video_tensor, audio_tensor, metadata