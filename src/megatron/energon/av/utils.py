# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Literal, Optional, Tuple, Union, overload

import numpy as np
import torch

from megatron.energon.av.av_decoder import AVData, AVDecoder


def get_clips_uniform(
    av_decoder: AVDecoder,
    clip_duration_seconds: float,
    num_clips: int,
    request_video: bool = False,
    request_audio: bool = False,
    video_out_frame_size: Optional[tuple[int, int]] = None,
) -> AVData:
    """Extracts a sequence of clips, such that each clip is of
    equal duration and the clips are equidistant from each other.

    Args:
        av_decoder: An AVDecoder instance.
        clip_duration_seconds: The duration of each clip in seconds.
        num_clips: The number of clips to extract.
        request_video: Whether to request video clips.
        request_audio: Whether to request audio clips.
        video_out_frame_size: The size of the video frames to output, or None to use the original size.

    Returns:
        An AVData object containing the extracted video and audio clips.
    """

    if not request_video and not request_audio:
        raise ValueError("You must request at least one of video or audio")

    video_duration = float("inf")
    audio_duration = float("inf")

    if request_video:
        video_duration, _ = av_decoder.get_video_duration()
        if video_duration is None:
            raise ValueError("No video duration found")

    if request_audio:
        audio_duration = av_decoder.get_audio_duration()
        if audio_duration is None:
            raise ValueError("No audio duration found")

    # Typically, audio and video don't have the exact same duration, so we take the minimum
    # so that we can safely extract clips of equal duration.
    total_duration = min(video_duration, audio_duration)

    assert total_duration != float("inf")

    if clip_duration_seconds == 0:
        # Special case of single frames: End point should be start of last frame
        video_fps = av_decoder.get_video_fps()
        video_spf = 1 / video_fps
        first_start_time = video_spf * 0.5
        last_start_time = total_duration - video_spf * 0.5
    else:
        first_start_time = 0
        last_start_time = total_duration - clip_duration_seconds

    clips = [
        (float(start_time), float(start_time + clip_duration_seconds))
        for start_time in np.linspace(first_start_time, last_start_time, num_clips)
    ]

    return av_decoder.get_clips(
        video_clip_ranges=clips if request_video else None,
        audio_clip_ranges=clips if request_audio else None,
        video_unit="seconds",
        audio_unit="seconds",
        video_out_frame_size=video_out_frame_size,
    )


@overload
def get_single_frames_uniform(
    av_decoder: "AVDecoder",
    num_frames: int,
    *,
    video_out_frame_size: Optional[Tuple[int, int]] = None,
    return_timestamps: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def get_single_frames_uniform(
    av_decoder: "AVDecoder",
    num_frames: int,
    *,
    video_out_frame_size: Optional[Tuple[int, int]] = None,
    return_timestamps: Literal[True],
) -> Tuple[torch.Tensor, List[float]]: ...


def get_single_frames_uniform(
    av_decoder: AVDecoder,
    num_frames: int,
    *,
    video_out_frame_size: Optional[tuple[int, int]] = None,
    return_timestamps: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, list[float]]]:
    """Extracts a sequence of clips, such that each clip contains
    only a single frame and the frames are equidistant from each other.

    Args:
        av_decoder: An AVDecoder instance.
        num_frames: The number of frames to extract.
        video_out_frame_size: The size of the video frames to output, or None to use the original size.

    Returns:
        A tensor of shape (num_frames, channels, height, width) containing the extracted frames.
    """

    av_data = get_clips_uniform(
        av_decoder=av_decoder,
        clip_duration_seconds=0,
        num_clips=num_frames,
        request_video=True,
        request_audio=False,
        video_out_frame_size=video_out_frame_size,
    )

    if len(av_data.video_clips) == 0:
        raise ValueError("No video frames found")

    # Concatenate all video single-frame clips to form a single tensor
    video_tensor = torch.cat(av_data.video_clips, dim=0)
    if return_timestamps:
        return video_tensor, [t for t, _ in av_data.video_timestamps]
    else:
        return video_tensor
