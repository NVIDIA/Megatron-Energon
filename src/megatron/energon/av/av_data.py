# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import torch

from megatron.energon.edataclass import edataclass


@edataclass
class AVData:
    #: A list of video clips in the shape (frames, channels, h, w)
    video_clips: Optional[list[torch.Tensor]]
    #: The timestamps for the video clips. List of tuples (start, end) in seconds
    video_timestamps: Optional[list[tuple[float, float]]]
    #: A list of audio clips in the shape (channels, samples)
    audio_clips: Optional[list[torch.Tensor]]
    #: The timestamps for the audio clips. List of tuples (start, end) in seconds
    audio_timestamps: Optional[list[tuple[float, float]]]
