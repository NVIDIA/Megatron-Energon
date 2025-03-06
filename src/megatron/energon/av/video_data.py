# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Dict, Union

import torch


@dataclasses.dataclass
class VideoData:
    #: The input video tensor in the shape (frames, channel, h, w)
    frames: torch.Tensor
    #: The input audio frames in the shape (number of channels, number of points)
    aframes: torch.Tensor
    #: Metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    info: Dict[str, Union[bool, int, float, str]]
