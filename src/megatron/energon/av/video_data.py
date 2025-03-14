# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, Optional, Union

import torch

from megatron.energon.dataclass_slots import dataclass_slots


@dataclass_slots
class VideoData:
    #: The input video tensor in the shape (frames, channel, h, w)
    frames: Optional[torch.Tensor]
    #: The input audio frames in the shape (number of channels, number of points)
    aframes: Optional[torch.Tensor]
    #: Metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    info: Dict[str, Union[bool, int, float, str]]
