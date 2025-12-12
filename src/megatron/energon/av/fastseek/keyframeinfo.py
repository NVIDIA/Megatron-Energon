# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import dataclass


@dataclass(slots=True)
class KeyframeInfo:
    """
    Information about a detected keyframe

    The exact meaning of the information will vary per container, however in general

    index: the unit of representation for a frame (e.g., frame number)
    pts: a timestamp that can be used by a decoder to seek to exactly this frame

    """

    #: The index of the keyframe. If None, the keyframe is not indexed by frame number.
    index: int | None
    #: The PTS of the keyframe. If None, the keyframe is not indexed by PTS.
    pts: int | None
