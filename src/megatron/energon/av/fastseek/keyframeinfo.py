# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import dataclass


@dataclass
class KeyframeInfo:
    """
    Information about a detected keyframe

    The exact meaning of the information will vary per container, however in general

    index: the unit of representation for a frame (e.g., frame number)
    pts: a timestamp that can be used by a decoder to seek to exactly this frame

    """

    index: int
    pts: int

    def __eq__(self, other) -> bool:
        if isinstance(other, KeyframeInfo):
            return self.index == other.index

        return self.index == other

    def __lt__(self, other) -> bool:
        if isinstance(other, KeyframeInfo):
            return self.index < other.index

        return self.index < other
