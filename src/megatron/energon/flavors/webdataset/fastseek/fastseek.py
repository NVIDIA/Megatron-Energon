# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from fractions import Fraction
from pathlib import Path

import av
import filetype
from bitstring.bits import BitsType
from sortedcontainers import SortedList

from .matroska import parse_matroska
from .mpeg import parse_atoms


class Fastseek:
    def __init__(self, file: Path | BitsType, probe: bool = False) -> None:
        if probe:
            with av.open(file) as input_container:
                keyframe_pts = [
                    p.pts for p in input_container.demux(video=0) if p.is_keyframe
                ]

                input_fps: Fraction = input_container.streams.video[0].average_rate
                input_tb: Fraction = input_container.streams.video[0].time_base
                frame_duration = int(1 / input_fps / input_tb)

                self.keyframes = SortedList(
                    [pts // frame_duration for pts in keyframe_pts]
                )

        if isinstance(file, Path):
            file = file.open("rb")

        ftype = filetype.guess(file)
        self.mime = ftype.mime

        if ftype.mime in ["video/mp4", "video/quicktime"]:
            for a in parse_atoms(file):
                if a.name == "stss":
                    self.keyframes: SortedList = SortedList(
                        [ss["number"] - 1 for ss in a.sync_sample_table]
                    )
                    break
        elif ftype.mime in ["video/x-matroska", "video/webm"]:
            self.keyframes, self.container_time_base = parse_matroska(file)
        else:
            raise ValueError(f"Unsupported file type: {ftype.mime}")

    def should_seek(self, current: int, target: int) -> bool:
        nearest_iframe_cts: int = self.nearest_keyframe(target)
        return current < nearest_iframe_cts <= target

    def nearest_keyframe(self, target: int) -> int:
        nearest_iframe_to_target_index: int = self.keyframes.bisect_left(target) - 1
        return self.keyframes[max(0, nearest_iframe_to_target_index)]
