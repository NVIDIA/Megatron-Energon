# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import av
from sortedcontainers import SortedList

from ..keyframeinfo import KeyframeInfo


def parse_probe(file):
    keyframes = {}
    with av.open(file) as input_container:
        for stream_idx, stream in enumerate(input_container.streams.video):
            packet_pts = [
                (index, p.pts)
                for index, p in enumerate(input_container.demux(video=stream_idx))
                if p.is_keyframe
            ]
            packet_pts.sort(key=lambda x: x[1])

            keyframes[stream.id] = SortedList([KeyframeInfo(*p) for p in packet_pts])

        return keyframes
