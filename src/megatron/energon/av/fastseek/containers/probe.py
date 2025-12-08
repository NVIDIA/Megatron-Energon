# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from ...av_init import av_open
from ..keyframeinfo import KeyframeInfo


def parse_probe(file):
    keyframes = {}
    with av_open(file) as input_container:
        for stream_idx, stream in enumerate(input_container.streams.video):
            keyframes[stream.id] = [
                KeyframeInfo(index, p.pts)
                for index, p in enumerate(input_container.demux(video=stream_idx))
                if p.is_keyframe
            ]

        return keyframes
