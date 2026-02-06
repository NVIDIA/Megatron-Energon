# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from bitstring.bits import BitsType

from ...av_init import av_open
from ..keyframeinfo import KeyframeInfo


def parse_probe(file: BitsType) -> dict[int, list[KeyframeInfo]]:
    """
    Parse the container file using pyav to find keyframes.

    Args:
        file: The container file to parse.

    Returns:
        A dictionary of keyframes, keyed by stream id. dict<stream_id: int, list<KeyframeInfo>>
    """
    keyframes = {}
    with av_open(file) as input_container:
        for stream_idx, stream in enumerate(input_container.streams.video):
            keyframes[stream.id] = [
                KeyframeInfo(index, p.pts)
                for index, p in enumerate(input_container.demux(video=stream_idx))
                if p.is_keyframe
            ]

    return keyframes
