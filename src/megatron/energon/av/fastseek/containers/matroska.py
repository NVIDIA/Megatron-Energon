# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from collections import defaultdict

from bitstring.bits import BitsType
from ebmlite import MasterElement, loadSchema
from sortedcontainers import SortedList

from ..keyframeinfo import KeyframeInfo


class CueTrackPositions:
    track: int

    def __init__(self, el: MasterElement) -> None:
        for c in el:
            if c.name == "CueTrack":
                self.track = c.value


class CuePoint:
    time: int
    track_positions: CueTrackPositions

    def __init__(self, el: MasterElement) -> None:
        for c in el:
            if c.name == "CueTime":
                self.time = c.value
            if c.name == "CueTrackPositions":
                self.track_positions = CueTrackPositions(c)


def parse_matroska(file: BitsType) -> SortedList:
    try:
        schema = loadSchema("matroska.xml")
        doc = schema.load(file, headers=True)
    except (KeyError, IOError, TypeError) as e:
        raise ValueError(f"Matroska parsing failed with error {e}")

    # Get cue times
    stack = [c for c in doc if c.name == "Segment"]
    cues = defaultdict(SortedList)
    while len(stack) > 0:
        el = stack.pop()
        if el.name == "CuePoint":
            cue = CuePoint(el)
            cues[cue.track_positions.track].add(KeyframeInfo(cue.time, cue.time))
        elif isinstance(el, MasterElement):
            stack.extend([c for c in el if c.name in ["Cues", "CuePoint"]])

    return cues
