from collections import defaultdict

from bitstring.bits import BitsType
from ebmlite import MasterElement, loadSchema
from sortedcontainers import SortedList

from ..keyframeinfo import KeyframeInfo


class CuePoint:
    def __init__(self, el: MasterElement) -> None:
        for c in el:
            if c.name == "CueTime":
                self.time = c.value
            if c.name == "CueTrackPositions":
                self.track_positions = CueTrackPositions(c)


class CueTrackPositions:
    def __init__(self, el: MasterElement) -> None:
        for c in el:
            if c.name == "CueTrack":
                self.track = c.value


def parse_matroska(file: BitsType) -> SortedList:
    schema = loadSchema("matroska.xml")
    doc = schema.load(file, headers=True)

    # Get cue times
    stack = [c for c in doc if c.name == "Segment"]
    cues = defaultdict(SortedList)
    while len(stack) > 0:
        el = stack.pop()
        if el.name == "CuePoint":
            cue = CuePoint(el)
            cues[cue.track_positions.track - 1].add(KeyframeInfo(cue.time, cue.time))
        elif isinstance(el, MasterElement):
            stack.extend([c for c in el if c.name in ["Cues", "CuePoint"]])

    return cues
