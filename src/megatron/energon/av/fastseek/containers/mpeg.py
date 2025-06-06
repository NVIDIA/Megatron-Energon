# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from collections import defaultdict
from itertools import accumulate
from typing import Any, Generator

from bitstring import ConstBitStream, Error
from bitstring.bits import BitsType
from sortedcontainers import SortedList

from ..keyframeinfo import KeyframeInfo

box_atoms = {"moov", "trak", "mdia", "minf", "stbl", "edts"}  # Non-exhaustive


def parse_table(cbs: ConstBitStream, table_size: int, struct: dict[str, str]) -> dict[str, Any]:
    return [
        dict(zip(struct.keys(), cbs.readlist(", ".join(struct.values()))))
        for _ in range(table_size)
    ]


class Atom:
    skip_version_and_flags: bool = False

    @staticmethod
    def make_atom(cbs: ConstBitStream) -> "Atom":
        size: int = cbs.read("uint:32")
        name: str = cbs.read("bytes:4").decode("ascii")
        box: bool = name in box_atoms

        if size == 0:
            raise RuntimeError(
                "MPEG parser detected a zero byte atom, this likely indicates a corrupt video."
            )

        subclass_list = [c for c in Atom.__subclasses__() if c.__name__ == name.upper()]
        atom_class: type = Atom
        if len(subclass_list) > 0:
            atom_class: type = subclass_list[0]
            cbs.bytepos += 4  # Skip version and flags TODO not every atom needs this

        atom = atom_class(size, name, box)
        atom._parse(cbs)

        return atom

    def __init__(self, size: int, name: str, box: bool) -> None:
        self.size: int = size
        self.name: str = name
        self.box: bool = box

    def _parse(self, cbs: ConstBitStream) -> None:
        if not self.box:
            cbs.bytepos += self.size - 8

    def __str__(self) -> str:
        return f"{self.name=}, {self.size=}, {self.box=}"


class TKHD(Atom):
    """
    Parses the track header atom, see https://developer.apple.com/documentation/quicktime-file-format/track_header_atom
    """

    def _parse(self, cbs: ConstBitStream) -> None:
        cbs.bytepos += 8  # skip creation time and modification time
        self.track_id: int = cbs.read("uint:32")
        cbs.bytepos += 68  # Skip rest of structure


class HDLR(Atom):
    """
    Parses the media handler atom, see https://developer.apple.com/documentation/quicktime-file-format/handler_reference_atom

    NOTE: currently unused but could speed up parsing by skipping audio tracks
    """

    def _parse(self, cbs: ConstBitStream) -> None:
        self.component_type = cbs.read("bytes:4").decode("ascii")
        self.component_subtype = cbs.read("bytes:4").decode("ascii")

        # Skip rest of structure, the last field is variable so we need to use the total size
        # 24 bytes already read (size (4), type (4), version (1), flags (3), component type (4), component subtype (4))
        cbs.bytepos += self.size - 20


class STSS(Atom):
    """
    Parses the sync sample atom https://developer.apple.com/documentation/quicktime-file-format/sample_table_atom/sync_sample_atom
    """

    def _parse(self, cbs: ConstBitStream) -> None:
        self.number_of_entries: int = cbs.read("uint:32")
        self.sync_sample_table: dict[str, Any] = parse_table(
            cbs, self.number_of_entries, {"number": "uint:32"}
        )


class STTS(Atom):
    """
    Parses the time to sample atom https://developer.apple.com/documentation/quicktime-file-format/time-to-sample_atom
    """

    def _parse(self, cbs: ConstBitStream) -> None:
        self.number_of_entries: int = cbs.read("uint:32")
        self.time_to_sample_table: dict[str, Any] = parse_table(
            cbs,
            self.number_of_entries,
            {"sample_count": "uint:32", "sample_duration": "uint:32"},
        )


class CTTS(Atom):
    """
    Parses the composition offset atom https://developer.apple.com/documentation/quicktime-file-format/composition_offset_atom
    """

    def _parse(self, cbs: ConstBitStream) -> None:
        self.number_of_entries: int = cbs.read("uint:32")
        self.composition_offset_table: dict[str, Any] = parse_table(
            cbs,
            self.number_of_entries,
            {
                "sample_count": "uint:32",
                "composition_offset": "int:32",
                "media_rate": "",
            },
        )


class ELST(Atom):
    """
    Parses the edit list atom https://developer.apple.com/documentation/quicktime-file-format/edit_list_atom
    """

    def _parse(self, cbs: ConstBitStream) -> None:
        self.number_of_entries: int = cbs.read("uint:32")
        self.edit_list_table: dict[str, Any] = parse_table(
            cbs,
            self.number_of_entries,
            {
                "track_duration": "uint:32",
                "media_time": "int:32",
                "media_rate": "int:32",
            },
        )


class MDAT(Atom):
    """
    Parses the media data atom https: https://developer.apple.com/documentation/quicktime-file-format/movie_data_atom

    This is only here to handle the unusual size handling of mdat, if the normal size field is set to 1
    then the actual size is stored as a 64 bit integer
    """

    def _parse(self, cbs: ConstBitStream) -> None:
        if self.size == 1:
            cbs.bytepos -= 4  # No version or flags for mdat
            self.size = cbs.read("uint:64")
            seekto = self.size - 16
        else:
            seekto = self.size - 12

        if cbs.bytepos + seekto >= (cbs.len / 8):
            raise StopIteration()

        cbs.bytepos += seekto


def parse_atoms(file: BitsType) -> Generator[Atom, None, None]:
    try:
        cbs = ConstBitStream(file)
        while cbs.pos < len(cbs):
            try:
                yield Atom.make_atom(cbs)
            except StopIteration:
                return
    except Error as e:
        raise ValueError(f"MPEG parsing failed with error {e}")


def parse_mpeg(file: BitsType) -> dict[int, SortedList]:
    sync_samples = {}
    decode_timestamps = {}
    presentation_time_offsets = defaultdict(lambda: defaultdict(lambda: 0))
    start_offsets = defaultdict(lambda: 0)
    current_track = -1
    for a in parse_atoms(file):
        if a.name == "tkhd":
            a: TKHD
            current_track = a.track_id
        elif a.name == "stts":
            a: STTS
            decode_timestamps[current_track] = list(
                accumulate(
                    sum(
                        [
                            [entry["sample_duration"]] * entry["sample_count"]
                            for entry in a.time_to_sample_table
                        ],
                        [0],
                    )
                )
            )
        elif a.name == "ctts":
            a: CTTS
            presentation_time_offsets[current_track] = sum(
                [
                    [entry["composition_offset"]] * entry["sample_count"]
                    for entry in a.composition_offset_table
                ],
                [],
            )
        elif a.name == "stss":
            a: STSS
            sync_samples[current_track] = [ss["number"] - 1 for ss in a.sync_sample_table]
        elif a.name == "elst":
            # NOTE the "media_time" here is a "delay" between decoding and presenting the first sample.
            # We follow the ffmpeg convention that the first frame displays at time 0 which means we should
            # *subtract* this offset from the decoding time values rather than adding it to presentation time values
            # TODO there can be more than one of these, figure out how to handle it
            a: ELST
            start_offsets[current_track] = -a.edit_list_table[0]["media_time"]
    keyframes = defaultdict(lambda: SortedList())
    try:
        for track_id in sync_samples.keys():
            for keyframe_number in sync_samples[track_id]:
                pts = (
                    decode_timestamps[track_id][keyframe_number]
                    + start_offsets[track_id]
                    + presentation_time_offsets[track_id][keyframe_number]
                )
                keyframes[track_id].add(KeyframeInfo(keyframe_number, pts))
    except (KeyError, IndexError) as e:
        raise ValueError(f"MPEG parsing failed with error {e}")

    return keyframes
