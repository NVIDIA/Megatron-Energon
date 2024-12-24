# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Generator

from bitstring import ConstBitStream
from bitstring.bits import BitsType

box_atoms = {"moov", "trak", "mdia", "minf", "stbl", "edts"}  # Non-exhaustive


def parse_table(
    cbs: ConstBitStream, table_size: int, struct: dict[str, str]
) -> dict[str, Any]:
    return [
        dict(zip(struct.keys(), cbs.readlist(", ".join(struct.values()))))
        for _ in range(table_size)
    ]


class Atom:
    @staticmethod
    def make_atom(cbs: ConstBitStream) -> "Atom":
        size: int = cbs.read("uint:32")
        name: str = cbs.read("bytes:4").decode("ascii")
        box: bool = name in box_atoms

        subclass_list = [c for c in Atom.__subclasses__() if c.__name__ == name.upper()]
        atom_class: type = Atom
        if len(subclass_list) > 0:
            atom_class: type = subclass_list[0]

        atom = atom_class(size, name, box)
        atom._parse(cbs)

        return atom

    def __init__(self, size: int, name: str, box: bool) -> None:
        self.size: int = size
        self.name: str = name
        self.box: bool = box

    def _parse(self, cb: ConstBitStream) -> None:
        if self.box:
            return

        cb.bytepos += self.size - 8

    def __str__(self) -> str:
        return f"{self.name=}, {self.size=}, {self.box=}"


class STSS(Atom):
    def _parse(self, cbs: ConstBitStream) -> None:
        cbs.bytepos += 4  # Version (1 byte) and flags (3 bytes), can be ignored

        self.number_of_entries: int = cbs.read("uint:32")
        self.sync_sample_table: dict[str, Any] = parse_table(
            cbs, self.number_of_entries, {"number": "uint:32"}
        )


def parse_atoms(file: BitsType) -> Generator[Atom, None, None]:
    cbs = ConstBitStream(file)
    while cbs.pos < len(cbs):
        yield Atom.make_atom(cbs)
