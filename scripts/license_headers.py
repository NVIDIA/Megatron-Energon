# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import click


@dataclass
class HeaderUpdater:
    file_ext: str
    line_comment: Optional[str] = None
    comment_start: Optional[str] = None
    comment_end: Optional[str] = None

    UPDATE_IDENTIFIER = "Copyright"

    HEADER_LINES: Tuple[str, ...] = (
        "Copyright (c) 2025, NVIDIA CORPORATION.",
        "SPDX-License-Identifier: BSD-3-Clause",
    )

    _expected_lines: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.line_comment is not None:
            self._expected_lines = tuple(self.line_comment + line for line in self.HEADER_LINES)
        else:
            assert self.comment_start is not None and self.comment_end is not None
            if len(self.HEADER_LINES) >= 2:
                self._expected_lines = (
                    self.comment_start + self.HEADER_LINES[0],
                    *self.HEADER_LINES[1:-1],
                    self.HEADER_LINES[-1] + self.comment_end,
                )
            else:
                assert len(self.HEADER_LINES) == 1
                self._expected_lines = (
                    self.comment_start + self.HEADER_LINES[0] + self.comment_end,
                )

    @staticmethod
    def split_front_matter(contents: str) -> Tuple[str, str]:
        """Splits a leading YAML front matter block (e.g. of a skill's SKILL.md), which must stay
        the first thing in the file, from the rest. The block is empty if there is none."""
        if contents.startswith("---\n"):
            end = contents.find("\n---\n", 3)
            if end != -1:
                end += len("\n---\n")
                rest = contents[end:].lstrip("\n")
                return contents[: len(contents) - len(rest)], rest
        return "", contents

    def has_header(self, file: Path) -> bool:
        _, contents = self.split_front_matter(file.read_text())
        lines = contents.split("\n", len(self._expected_lines))
        return tuple(lines[: len(self._expected_lines)]) == self._expected_lines

    def fix_header(self, file: Path):
        front_matter, contents = self.split_front_matter(file.read_text())
        first_comment = self.line_comment if self.line_comment is not None else self.comment_start
        if contents.startswith(first_comment) and contents[len(first_comment) :].startswith(
            self.UPDATE_IDENTIFIER
        ):
            # Already has header, but want to update
            *header_lines, remainder = contents.split("\n", len(self._expected_lines))
            new_contents = "\n".join(self._expected_lines) + "\n" + remainder
        else:
            # No header, add it
            new_contents = "\n".join(self._expected_lines) + "\n" + contents
        file.write_text(front_matter + new_contents)


headers = (
    HeaderUpdater(
        file_ext=".py",
        line_comment="# ",
    ),
    HeaderUpdater(
        file_ext=".sh",
        line_comment="# ",
    ),
    # Do not add to yamls
    # HeaderUpdater(
    #     file_ext=".yml",
    #     line_comment="# ",
    # ),
    # HeaderUpdater(
    #     file_ext=".yaml",
    #     line_comment="# ",
    # ),
    HeaderUpdater(
        file_ext=".md",
        comment_start="<!--- ",
        comment_end=" -->",
    ),
)


@click.command()
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
)
@click.option(
    "--fix", is_flag=True, help="Automatically add missing/adapt existing license headers."
)
def main(path: Path, fix: bool) -> None:
    # Get all files to check
    success = True
    for header_update in headers:
        for filepath in path.rglob(f"*{header_update.file_ext}"):
            if not header_update.has_header(filepath):
                success = False
                print(f"Missing header in {filepath}")
                if fix:
                    header_update.fix_header(filepath)
                    print(f"Added header to {filepath}")

    if success:
        print("All files have the correct license header.")
    elif fix:
        print("Some license headers were fixed.")
    else:
        print(
            "Some license headers are missing or incorrect. Run `python scripts/license_headers.py . --fix` to fix."
        )
        exit(1)


if __name__ == "__main__":
    main()
