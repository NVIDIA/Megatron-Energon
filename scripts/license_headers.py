# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import datetime
import re
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
        "Copyright (c) {year}, NVIDIA CORPORATION.",
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

    def has_header(self, file: Path) -> bool:
        with file.open() as rf:
            num_lines = 0
            for i, (line, expected) in enumerate(zip(rf, self._expected_lines)):
                has_other_year = False
                if i == 0:
                    search_pattern = (
                        "^" + re.escape(expected).replace("\\{year\\}", "[0-9]{4}") + "$"
                    )
                    has_other_year = re.search(search_pattern, line) is not None
                num_lines += 1
                if not has_other_year and line.rstrip("\n") != expected:
                    return False
            return num_lines == len(self._expected_lines)

    def _format_header(self, year: Optional[str] = None) -> Tuple[str, ...]:
        if year is None:
            year = str(datetime.datetime.now().year)
        return (self._expected_lines[0].format(year=year),) + self._expected_lines[1:]

    def fix_header(self, file: Path):
        contents = file.read_text()
        first_comment = self.line_comment if self.line_comment is not None else self.comment_start
        # Warning: Assumes that previous content is at the very start of the file.
        if contents.startswith(first_comment) and contents[len(first_comment) :].startswith(
            self.UPDATE_IDENTIFIER
        ):
            # Already has header, but want to update
            # Warning: Assumes that the previous header and update have the same number of lines.
            *header_lines, remainder = contents.split("\n", len(self._expected_lines))
            # Skip matches where only the year is different; we want to keep the original copyright
            # number.
            search_pattern = (
                "^" + re.escape(self._expected_lines[0]).replace("\\{year\\}", "[0-9]{4}") + "$"
            )
            if re.search(search_pattern, header_lines[0]) is not None:
                return
            new_contents = "\n".join(self._format_header()) + "\n" + remainder
        else:
            # No header, add it
            new_contents = "\n".join(self._format_header()) + "\n" + contents
        file.write_text(new_contents)


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
