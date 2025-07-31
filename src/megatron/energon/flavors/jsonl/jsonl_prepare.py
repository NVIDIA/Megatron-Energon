# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    TypeVar,
    Union,
)

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.jsonl.ijsonl import IJsonlFile, IJsonlIndexWriter

logger = logging.getLogger(__name__)

T = TypeVar("T", covariant=True)


class JsonlPreparator:
    @staticmethod
    def iter_dataset_content(
        path: Union[str, EPath],
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yield example dataset content for a few samples.

        Args:
            path: Path to the tar file.
        """
        with EPath(path).open("rb") as f:
            with IJsonlFile(f) as index_reader:
                for entry in index_reader:
                    yield {"json": entry}

    @classmethod
    def prepare_dataset(
        cls,
        path: Union[Path, EPath],
    ) -> int:
        """
        Preprocess the jsonl file. Preprocessing is done in parallel.
        Counts the number of samples.

        Args:
            path: Path to the jsonl file

        Returns:
            Count of samples in the jsonl file.
        """
        count = 0
        # Processing is lagging behind. The offsets include empty lines. The whole file must be covered!
        last_offset = 0
        with IJsonlIndexWriter(EPath(path)) as iw:
            with EPath(path).open("rb") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        if last_offset:
                            last_offset = f.tell()
                        continue
                    assert line.startswith(b"{") and line.endswith(b"}"), (
                        f"Line {line} does not start and end with a json object {{}}."
                    )
                    iw.append(last_offset)
                    last_offset = f.tell()
                    count += 1
                assert last_offset == f.tell(), (
                    f"The last offset {last_offset} does not match the file size {f.tell()}."
                )
                assert last_offset != 0, "File is empty."
                iw.append(last_offset)
        return count
