# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import BinaryIO

from multistorageclient.file import PosixFile

from megatron.energon.epathlib.epath import EPath


def open_sized_index(index_path: EPath) -> tuple[BinaryIO, int]:
    """Open an offset index and return its stream and number of 8-byte entries."""
    stream = index_path.open("rb")
    try:
        # Object streams can expose a temporary fd. A composite profile can also
        # open a plain POSIX child while supplying its own authoritative size.
        if type(stream) is PosixFile and getattr(index_path.fs, "_metadata_provider", True) is None:
            size = os.fstat(stream.fileno()).st_size
        else:
            size = index_path.size()
        return stream, size // 8
    except BaseException:
        stream.close()
        raise
