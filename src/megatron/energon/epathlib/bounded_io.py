# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Bound POSIX read-ahead for indexed sample access."""

from typing import BinaryIO

from megatron.energon.epathlib import EPath


def open_sample_file(path: EPath) -> BinaryIO:
    """Open an indexed sample stream with a 64 KiB POSIX read-ahead budget.

    Small adjacent reads share this buffer. Large payload reads still use the
    standard buffered reader's direct-read path. Preserve the storage client's
    behavior for non-POSIX backends; never share mutable file positions.
    """
    return path.open("rb", buffering=64 * 1024)
