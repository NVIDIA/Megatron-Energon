# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause


import hashlib
from pathlib import Path

from filelock import FileLock

from megatron.energon.epathlib import EPath

LOCAL_COPY_TMP_DIR = Path("/tmp/energon_local_copy")


def ensure_local_copy(path: EPath) -> EPath:
    """If the path is not local, copy it to a temporary directory and return the
    path to the temporary directory.

    Args:
        path: The path to the file to copy.

    Returns:
        The path to the local copy of the file or the original path if it is already local.
    """

    assert path.is_file(), f"Path {path} is not a file"

    if path.is_local():
        return path

    LOCAL_COPY_TMP_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(path).encode()).hexdigest()
    final_path = LOCAL_COPY_TMP_DIR / f"{digest}.bin"
    lock_path = final_path.with_suffix(".lock")
    tmp_path = final_path.with_suffix(".part")

    # Block until lock is free
    with FileLock(lock_path, timeout=60 * 5):
        if final_path.exists():  # someone else already produced it
            return EPath(final_path)

        # We are the downloader
        try:
            path.copy(EPath(tmp_path))
            tmp_path.rename(final_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    return EPath(final_path)
