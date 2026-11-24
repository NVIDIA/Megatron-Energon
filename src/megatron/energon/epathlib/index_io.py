# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import os


def open_sized_index(index_path):
    """Avoid full path metadata only for a plain MSC POSIX stream.

    ObjectFile.fileno() may describe a temporary/dummy file, so generic fileno
    detection is unsafe. Metadata-provider and in-memory snapshot semantics
    retain the original size query. No payload data is read to obtain the size.
    """
    from multistorageclient.file import PosixFile

    stream = index_path.open("rb")
    try:
        client = getattr(stream, "_storage_client", None)
        if type(stream) is PosixFile and getattr(client, "_metadata_provider", True) is None:
            size = os.fstat(stream.fileno()).st_size
        else:
            size = index_path.size()
        return stream, size // 8
    except BaseException:
        stream.close()
        raise
