# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from bitstring.bits import BitsType

try:
    # Try importing optional dependencies
    import av
    import av.container

except ImportError:
    pass


def av_open(file: BitsType) -> "av.container.InputContainer":
    """Open a file with PyAV.

    This function is a wrapper around av.open that disables additional threads in the container.
    """

    input_container = av.open(file, "r")
    try:
        initialize_av_container(input_container)
    except Exception:
        input_container.close()
        raise
    return input_container


def initialize_av_container(input_container: "av.container.InputContainer") -> None:
    """Every PyAV container should be initialized with this function.

    This function ensures that no additional threads are created.
    This is to avoid deadlocks in ffmpeg when deallocating the container.
    Furthermore, we cannot have multiple threads before forking the process when
    using torch data loaders with multiple workers.
    """

    for stream in input_container.streams:
        cc = stream.codec_context

        if cc is not None:
            cc.thread_type = "NONE"
            cc.thread_count = 0
