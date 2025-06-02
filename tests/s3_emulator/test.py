# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from contextlib import contextmanager
from typing import Generator

from .server import S3EmulatorServer


@contextmanager
def s3_emulator(
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    credentials: dict[str, str] | None = None,
    root_dir: str | None = None,
    region: str = "us-east-1",
) -> Generator[S3EmulatorServer, None, None]:
    """Context manager for running an S3 emulator server in the background.

    Args:
        host: Host to bind the server to
        port: Port to bind the server to. Use 0 to let the OS choose a free port.
        credentials: Optional credentials mapping
        root_dir: Optional directory to persist S3 data
        region: Region for authentication

    Yields:
        The running S3 emulator server instance
    """
    server = S3EmulatorServer(
        host=host,
        port=port,
        credentials=credentials,
        root_dir=root_dir,
        region=region,
    )
    try:
        server.start_background()
        server.wait_until_ready()
        yield server
    finally:
        server.shutdown()
        server.join()
