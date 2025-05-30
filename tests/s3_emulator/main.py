# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path

import click

from .server import S3EmulatorServer


@click.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind the server to",
)
@click.option(
    "--port",
    default=9000,
    type=int,
    help="Port to bind the server to",
)
@click.option(
    "--root-dir",
    type=click.Path(path_type=Path),
    help="Directory to persist S3 data",
)
@click.option(
    "--access-key",
    default="test",
    help="Access key for authentication",
)
@click.option(
    "--secret-key",
    default="test",
    help="Secret key for authentication",
)
@click.option(
    "--region",
    default="us-east-1",
    help="Region for authentication",
)
def main(
    host: str, port: int, root_dir: Path | None, access_key: str, secret_key: str, region: str
) -> None:
    """Start an S3 emulator server."""
    server = S3EmulatorServer(
        host=host,
        port=port,
        credentials={access_key: secret_key},
        root_dir=root_dir,
        region=region,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
