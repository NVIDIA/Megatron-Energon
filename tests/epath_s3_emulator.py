# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from multistorageclient.rclone import read_rclone_config

from tests.s3_emulator.state import S3State
from tests.s3_emulator.test import s3_emulator


@contextmanager
def setup_s3_emulator(
    *,
    port: int = 0,
    access_key: str = "test",
    secret_key: str = "test",
    root_dir: str | None = None,
    region: str = "us-east-1",
    profile_name: str = "s3test",
) -> Generator[S3State, None, None]:
    """Set up S3 emulator and write necessary config files.

    Args:
        port: Port to bind the server to. Use 0 to let the OS choose a free port.
        access_key: Access key for authentication
        secret_key: Secret key for authentication
        root_dir: Optional directory to persist S3 data
        region: Region for authentication
        profile_name: Name of the rclone profile. Must be different in all tests, to ensure that a
            cached rclone config is used in MSC.

    Returns:
        The S3 emulator state. Can be used to quickly upload files to the emulator.
    """

    try:
        with s3_emulator(
            host="127.0.0.1",
            port=port,
            credentials={access_key: secret_key},
            root_dir=root_dir,
            region=region,
        ) as emu:
            # Create config directory
            config_dir = Path("/tmp/XDG_CONFIG_HOME/.config/rclone")
            config_dir.mkdir(parents=True, exist_ok=True)

            # Write rclone config
            config_path = config_dir / "rclone.conf"
            with config_path.open("w") as f:
                f.write(
                    "\n".join(
                        [
                            f"[{profile_name}]",
                            "type = s3",
                            "env_auth = false",
                            f"access_key_id = {access_key}",
                            f"secret_access_key = {secret_key}",
                            f"region = {region}",
                            f"endpoint = http://127.0.0.1:{emu.port}",
                        ]
                    )
                )

            # Set environment variables
            os.environ["XDG_CONFIG_HOME"] = "/tmp/XDG_CONFIG_HOME/.config"
            os.environ["HOME"] = "/tmp/XDG_CONFIG_HOME"

            # Hack to clear the cache of the rclone config for msc to get the "s3" profile
            read_rclone_config.cache_clear()

            yield emu.state

            read_rclone_config.cache_clear()
    except Exception as e:
        print("ERROR in s3_emulator", flush=True)
        print("Full traceback:", flush=True)
        import traceback

        traceback.print_exc()
        raise e
