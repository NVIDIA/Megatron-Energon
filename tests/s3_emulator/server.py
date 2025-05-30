# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Mapping

from .auth import S3Auth
from .handler import S3RequestHandler
from .state import S3State

__all__ = ["S3EmulatorServer"]


class S3EmulatorServer:
    """A lightweight, *blocking* S3 HTTP emulator.

    Typical usage::

        from s3_emulator import S3EmulatorServer

        server = S3EmulatorServer(
            host="127.0.0.1",
            port=9000,
            credentials={"ACCESS": "SECRET"},
        )
        server.serve_forever()

    By default the server blocks the calling thread. Wrap :py:meth:`serve_forever`
    inside :pyclass:`threading.Thread` when integrating in a larger test suite.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        *,
        credentials: Mapping[str, str] | None = None,
        root_dir: str | Path | None = None,
        region: str = "us-east-1",
    ):  # noqa: D401
        self._state = S3State(Path(root_dir) if root_dir else None)
        self._auth = S3Auth(credentials or {"test": "test"}, region=region)

        class _Server(ThreadingHTTPServer):
            state = self._state
            auth = self._auth

        self._httpd: ThreadingHTTPServer = _Server((host, port), S3RequestHandler)
        self._thread: threading.Thread | None = None
        print(f"S3 emulator listening on http://{host}:{self.port}")

    @property
    def port(self) -> int:
        return self._httpd.server_port

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> S3State:  # noqa: D401
        return self._state

    def serve_forever(self):  # noqa: D401
        try:
            self._httpd.serve_forever()
        finally:
            self._state.flush()

    def shutdown(self):  # noqa: D401
        self._httpd.shutdown()
        self._state.flush()

    # Convenience helpers -------------------------------------------------

    def start_background(self):  # noqa: D401
        """Start the server in a background :pyclass:`threading.Thread`."""
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Server already running")

        def _run():
            self.serve_forever()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None):  # noqa: D401
        """Join the background thread (see :py:meth:`start_background`)."""
        if self._thread is None:
            return
        self._thread.join(timeout)
