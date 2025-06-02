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
    """A lightweight, blocking S3 HTTP emulator.

    This server provides a minimal S3-compatible HTTP interface for testing purposes.
    It supports basic S3 operations like bucket and object management.

    Example (blocking)::
        server = S3EmulatorServer(
            host="127.0.0.1",
            port=9000,
            credentials={"ACCESS": "SECRET"},
        )
        server.serve_forever()

    Example (threaded)::
        server = S3EmulatorServer(
            host="127.0.0.1",
            port=9000,
            credentials={"ACCESS": "SECRET"},
        )
        server.start_background()
        # ...
        server.shutdown()
        server.join()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 0,
        *,
        credentials: Mapping[str, str] | None = None,
        root_dir: str | Path | None = None,
        region: str = "us-east-1",
    ):
        """
        This server provides a minimal S3-compatible HTTP interface for testing purposes.
        It supports basic S3 operations like bucket and object management.

        There is no need to check that the port is bound, it is already bound after initialization.
        Retrieve the real port with `.port` if set to 0.
        The server is listening to the port immediately, but will only start processing after
        `start_background()` (threaded) or `.serve_forever()` (blocking) is called.

        Args:
            host: The host address to bind to.
            port: The port to bind to. Use 0 to let the OS choose a free port.
            credentials: Optional mapping of access keys to secret keys.
            root_dir: Optional path to persist the S3 store on disk.
            region: AWS region to emulate.
        """
        self._state = S3State(Path(root_dir) if root_dir else None)
        self._auth = S3Auth(credentials or {"test": "test"}, region=region)

        class _Server(ThreadingHTTPServer):
            state = self._state
            auth = self._auth

        self._httpd: ThreadingHTTPServer = _Server((host, port), S3RequestHandler)
        self._thread: threading.Thread | None = None
        print(f"S3 emulator on http://{host}:{self.port}", flush=True)

    @property
    def port(self) -> int:
        """Returns the port number the server is bound to."""
        return self._httpd.server_port

    @property
    def state(self) -> S3State:
        """Returns the internal S3 state object."""
        return self._state

    def serve_forever(self):
        """Start the server and block until shutdown is called.

        This method will block the calling thread. For non-blocking usage,
        see start_background().
        """
        try:
            self._httpd.serve_forever()
        finally:
            self._state.flush()

    def shutdown(self):
        """Shutdown the server and flush any pending state changes."""
        self._httpd.shutdown()
        self._state.flush()

    def start_background(self):
        """Start the server in a background thread."""
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Server already running")

        def _run():
            self.serve_forever()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None):
        """Join the background thread.

        Args:
            timeout: Optional timeout in seconds to wait for thread completion.
        """
        if self._thread is None:
            return
        self._thread.join(timeout)
