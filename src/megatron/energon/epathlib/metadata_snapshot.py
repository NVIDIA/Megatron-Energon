# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Create or read metadata snapshots around construction of a frozen dataset.

The context must end before starting workers; sample/media I/O is unaffected.
Recreate the snapshot whenever the recipe or dataset metadata changes.
"""

import base64
import contextlib
import contextvars
import hashlib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

logger = logging.getLogger(__name__)

_MAX_CONTENT_BYTES = 64 * 1024 * 1024


def _check_digest(payload: bytes, sha256: str | None) -> None:
    if sha256 is not None and hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("Metadata snapshot SHA256 mismatch")


@dataclass(frozen=True)
class MetadataSnapshot:
    contents: dict[str, bytes]
    file_checks: dict[str, bool]
    sizes: dict[str, int]
    jsonl_sizes: dict[str, int]

    @classmethod
    def load(cls, path: str | Path, *, sha256: str | None = None) -> "MetadataSnapshot":
        """Read one package and check its digest when supplied."""
        payload = Path(path).read_bytes()
        _check_digest(payload, sha256)
        document = json.loads(payload)
        return cls(
            contents={
                key: base64.b64decode(value, validate=True)
                for key, value in document["contents"].items()
            },
            file_checks=document["file_checks"],
            sizes=document["sizes"],
            jsonl_sizes=document["jsonl_sizes"],
        )

    def save(self, path: str | Path) -> None:
        """Atomically publish the package and log its path and SHA256 at INFO level."""
        document = {
            "contents": {
                key: base64.b64encode(value).decode("ascii") for key, value in self.contents.items()
            },
            "file_checks": self.file_checks,
            "sizes": self.sizes,
            "jsonl_sizes": self.jsonl_sizes,
        }
        payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_bytes(payload)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        logger.info(
            "Created metadata snapshot %s; sha256=%s", path, hashlib.sha256(payload).hexdigest()
        )

    def open(self, path: str, mode: str):
        """Return an independent stream; binary data and cursor state aren't shared."""
        if mode not in ("r", "rb") or path not in self.contents:
            return None
        stream = io.BytesIO(self.contents[path])
        return stream if mode == "rb" else io.TextIOWrapper(stream)


class _MetadataCapture:
    """Collect construction metadata while ordinary storage operations run."""

    def __init__(self) -> None:
        self.snapshot = MetadataSnapshot({}, {}, {}, {})
        self.captured_bytes = 0

    def record_contents(self, path: str, stream: BinaryIO) -> None:
        """Read an opened metadata file, bounding the total stored contents."""
        previous_size = len(self.snapshot.contents.get(path, b""))
        remaining = _MAX_CONTENT_BYTES - self.captured_bytes + previous_size
        payload = stream.read(remaining + 1)
        if len(payload) > remaining:
            raise ValueError("Metadata capture exceeded 64 MiB; narrow the construction scope")
        self.snapshot.contents[path] = payload
        self.captured_bytes += len(payload) - previous_size


_ACTIVE_SNAPSHOT: contextvars.ContextVar[MetadataSnapshot | _MetadataCapture | None] = (
    contextvars.ContextVar("energon_metadata_snapshot", default=None)
)


def active_snapshot() -> MetadataSnapshot | None:
    """Return the snapshot only when reading mode is active."""
    active = _ACTIVE_SNAPSHOT.get()
    return active if isinstance(active, MetadataSnapshot) else None


def active_capture() -> _MetadataCapture | None:
    """Return the recorder only when creation mode is active."""
    active = _ACTIVE_SNAPSHOT.get()
    return active if isinstance(active, _MetadataCapture) else None


@contextlib.contextmanager
def use_metadata_snapshot(
    path: str | Path,
    *,
    create: bool = False,
    sha256: str | None = None,
) -> Iterator[MetadataSnapshot]:
    """Create or read a metadata package at a filesystem path during construction.

    With ``create=True``, perform ordinary storage reads and record their results,
    including repeated queries. Write ``path`` only after successful construction,
    then log its path and SHA256 at INFO level.
    Otherwise, read the existing package at ``path``; a missing file is an error.
    An optional ``sha256`` checks only the package being read and is ignored
    when ``create=True``. Neither mode checks whether source metadata has changed.
    Only one process may create a given snapshot at a time; the caller must
    coordinate writers and wait for publication before starting readers.
    Exit before starting workers.
    """
    path = Path(path)
    if create:
        active = _MetadataCapture()
        snapshot = active.snapshot
    else:
        active = snapshot = MetadataSnapshot.load(path, sha256=sha256)
    token = _ACTIVE_SNAPSHOT.set(active)
    try:
        yield snapshot
        if create:
            snapshot.save(path)
    finally:
        _ACTIVE_SNAPSHOT.reset(token)
