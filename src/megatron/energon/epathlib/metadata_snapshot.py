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
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator, Literal, TypeVar

_Value = TypeVar("_Value", bool, int)
_MetadataKind = Literal["file_checks", "sizes", "jsonl_sizes"]
_MAX_CONTENT_BYTES = 64 * 1024 * 1024


def _check_digest(payload: bytes, sha256: str | None) -> None:
    if sha256 is not None and hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("Metadata snapshot SHA256 mismatch")


@dataclass(frozen=True)
class MetadataSnapshot:
    recipe_identity: str
    contents: dict[str, bytes]
    file_checks: dict[str, bool]
    sizes: dict[str, int]
    jsonl_sizes: dict[str, int]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        sha256: str | None = None,
        recipe_identity: str | None = None,
    ) -> "MetadataSnapshot":
        """Read one package; check its digest and recipe identity when supplied."""
        payload = Path(path).read_bytes()
        _check_digest(payload, sha256)
        document = json.loads(payload)
        if document["format_version"] != 1:
            raise ValueError("Unsupported metadata snapshot format")
        if recipe_identity is not None and document["recipe_identity"] != recipe_identity:
            raise ValueError("Metadata snapshot recipe identity mismatch")
        return cls(
            recipe_identity=document["recipe_identity"],
            contents={
                key: base64.b64decode(value, validate=True)
                for key, value in document["contents"].items()
            },
            file_checks=document["file_checks"],
            sizes=document["sizes"],
            jsonl_sizes=document["jsonl_sizes"],
        )

    def save(self, path: str | Path) -> None:
        """Atomically publish the package."""
        document = {
            "format_version": 1,
            "recipe_identity": self.recipe_identity,
            "contents": {
                key: base64.b64encode(value).decode("ascii") for key, value in self.contents.items()
            },
            "file_checks": self.file_checks,
            "sizes": self.sizes,
            "jsonl_sizes": self.jsonl_sizes,
        }
        payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        path = Path(path)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            temporary = Path(f.name)
            try:
                f.write(payload)
                f.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

    def lookup(self, kind: _MetadataKind, path: str, read: Callable[[], _Value]) -> _Value:
        """Serve a recorded scalar, or perform the ordinary storage lookup."""
        values = getattr(self, kind)
        return values[path] if path in values else read()

    def open(self, path: str, mode: str, opener: Callable[[str], IO] | None = None):
        """Return an independent stream; binary data and cursor state aren't shared."""
        if mode not in ("r", "rb") or path not in self.contents:
            return None
        stream = io.BytesIO(self.contents[path])
        return stream if mode == "rb" else io.TextIOWrapper(stream)


class _MetadataCapture:
    """Record through the same scoped I/O hooks used for replay."""

    def __init__(self) -> None:
        self.snapshot = MetadataSnapshot("", {}, {}, {}, {})
        self.contents = self.snapshot.contents
        self.file_checks = self.snapshot.file_checks
        self.sizes = self.snapshot.sizes
        self.jsonl_sizes = self.snapshot.jsonl_sizes
        self.captured_bytes = 0

    def lookup(self, kind: _MetadataKind, path: str, read: Callable[[], _Value]) -> _Value:
        values = getattr(self, kind)
        if path not in values:
            values[path] = read()
        return values[path]

    def open(self, path: str, mode: str, opener: Callable[[str], IO]):
        if mode not in ("r", "rb") or Path(path).suffix not in (".yaml", ".yml", ".json"):
            return None
        if path not in self.contents:
            remaining = _MAX_CONTENT_BYTES - self.captured_bytes
            with opener("rb") as stream:
                payload = stream.read(remaining + 1)
            if len(payload) > remaining:
                raise ValueError("Metadata capture exceeded 64 MiB; narrow the construction scope")
            self.contents[path] = payload
            self.captured_bytes += len(payload)
        return self.snapshot.open(path, mode)


_ACTIVE_SNAPSHOT: contextvars.ContextVar[MetadataSnapshot | _MetadataCapture | None] = (
    contextvars.ContextVar("energon_metadata_snapshot", default=None)
)


def active_snapshot() -> MetadataSnapshot | _MetadataCapture | None:
    return _ACTIVE_SNAPSHOT.get()


@contextlib.contextmanager
def use_metadata_snapshot(
    path: str | Path | MetadataSnapshot,
    *,
    create: bool = False,
    sha256: str | None = None,
) -> Iterator[MetadataSnapshot]:
    """Create or read a metadata package while constructing a dataset.

    With ``create=True``, record metadata from storage and write ``path`` only
    after successful construction. Otherwise, read the existing package at
    ``path``; a missing file is an error. An optional ``sha256`` checks only the
    package being read and is ignored when ``create=True``. Neither mode
    checks whether source metadata has changed. Exit before starting workers.

    Already loaded ``MetadataSnapshot`` objects remain supported for applications
    distributing snapshots in memory; omit ``create`` and ``sha256`` in that case.
    """
    if isinstance(path, MetadataSnapshot):
        if create or sha256 is not None:
            raise ValueError("create and sha256 require a snapshot file path")
        active = path
        snapshot = path
    elif create:
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
