# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit immutable snapshots for metadata-heavy loader construction.

Snapshots are opt-in: callers must select the expected content digest and recipe
identity. They describe a frozen dataset revision, not a live filesystem cache.
Rebuild and select a new digest whenever the recipe or dataset metadata changes.
The context must end before starting workers; sample/media I/O is unaffected.
"""

import base64
import contextlib
import contextvars
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class MetadataSnapshot:
    recipe_identity: str
    contents: dict[str, bytes]
    file_checks: dict[str, bool]
    sizes: dict[str, int]
    jsonl_sizes: dict[str, int]

    @classmethod
    def load(cls, path: str | Path, *, sha256: str, recipe_identity: str) -> "MetadataSnapshot":
        """Read one package, checking its digest and expected recipe identity."""
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError("Metadata snapshot SHA256 mismatch")
        document = json.loads(payload)
        if document["format_version"] != 1:
            raise ValueError("Unsupported metadata snapshot format")
        if document["recipe_identity"] != recipe_identity:
            raise ValueError("Metadata snapshot recipe identity mismatch")
        return cls(
            recipe_identity=recipe_identity,
            contents={
                key: base64.b64decode(value, validate=True)
                for key, value in document["contents"].items()
            },
            file_checks=document["file_checks"],
            sizes=document["sizes"],
            jsonl_sizes=document["jsonl_sizes"],
        )

    def open(self, path: str, mode: str):
        """Return an independent stream; binary data and cursor state aren't shared."""
        if mode not in ("r", "rb") or path not in self.contents:
            return None
        stream = io.BytesIO(self.contents[path])
        return stream if mode == "rb" else io.TextIOWrapper(stream)


_ACTIVE_SNAPSHOT: contextvars.ContextVar[MetadataSnapshot | None] = contextvars.ContextVar(
    "energon_metadata_snapshot", default=None
)


def active_snapshot() -> MetadataSnapshot | None:
    return _ACTIVE_SNAPSHOT.get()


@contextlib.contextmanager
def use_metadata_snapshot(snapshot: MetadataSnapshot) -> Iterator[None]:
    """Serve recorded construction metadata; unknown paths use their normal backend."""
    token = _ACTIVE_SNAPSHOT.set(snapshot)
    try:
        yield
    finally:
        _ACTIVE_SNAPSHOT.reset(token)
