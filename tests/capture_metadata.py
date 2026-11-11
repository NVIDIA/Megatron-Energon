# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import base64
import contextlib
import hashlib
import io
import json
import os
import time
from pathlib import Path
from unittest.mock import patch


@contextlib.contextmanager
def capture_metadata(output: Path, recipe_identity: str):
    # Import only when the offline capture is explicitly requested.
    from megatron.energon.epathlib import EPath
    from megatron.energon.flavors.jsonl.ijsonl import IJsonlIndexReader

    contents, checks, sizes, jsonl_sizes, audit = {}, {}, {}, {}, {}
    original_open, original_check, original_size = EPath.open, EPath.is_file, EPath.size
    original_jsonl_size = IJsonlIndexReader.size
    byte_budget = 64 * 1024 * 1024
    captured_bytes = 0
    calls = {"open": 0, "is_file": 0, "size": 0, "jsonl_size": 0}
    started = time.perf_counter()

    def record_path(path):
        key = str(path)
        if key in audit or not path.is_local():
            return
        local = path.local_path()
        try:
            stat = local.stat()
            audit[key] = {
                "path": str(local),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        except FileNotFoundError:
            audit[key] = {"path": str(local), "missing": True}

    def opened(path, mode="r", block_size=None, prefetch_file=False):
        nonlocal captured_bytes
        calls["open"] += 1
        key = str(path)
        if mode in ("r", "rb") and Path(key).suffix in (".yaml", ".yml", ".json"):
            if key not in contents:
                with original_open(
                    path,
                    "rb",
                    block_size=block_size,
                    prefetch_file=prefetch_file,
                ) as stream:
                    payload = stream.read(byte_budget - captured_bytes + 1)
                captured_bytes += len(payload)
                if captured_bytes > byte_budget:
                    raise ValueError("Metadata capture exceeded 64 MiB; narrow the capture scope")
                contents[key] = payload
                record_path(path)
            stream = io.BytesIO(contents[key])
            return stream if mode == "rb" else io.TextIOWrapper(stream)
        return original_open(
            path,
            mode,
            block_size=block_size,
            prefetch_file=prefetch_file,
        )

    def is_file(path):
        calls["is_file"] += 1
        result = original_check(path)
        checks[str(path)] = result
        record_path(path)
        return result

    def size(path):
        calls["size"] += 1
        result = original_size(path)
        sizes[str(path)] = result
        record_path(path)
        return result

    def jsonl_size(path):
        calls["jsonl_size"] += 1
        result = original_jsonl_size(path)
        jsonl_sizes[str(path)] = result
        return result

    with (
        patch.object(EPath, "open", opened),
        patch.object(EPath, "is_file", is_file),
        patch.object(EPath, "size", size),
        patch.object(IJsonlIndexReader, "size", staticmethod(jsonl_size)),
    ):
        yield

    document = {
        "format_version": 1,
        "recipe_identity": recipe_identity,
        "contents": {
            key: base64.b64encode(value).decode("ascii") for key, value in contents.items()
        },
        "file_checks": checks,
        "sizes": sizes,
        "jsonl_sizes": jsonl_sizes,
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    temporary = output.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)
    output.with_suffix(".audit.json").write_text(
        json.dumps(
            {
                "snapshot_sha256": hashlib.sha256(payload).hexdigest(),
                "recipe_identity": recipe_identity,
                "calls": calls,
                "audit": audit,
                "captured_bytes": captured_bytes,
                "package_bytes": len(payload),
                "capture_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )
