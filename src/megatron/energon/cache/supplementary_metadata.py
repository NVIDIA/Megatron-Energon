# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in lookup of completed private metadata indexes; never rewrites sources.

Call install(preparation_root) inside each loader owner before constructing its
workers. Media payload access and SourceInfo remain with the original stores.
Only complete indexes are selected. Uncovered lookups preserve the old behavior.
"""

import hashlib
import json
import os
import sqlite3
import threading
from collections import Counter
from pathlib import Path

STATE = threading.local()
COUNTERS = Counter()


def install(preparation_root):
    from megatron.energon.cache.file_store import SystemFileStore, WebdatasetFileStore
    from megatron.energon.media.metadata import deserialize_media_metadata

    root = Path(preparation_root)
    manifest = (root / "ready-map.json").read_bytes()
    plan = json.loads(manifest)
    ready = {source.rstrip("/"): item for source, item in plan["sources"].items()}
    # Optional, separately verified coverage of observed keys while whole roots build.
    observed = root / "observed/index.sqlite"
    observed_validation = root / "observed/validation.json"
    observed_ready = (
        observed.exists()
        and observed_validation.exists()
        and json.loads(observed_validation.read_text()).get("status") == "PASS"
    )

    def connection(path):
        if getattr(STATE, "pid", None) != os.getpid():
            STATE.pid = os.getpid()
            STATE.connections = {}
        key = str(path)
        if key not in STATE.connections:
            STATE.connections[key] = sqlite3.connect(
                path.as_uri() + "?mode=ro&immutable=1", uri=True
            )
        return STATE.connections[key]

    def wrap(original):
        def get_media_metadata(store, key):
            source = str(store.get_path()).rstrip("/")
            item = ready.get(source)
            if item is not None:
                full_key = item["key_prefix"] + key
                row = (
                    connection(root / "databases" / (item["id"] + ".sqlite"))
                    .execute(
                        "SELECT metadata_type,metadata_json FROM media_metadata WHERE entry_key=?",
                        (full_key,),
                    )
                    .fetchone()
                )
                if row is not None:
                    COUNTERS["private_metadata_hits"] += 1
                    return deserialize_media_metadata(*row)
                COUNTERS["private_metadata_misses"] += 1
            else:
                COUNTERS["source_not_ready"] += 1
            if observed_ready:
                row = (
                    connection(observed)
                    .execute(
                        "SELECT metadata_type,metadata_json FROM media_metadata WHERE source=? AND entry_key=?",
                        (source, key),
                    )
                    .fetchone()
                )
                if row is not None:
                    COUNTERS["observed_metadata_hits"] += 1
                    return deserialize_media_metadata(*row)
            return original(store, key)

        return get_media_metadata

    for cls in (SystemFileStore, WebdatasetFileStore):
        if getattr(cls.get_media_metadata, "_private_metadata_lookup", False):
            raise RuntimeError("Metadata lookup already installed")
        cls.get_media_metadata = wrap(cls.get_media_metadata)
        cls.get_media_metadata._private_metadata_lookup = True
    return {
        "ready_sources": len(ready),
        "total_sources": plan["total_sources"],
        "ready_indexes": len({x["id"] for x in ready.values()}),
        "preparation_root": str(root),
        "observed_index_enabled": observed_ready,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
