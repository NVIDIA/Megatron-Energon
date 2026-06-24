# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import os
import re

split_name_re = re.compile(r"^((?:.*/|)[^.]+)[.]([^/]*)$")
skip_meta_re = re.compile(r"__[^/]*__($|/)")
MAIN_FOLDER_NAME = ".nv-meta"
JOIN_INDEX_FILENAME = "join_index.bin"
INDEX_BATCH_SIZE = 128
INDEX_SQLITE_FILENAME = "index.sqlite"
INDEX_LMDB_DIRNAME = "index.lmdb"
# Default map size for prepare (1 TiB); override via ENERGON_INDEX_LMDB_MAP_SIZE bytes.
INDEX_LMDB_MAP_SIZE = int(os.environ.get("ENERGON_INDEX_LMDB_MAP_SIZE", str(1 << 40)))
INDEX_UUID_FILENAME = "index.uuid"
INFO_JSON_FILENAME = ".info.json"
INFO_YAML_FILENAME = ".info.yaml"
SPLIT_YAML_FILENAME = "split.yaml"
