# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import re

split_name_re = re.compile(r"^((?:.*/|)[^.]+)[.]([^/]*)$")
skip_meta_re = re.compile(r"__[^/]*__($|/)")
MAIN_FOLDER_NAME = ".nv-meta"
JOIN_INDEX_FILENAME = "join_index.bin"
INDEX_SQLITE_FILENAME = "index.sqlite"
INDEX_UUID_FILENAME = "index.uuid"
INFO_JSON_FILENAME = ".info.json"
INFO_YAML_FILENAME = ".info.yaml"
SPLIT_YAML_FILENAME = "split.yaml"
