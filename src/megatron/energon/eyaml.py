# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import json
from typing import Any, BinaryIO, Union

import ryml

from megatron.energon.epathlib.epath import EPath


def load_yaml_json(path: EPath) -> Any:
    """Load a YAML or JSON file from a path. Loading json is much faster than yaml.

    Args:
        path: The path to the file to load.

    Returns:
        The parsed YAML or JSON data as a native Python object.
    """
    if path.name.endswith(".yaml"):
        with path.open("rb") as f:
            return load_yaml(f)
    elif path.name.endswith(".json"):
        with path.open("rb") as f:
            return json.load(f)
    else:
        raise ValueError(f"Invalid file extension: {path.name}")


def load_yaml(stream: Union[BinaryIO, bytes]) -> Any:
    """Load a YAML file from a stream or bytes object using rapidyaml/ryml.
    This is much faster than the standard yaml library, but we don't
    support all YAML features.

    Args:
        stream: A stream or bytes object containing the YAML data.

    Returns:
        The parsed YAML data as a native Python object.
    """

    if isinstance(stream, bytes):
        bytes_data = stream
    else:
        bytes_data = stream.read()
    tree = ryml.parse_in_arena(bytes_data)
    native_obj = _ryml_tree_to_native(tree)

    return native_obj


def _cast_scalar(tree: ryml.Tree, nid: int) -> object:
    """Convert YAML scalar to bool | int | float | None | str."""
    memview = tree.val(nid)
    s = memview.tobytes().decode()

    if tree.is_val_quoted(nid):
        return s

    s_lower = s.lower()

    if s_lower in {"null", "~"}:
        return None
    if s_lower in {"true", "yes"}:
        return True
    if s_lower in {"false", "no"}:
        return False
    try:
        if "." in s or "e" in s_lower:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _to_native(t: ryml.Tree, nid: int):
    if t.is_map(nid):
        # iterate children of a mapping node
        return {t.key(ch).tobytes().decode(): _to_native(t, ch) for ch in ryml.children(t, nid)}
    if t.is_seq(nid):
        return [_to_native(t, ch) for ch in ryml.children(t, nid)]
    # scalar leaf
    return _cast_scalar(t, nid)


def _ryml_tree_to_native(tree: ryml.Tree):
    return _to_native(tree, tree.root_id())
