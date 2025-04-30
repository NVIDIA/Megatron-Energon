# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, BinaryIO, Union

import ryml


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
        return {
            t.key(ch).tobytes().decode(): _to_native(t, ch) for ch in ryml.children(t, nid)
        }  # children() helper&#8203;:contentReference[oaicite:3]{index=3}
    if t.is_seq(nid):
        return [_to_native(t, ch) for ch in ryml.children(t, nid)]
    # scalar leaf
    return _cast_scalar(t, nid)


def ryml_tree_to_native(tree: ryml.Tree):
    return _to_native(tree, tree.root_id())


def load_yaml(stream: Union[BinaryIO, bytes]) -> Any:
    if isinstance(stream, bytes):
        bytes_data = stream
    else:
        assert isinstance(stream, BinaryIO)
        bytes_data = stream.read()
    tree = ryml.parse_in_arena(bytes_data)
    native_obj = ryml_tree_to_native(tree)

    return native_obj
