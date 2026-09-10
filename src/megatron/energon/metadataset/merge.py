# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import replace
from typing import Any

from megatron.energon.cache.file_store import SystemFileStore, WebdatasetFileStore
from megatron.energon.metadataset.loader_interface import LoadedDataset


def _same_config(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            _same_config(value, right[key]) for key, value in left.items()
        )
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(_same_config(a, b) for a, b in zip(left, right))
    if type(left) in (str, int, float, bool, bytes, type(None)):
        return left == right
    # Custom values (such as decoders or tensors) must be the same object.
    return False


def _aux_config(dataset: LoadedDataset) -> dict:
    return {
        name: (type(store), store.get_path())
        if type(store) in (SystemFileStore, WebdatasetFileStore)
        else store
        for name, store in (dataset.aux or {}).items()
    }


def merge_weighted_datasets(datasets: list[LoadedDataset]) -> list[LoadedDataset]:
    """Combine identical leaf loading inputs without creating duplicate sample streams."""
    merged: list[LoadedDataset] = []
    by_path: dict[str, list[LoadedDataset]] = {}
    for dataset in datasets:
        source = dataset._source_config
        if source is None:
            merged.append(dataset)
            continue
        candidates = by_path.setdefault(source["path"], [])
        for previous in candidates:
            if _same_config(source, previous._source_config) and _same_config(
                _aux_config(dataset), _aux_config(previous)
            ):
                previous.weight += dataset.weight
                warnings.warn(
                    f"Combining duplicate weighted dataset {source['path']} into one sample stream.",
                    UserWarning,
                    stacklevel=2,
                )
                break
        else:
            entry = replace(dataset)
            candidates.append(entry)
            merged.append(entry)
    return merged
