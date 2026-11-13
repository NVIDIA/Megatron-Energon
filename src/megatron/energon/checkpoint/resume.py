# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterator

from megatron.energon.flavors.base_dataset import FlexState, SavableDataset
from megatron.energon.rng import WorkerRng
from megatron.energon.savable import Savable
from megatron.energon.savable_loader import (
    SavableDataLoader,
    SavableDataLoaderState,
    SavableDatasetCheckpoint,
    SavableDatasetState,
    SavableDatasetWrapper,
    SimpleSavableDatasetWrapper,
)
from megatron.energon.wrappers.base import BaseWrapperDataset
from megatron.energon.wrappers.blend_dataset import BlendDataset
from megatron.energon.wrappers.group_batch_dataset import GroupBatchDataset

RESUME_METADATA_VERSION = 1

#: The saved states by config hash, each with its exhausted flag in the saved blend.
_SavedNodes = dict[str, list[tuple[FlexState, bool]]]


@dataclass(frozen=True)
class ResumeDatasetMetadata:
    """Resume metadata of one node of the dataset tree.

    Attributes:
        type: The class name of the dataset.
        config_hash: Digest of the configuration of the node and of all nodes below it.
        datasets: Descriptions of the recipe datasets read by the node. Only set on the innermost
            node that names them.
        children: The metadata of the wrapped datasets.
    """

    type: str
    config_hash: str
    datasets: tuple[str, ...] = ()
    children: tuple["ResumeDatasetMetadata", ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResumeDatasetMetadata":
        return cls(
            type=data["type"],
            config_hash=data["config_hash"],
            datasets=tuple(data.get("datasets", ())),
            children=tuple(cls.from_dict(child) for child in data.get("children", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "config_hash": self.config_hash,
            "datasets": list(self.datasets),
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(frozen=True)
class ResumeMetadata:
    version: int
    root: ResumeDatasetMetadata

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResumeMetadata":
        return cls(
            version=data["version"],
            root=ResumeDatasetMetadata.from_dict(data["root"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "root": self.root.to_dict(),
        }


def _normalize_metadata(metadata: ResumeMetadata | dict[str, Any]) -> ResumeMetadata:
    if not isinstance(metadata, ResumeMetadata):
        metadata = ResumeMetadata.from_dict(metadata)
    if metadata.version != RESUME_METADATA_VERSION:
        raise ValueError(
            f"Unsupported resume metadata version {metadata.version}; "
            f"expected {RESUME_METADATA_VERSION}."
        )
    return metadata


def _canonical_config(value: Any) -> Any:
    """Normalize a config, so that it is hashed equally for every loader of the same recipe."""
    if isinstance(value, dict):
        return {str(key): _canonical_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_config(item) for item in value]
    return value


def _describe_datasets(config: Any) -> Iterator[str]:
    """Describe the recipe datasets named by a config, i.e. its mappings having a ``_path``."""
    if isinstance(config, dict):
        if "_path" in config:
            details = [
                f"{name}={config[name]!r}"
                for name in ("split_part", "subset", "filter_name")
                if config.get(name) is not None
            ]
            yield f"{config['_path']} ({', '.join(details)})" if details else str(config["_path"])
        else:
            for item in config.values():
                yield from _describe_datasets(item)
    elif isinstance(config, list):
        for item in config:
            yield from _describe_datasets(item)


def _datasets(meta: ResumeDatasetMetadata) -> list[str]:
    """Descriptions of all recipe datasets read by a node or by the nodes below it."""
    return [*meta.datasets, *(dataset for child in meta.children for dataset in _datasets(child))]


def _metadata_for_dataset(dataset: SavableDataset[Any]) -> ResumeDatasetMetadata:
    children: tuple[ResumeDatasetMetadata, ...] = ()
    if isinstance(dataset, BaseWrapperDataset):
        children = tuple(_metadata_for_dataset(child) for child in dataset.datasets)
    config = _canonical_config(dataset.config())
    digest = hashlib.sha256(
        json.dumps(
            [type(dataset).__name__, config, [child.config_hash for child in children]],
            sort_keys=True,
            default=str,
        ).encode()
    )
    datasets: tuple[str, ...] = ()
    if not any(_datasets(child) for child in children):
        # The config of a wrapper repeats the configs of its children, which already name them.
        datasets = tuple(sorted(_describe_datasets(config)))
    return ResumeDatasetMetadata(
        type=type(dataset).__name__,
        config_hash=digest.hexdigest(),
        datasets=datasets,
        children=children,
    )


def _dataloader_root_dataset(
    dataloader_or_dataset: SavableDataLoader[object] | SavableDataset[object],
) -> SavableDataset[object]:
    if isinstance(dataloader_or_dataset, SavableDataLoader):
        dataset = dataloader_or_dataset.dataset
        if isinstance(dataset, SavableDatasetWrapper):
            return dataset.dataset
        if isinstance(dataset, SimpleSavableDatasetWrapper):
            return dataset
        raise TypeError(f"Unsupported SavableDataLoader dataset type: {type(dataset)!r}")
    if isinstance(dataloader_or_dataset, SavableDataset):
        return dataloader_or_dataset
    raise TypeError(f"Unsupported resume metadata root: {type(dataloader_or_dataset)!r}")


def collect_resume_metadata(
    loader_or_dataset: SavableDataLoader[object] | SavableDataset[object],
) -> ResumeMetadata:
    """Collect checkpoint resume metadata from a dataloader or runtime dataset tree."""

    root = _dataloader_root_dataset(loader_or_dataset)
    return ResumeMetadata(version=RESUME_METADATA_VERSION, root=_metadata_for_dataset(root))


def _fresh_savable_state(value: Savable) -> Any:
    if isinstance(value, WorkerRng):
        return FlexState(rng=None)
    if isinstance(value, SavableDataset):
        return _fresh_dataset_state(value)
    return value.save_state()


def _fresh_dataset_state(dataset: SavableDataset[Any]) -> FlexState:
    state = FlexState()
    state["__class__"] = type(dataset).__name__
    for key in dataset._savable_fields:
        attr = getattr(dataset, key)
        if isinstance(attr, Savable):
            state[key] = _fresh_savable_state(attr)
        else:
            state[key] = deepcopy(attr)

    if isinstance(dataset, BaseWrapperDataset):
        state["datasets"] = [_fresh_dataset_state(child) for child in dataset.datasets]
    if isinstance(dataset, GroupBatchDataset):
        state["bucket_sample_index"] = dataset._group_key_sample_index.save_state()
        state["batch_sample_index"] = dataset._batch_sample_index.save_state()
        state["buckets"] = {}
        state["source_exhausted"] = False
    return state


def _matchable_hashes(meta: ResumeDatasetMetadata) -> Iterator[str]:
    """Config hashes of all nodes whose saved state may be reused.

    A node without recipe datasets is not matched on its own, because nodes of different datasets
    can share its config, e.g. the samplers of two datasets with the same shards.
    """
    if _datasets(meta):
        yield meta.config_hash
        for child in meta.children:
            yield from _matchable_hashes(child)


def _index_saved_nodes(
    saved_state: FlexState,
    saved_meta: ResumeDatasetMetadata,
    saved_nodes: _SavedNodes,
    exhausted: bool = False,
) -> None:
    """Collect the saved states of all matchable nodes by their config hash."""
    if not _datasets(saved_meta):
        return
    saved_nodes.setdefault(saved_meta.config_hash, []).append((saved_state, exhausted))
    child_states = saved_state.get("datasets", ())
    if saved_meta.type == BlendDataset.__name__:
        children_exhausted = saved_state["exhausted"]
    else:
        children_exhausted = [False] * len(child_states)
    for child_state, child_meta, child_exhausted in zip(
        child_states, saved_meta.children, children_exhausted
    ):
        _index_saved_nodes(child_state, child_meta, saved_nodes, child_exhausted)


def _migrate_dataset_state(
    current_dataset: SavableDataset[Any],
    current_meta: ResumeDatasetMetadata,
    current_hashes: Counter[str],
    saved_nodes: _SavedNodes,
    retained: set[str],
) -> tuple[FlexState, bool]:
    """Migrate the saved state to a node of the current dataset tree.

    A node continues from the saved node with the same config hash, wherever that is located in
    the saved tree. Otherwise it starts fresh and its children are migrated.

    Args:
        current_dataset: The current dataset node.
        current_meta: The resume metadata of the current node.
        current_hashes: Counts the config hashes of the matchable nodes of the current tree.
        saved_nodes: The saved states of the matchable nodes by their config hash.
        retained: Collects the descriptions of all datasets whose saved progress is reused.

    Returns:
        The state to restore the current node from, and whether the node already finished the
        current pass of the saved blend.
    """
    candidates = saved_nodes.get(current_meta.config_hash, ())
    # Several nodes with the same config, e.g. of a dataset listed twice, cannot be told apart.
    if len(candidates) == 1 and current_hashes[current_meta.config_hash] == 1:
        saved_state, exhausted = candidates[0]
        retained.update(_datasets(current_meta))
        return deepcopy(saved_state), exhausted

    state = _fresh_dataset_state(current_dataset)
    if isinstance(current_dataset, BaseWrapperDataset):
        children = [
            _migrate_dataset_state(child, child_meta, current_hashes, saved_nodes, retained)
            for child, child_meta in zip(current_dataset.datasets, current_meta.children)
        ]
        state["datasets"] = [child_state for child_state, _ in children]
        if isinstance(current_dataset, BlendDataset):
            # A child which already finished the current pass must not be iterated again in it.
            state["exhausted"] = [child_exhausted for _, child_exhausted in children]
    return state, False


def _warn_lost_progress(lost: set[str]) -> None:
    if not lost:
        return
    warnings.warn(
        f"Checkpoint migration could not reuse the saved progress of {len(lost)} dataset(s) that "
        f"are still part of the recipe, they start from a fresh state: {', '.join(sorted(lost))}. "
        "The progress of a dataset is only reused if it is listed once and if its configuration "
        "and its assignment of samples to workers are unchanged. With more than one worker, that "
        "assignment changes if datasets in front of it are added or deleted. To remove a dataset, "
        "set its weight to 0 instead of deleting it.",
        UserWarning,
        stacklevel=3,
    )


def migrate_dataloader_state(
    dataloader: SavableDataLoader[Any],
    saved_state: SavableDataLoaderState | None,
    saved_metadata: ResumeMetadata | dict[str, Any],
) -> SavableDataLoaderState | None:
    """Build a state for the loader of a changed recipe, reusing the saved dataset progress.

    The supported recipe changes are adding datasets and setting the weight of datasets to 0.
    Emits a :class:`UserWarning` naming the datasets which are still part of the recipe, but whose
    saved progress could not be reused.

    Args:
        dataloader: The newly constructed loader to migrate the state to.
        saved_state: The state saved by ``save_state_rank`` of the previous loader. May be None,
            if that loader had nothing to save yet.
        saved_metadata: The resume metadata collected from the previous loader.

    Returns:
        The state to pass to ``restore_state_rank`` of the given loader. None, if no state was
        saved.
    """

    if saved_state is None:
        return None

    root = _dataloader_root_dataset(dataloader)
    saved_root_meta = _normalize_metadata(saved_metadata).root
    current_root_meta = collect_resume_metadata(root).root
    current_hashes = Counter(_matchable_hashes(current_root_meta))
    expected = set(_datasets(current_root_meta)) & set(_datasets(saved_root_meta))
    lost: set[str] = set()

    def migrate(saved_dataset_state: FlexState) -> FlexState:
        saved_nodes: _SavedNodes = {}
        _index_saved_nodes(saved_dataset_state, saved_root_meta, saved_nodes)
        retained: set[str] = set()
        migrated_dataset_state, _ = _migrate_dataset_state(
            root, current_root_meta, current_hashes, saved_nodes, retained
        )
        lost.update(expected - retained)
        return migrated_dataset_state

    migrated_worker_states: list[FlexState] = []

    if dataloader.num_workers == 0:
        assert len(saved_state.worker_states) == 1
        saved_root_state = saved_state.worker_states[0]
        migrated_root_state = migrate(saved_root_state)
        # As for worker processes, the sample index of the loader does not depend on the dataset
        # tree and continues.
        migrated_root_state["_sample_index"] = saved_root_state["_sample_index"]
        migrated_worker_states.append(migrated_root_state)
    else:
        for worker_state in saved_state.worker_states:
            assert isinstance(worker_state, SavableDatasetCheckpoint)
            if worker_state.state is None:
                migrated_worker_states.append(worker_state)
                continue
            migrated_dataset_state = migrate(worker_state.state.dataset_state)
            migrated_worker_states.append(
                SavableDatasetCheckpoint(
                    state=SavableDatasetState(
                        rng=worker_state.state.rng,
                        dataset_state=migrated_dataset_state,
                        sample_index=worker_state.state.sample_index,
                    ),
                    offset=worker_state.offset,
                )
            )

    _warn_lost_progress(lost)

    return SavableDataLoaderState(
        worker_states=migrated_worker_states,
        next_worker_id=saved_state.next_worker_id,
        micro_batch_size=saved_state.micro_batch_size,
    )
