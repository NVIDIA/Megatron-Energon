# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar

from megatron.energon.epathlib import EPath
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

T = TypeVar("T")


@dataclass(frozen=True)
class ResumeSubset:
    """Hashable checkpoint representation of a primary dataset subset."""

    range: tuple[float, float] | None = None
    absolute_range: tuple[int, int | None] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "range": self.range,
            "absolute_range": self.absolute_range,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object] | None,
    ) -> "ResumeSubset | None":
        if data is None:
            return None

        range_data = data.get("range")
        if range_data is not None:
            if not isinstance(range_data, (list, tuple)) or len(range_data) != 2:
                raise ValueError(f"Resume subset range must contain two values: {range_data!r}")
            range_start, range_end = range_data
            if not isinstance(range_start, (int, float)) or not isinstance(range_end, (int, float)):
                raise ValueError(f"Resume subset range must be numeric: {range_data!r}")
            subset_range = (float(range_start), float(range_end))
        else:
            subset_range = None

        absolute_range_data = data.get("absolute_range")
        if absolute_range_data is not None:
            if not isinstance(absolute_range_data, (list, tuple)) or len(absolute_range_data) != 2:
                raise ValueError(
                    f"Resume subset absolute_range must contain two values: {absolute_range_data!r}"
                )
            absolute_start, absolute_end = absolute_range_data
            if not isinstance(absolute_start, int):
                raise ValueError(
                    f"Resume subset absolute_range start must be an integer: {absolute_start!r}"
                )
            if absolute_end is not None and not isinstance(absolute_end, int):
                raise ValueError(
                    f"Resume subset absolute_range end must be an integer or None: {absolute_end!r}"
                )
            absolute_range = (absolute_start, absolute_end)
        else:
            absolute_range = None

        return cls(range=subset_range, absolute_range=absolute_range)


@dataclass(frozen=True)
class ResumeIdentity:
    """Checkpoint-stable identity for reusing dataset progress across recipe edits."""

    path: str
    split_part: str | None = None
    subset: ResumeSubset | None = None
    filter_name: str | None = None
    aux: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "split_part": self.split_part,
            "subset": None if self.subset is None else self.subset.to_dict(),
            "filter_name": self.filter_name,
            "aux": list(self.aux),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResumeIdentity":
        return cls(
            path=data["path"],
            split_part=data.get("split_part"),
            subset=ResumeSubset.from_dict(data.get("subset")),
            filter_name=data.get("filter_name"),
            aux=tuple((str(key), str(value)) for key, value in data.get("aux", ())),
        )


def make_resume_identity(
    *,
    path: str | Path | EPath,
    split_part: str | None,
    subset: ResumeSubset | None = None,
    filter_name: str | None = None,
    aux: dict[str, Any] | None = None,
) -> ResumeIdentity:
    """Create a recipe leaf identity for checkpoint resume matching."""

    aux_items = tuple(
        (str(key), str(value))
        for key, value in sorted((aux or {}).items(), key=lambda item: item[0])
    )
    return ResumeIdentity(
        path=str(path),
        split_part=split_part,
        subset=subset,
        filter_name=filter_name,
        aux=aux_items,
    )


def _normalize_identities(
    identities: Iterable[ResumeIdentity | dict[str, Any]],
) -> tuple[ResumeIdentity, ...]:
    normalized: list[ResumeIdentity] = []
    for identity in identities:
        if isinstance(identity, ResumeIdentity):
            normalized.append(identity)
        else:
            normalized.append(ResumeIdentity.from_dict(identity))
    return tuple(sorted(normalized, key=lambda identity: _identity_key(identity)))


def _identity_key(identity: ResumeIdentity) -> tuple[Any, ...]:
    return (
        identity.path,
        identity.split_part,
        repr(identity.subset),
        identity.filter_name,
        identity.aux,
    )


def _identity_dicts(identities: Sequence[ResumeIdentity]) -> list[dict[str, Any]]:
    return [identity.to_dict() for identity in identities]


@dataclass(frozen=True)
class ResumeDatasetMetadata:
    type: str
    config: dict[str, Any]
    identities: tuple[ResumeIdentity, ...]
    children: tuple["ResumeDatasetMetadata", ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResumeDatasetMetadata":
        return cls(
            type=data["type"],
            config=data["config"],
            identities=_normalize_identities(data.get("identities", ())),
            children=tuple(cls.from_dict(child) for child in data.get("children", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "config": self.config,
            "identities": _identity_dicts(self.identities),
            "children": tuple(child.to_dict() for child in self.children),
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
    if isinstance(metadata, ResumeMetadata):
        return metadata
    return ResumeMetadata.from_dict(metadata)


def resume_children(dataset: SavableDataset[Any]) -> tuple[SavableDataset[Any], ...]:
    """Return resume traversal children for a runtime dataset node."""

    if isinstance(dataset, BaseWrapperDataset):
        return dataset.datasets
    return ()


def _identity_from_config_value(value: Any) -> tuple[ResumeIdentity, ...]:
    if isinstance(value, dict):
        if "_path" in value:
            subset_data = value.get("subset")
            if subset_data is not None and not isinstance(subset_data, Mapping):
                raise ValueError(f"Resume subset must be a mapping or None: {subset_data!r}")
            split_part = value.get("split_part")
            filter_name = value.get("filter_name")
            return (
                make_resume_identity(
                    path=str(value["_path"]),
                    split_part=None if split_part is None else str(split_part),
                    subset=ResumeSubset.from_dict(subset_data),
                    filter_name=None if filter_name is None else str(filter_name),
                ),
            )

        identities: list[ResumeIdentity] = []
        for child_value in value.values():
            identities.extend(_identity_from_config_value(child_value))
        return _normalize_identities(identities)

    if isinstance(value, (list, tuple)):
        identities = []
        for child_value in value:
            identities.extend(_identity_from_config_value(child_value))
        return _normalize_identities(identities)

    return ()


def _metadata_for_dataset(
    dataset: SavableDataset[Any],
) -> ResumeDatasetMetadata:
    children = resume_children(dataset)
    children_metadata = tuple(_metadata_for_dataset(child) for child in children)
    identities = _identity_from_config_value(dataset.config())
    if not identities:
        identities = _normalize_identities(
            identity
            for child_metadata in children_metadata
            for identity in child_metadata.identities
        )
    return ResumeDatasetMetadata(
        type=type(dataset).__name__,
        config=dataset.config(),
        identities=identities,
        children=children_metadata,
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
    return ResumeMetadata(version=1, root=_metadata_for_dataset(root))


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
    return state


def _same_identity_set(left: ResumeDatasetMetadata, right: ResumeDatasetMetadata) -> bool:
    left_ids = left.identities
    right_ids = right.identities
    return bool(left_ids) and left_ids == right_ids


def _identity_overlap(left: ResumeDatasetMetadata, right: ResumeDatasetMetadata) -> int:
    return len(set(left.identities) & set(right.identities))


def _is_exact_restore_compatible(
    current_meta: ResumeDatasetMetadata, saved_meta: ResumeDatasetMetadata
) -> bool:
    return (
        current_meta.type == saved_meta.type
        and current_meta.config == saved_meta.config
        and len(current_meta.children) == len(saved_meta.children)
    )


def _find_saved_child(
    current_child_meta: ResumeDatasetMetadata,
    saved_children_meta: list[ResumeDatasetMetadata],
    used_saved_indexes: set[int],
) -> int | None:
    for idx, saved_child_meta in enumerate(saved_children_meta):
        if idx not in used_saved_indexes and _same_identity_set(
            current_child_meta, saved_child_meta
        ):
            return idx

    overlaps = [
        (idx, _identity_overlap(current_child_meta, saved_child_meta))
        for idx, saved_child_meta in enumerate(saved_children_meta)
        if idx not in used_saved_indexes
    ]
    overlaps = [(idx, overlap) for idx, overlap in overlaps if overlap > 0]
    if len(overlaps) == 1:
        return overlaps[0][0]
    if overlaps:
        overlaps.sort(key=lambda item: item[1], reverse=True)
        if len(overlaps) == 1 or overlaps[0][1] > overlaps[1][1]:
            return overlaps[0][0]
    return None


def _migrate_dataset_state(
    current_dataset: SavableDataset[Any],
    current_meta: ResumeDatasetMetadata,
    saved_state: FlexState,
    saved_meta: ResumeDatasetMetadata,
) -> FlexState:
    if _is_exact_restore_compatible(current_meta, saved_meta):
        return deepcopy(saved_state)

    fresh_state = _fresh_dataset_state(current_dataset)
    current_children = list(resume_children(current_dataset))
    current_children_meta = list(current_meta.children)
    saved_children_meta = list(saved_meta.children)
    saved_children_state = list(saved_state.get("datasets", ()))

    if not current_children or "datasets" not in fresh_state:
        if _same_identity_set(current_meta, saved_meta) and current_meta.type == saved_meta.type:
            return deepcopy(saved_state)
        return fresh_state

    migrated_children = list(fresh_state["datasets"])
    used_saved_indexes: set[int] = set()

    positional_restore = (
        current_meta.type == saved_meta.type
        and len(current_children) == len(saved_children_state)
        and len(current_children_meta) == len(saved_children_meta)
        and not isinstance(current_dataset, BlendDataset)
    )

    for current_idx, (child, child_meta) in enumerate(zip(current_children, current_children_meta)):
        saved_idx: int | None
        if positional_restore:
            saved_idx = current_idx
        else:
            saved_idx = _find_saved_child(child_meta, saved_children_meta, used_saved_indexes)

        if saved_idx is None or saved_idx >= len(saved_children_state):
            continue

        used_saved_indexes.add(saved_idx)
        migrated_children[current_idx] = _migrate_dataset_state(
            child,
            child_meta,
            saved_children_state[saved_idx],
            saved_children_meta[saved_idx],
        )

    fresh_state["datasets"] = migrated_children
    return fresh_state


def _migrate_dataset_state_for_root(
    current_root: SavableDataset[Any],
    saved_dataset_state: FlexState,
    saved_metadata: ResumeMetadata,
) -> FlexState:
    current_metadata = collect_resume_metadata(current_root).root
    saved_root_metadata = saved_metadata.root
    return _migrate_dataset_state(
        current_root,
        current_metadata,
        saved_dataset_state,
        saved_root_metadata,
    )


def migrate_dataloader_state(
    dataloader: SavableDataLoader[Any],
    saved_state: SavableDataLoaderState,
    saved_metadata: ResumeMetadata | dict[str, Any],
) -> SavableDataLoaderState:
    """Build a current-topology dataloader state with matched saved leaf progress overlaid."""

    root = _dataloader_root_dataset(dataloader)
    saved_metadata = _normalize_metadata(saved_metadata)
    migrated_worker_states: list[FlexState] = []

    if dataloader.num_workers == 0:
        assert len(saved_state.worker_states) == 1
        migrated_worker_states.append(
            _migrate_dataset_state_for_root(
                root,
                saved_state.worker_states[0],
                saved_metadata,
            )
        )
    else:
        for worker_state in saved_state.worker_states:
            assert isinstance(worker_state, SavableDatasetCheckpoint)
            if worker_state.state is None:
                migrated_worker_states.append(worker_state)
                continue
            migrated_dataset_state = _migrate_dataset_state_for_root(
                root,
                worker_state.state.dataset_state,
                saved_metadata,
            )
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

    return SavableDataLoaderState(
        worker_states=migrated_worker_states,
        next_worker_id=saved_state.next_worker_id,
        micro_batch_size=saved_state.micro_batch_size,
    )
