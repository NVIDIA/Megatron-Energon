# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

from megatron.energon.cache import FileStore
from megatron.energon.cache.file_store import ByteRangeStore, SystemFileStore
from megatron.energon.dataset_config import load_config
from megatron.energon.deprecation import warn_deprecated
from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.common.manifest.paths import INDEX_SQLITE_FILENAME, MAIN_FOLDER_NAME
from megatron.energon.flavors.common.manifest.types import DatasetSubset
from megatron.energon.flavors.dataset_type import EnergonDatasetType, get_dataset_type
from megatron.energon.flavors.webdataset.file_store import WebdatasetFileStore
from megatron.energon.recipe.dataset_loader import DatasetLoader
from megatron.energon.recipe.loader_interface import (
    DatasetBlendMode,
    DatasetLoaderInterface,
    LoadedDatasetList,
    TraversedDatasetReference,
)
from megatron.energon.tags import TagsAlias, resolve_tags
from megatron.energon.worker import WorkerConfig

# Regex for any URL-like string (any protocol)
url_regex = re.compile(r"^(?P<protocol>[a-z][a-z0-9+.-]*)://(?P<path>.*)", re.IGNORECASE)
aux_filestore_protocol_regex = re.compile(r"^[a-z][a-z0-9.-]*$", re.IGNORECASE)


class AuxFileStoreReference:
    """Base class for auxiliary recipe references that materialize as FileStores."""

    def _resolve_path(self, recipe_path: Optional[EPath]) -> EPath:
        raise NotImplementedError

    def post_initialize(self, recipe_path: Optional[EPath] = None) -> None:
        self._resolve_path(recipe_path)

    def get_file_store(self) -> FileStore:
        raise NotImplementedError

    def get_traversed_path(self) -> EPath:
        raise NotImplementedError


@edataclass
class AuxDatasetReference:
    path: Union[str, EPath]

    def _resolve_path(self, recipe_path: Optional[EPath]) -> EPath:
        assert recipe_path is not None
        if not isinstance(self.path, EPath):
            self.path = recipe_path.parent / self.path
        return self.path

    def post_initialize(self, recipe_path: Optional[EPath] = None) -> None:
        self._resolve_path(recipe_path)
        assert not self.path.is_file(), (
            "Auxiliary datasets must not be recipe, but direct dataset references"
        )
        assert (self.path / MAIN_FOLDER_NAME / INDEX_SQLITE_FILENAME).is_file(), (
            "Auxiliary datasets must be prepared Energon datasets. This one does not exist or is not prepared: "
            + str(self.path)
        )

    def get_file_store(self) -> FileStore:
        assert isinstance(self.path, EPath), "Missing call to post_initialize"
        return WebdatasetFileStore(self.path)


@edataclass
class AuxFilesystemReference(AuxFileStoreReference):
    fs_path: Union[str, EPath]

    def _resolve_path(self, recipe_path: Optional[EPath]) -> EPath:
        assert recipe_path is not None
        if not isinstance(self.fs_path, EPath):
            self.fs_path = recipe_path.parent / self.fs_path
        return self.fs_path

    def post_initialize(self, recipe_path: Optional[EPath] = None) -> None:
        self._resolve_path(recipe_path)

    def get_file_store(self) -> FileStore:
        assert isinstance(self.fs_path, EPath), "Missing call to post_initialize"
        return SystemFileStore(self.fs_path)

    def get_traversed_path(self) -> EPath:
        assert isinstance(self.fs_path, EPath), "Missing call to post_initialize"
        return self.fs_path


@edataclass
class AuxByteRangeStoreReference(AuxFileStoreReference):
    byterange_fs_path: Union[str, EPath]

    def _resolve_path(self, recipe_path: Optional[EPath]) -> EPath:
        assert recipe_path is not None
        if not isinstance(self.byterange_fs_path, EPath):
            self.byterange_fs_path = recipe_path.parent / self.byterange_fs_path
        return self.byterange_fs_path

    def post_initialize(self, recipe_path: Optional[EPath] = None) -> None:
        self._resolve_path(recipe_path)

    def get_file_store(self) -> FileStore:
        assert isinstance(self.byterange_fs_path, EPath), "Missing call to post_initialize"
        return ByteRangeStore(self.byterange_fs_path)

    def get_traversed_path(self) -> EPath:
        assert isinstance(self.byterange_fs_path, EPath), "Missing call to post_initialize"
        return self.byterange_fs_path


AuxReference = Union[AuxDatasetReference, AuxFileStoreReference]
AuxFileStoreProtocolFactory = Callable[[Union[str, EPath]], AuxFileStoreReference]

_AUX_FILESTORE_PROTOCOL_FACTORIES: dict[str, AuxFileStoreProtocolFactory] = {}


def _normalize_aux_filestore_protocol(protocol: str) -> str:
    protocol = protocol.lower()
    if aux_filestore_protocol_regex.fullmatch(protocol) is None:
        raise ValueError(
            f"Invalid auxiliary filestore protocol {protocol!r}. "
            "Use a URI protocol name without '+', ':', or '/'."
        )
    return protocol


def register_aux_filestore_protocol(
    protocol: str,
    factory: AuxFileStoreProtocolFactory,
    *,
    override: bool = False,
) -> None:
    """Register a recipe aux URI protocol that materializes as a FileStore reference."""

    protocol = _normalize_aux_filestore_protocol(protocol)
    if not override and protocol in _AUX_FILESTORE_PROTOCOL_FACTORIES:
        raise ValueError(f"Auxiliary filestore protocol {protocol!r} is already registered")
    _AUX_FILESTORE_PROTOCOL_FACTORIES[protocol] = factory


def _get_aux_filestore_protocol_factory(
    protocol: str,
) -> Optional[AuxFileStoreProtocolFactory]:
    return _AUX_FILESTORE_PROTOCOL_FACTORIES.get(protocol.lower())


register_aux_filestore_protocol("filesystem", lambda path: AuxFilesystemReference(fs_path=path))
register_aux_filestore_protocol(
    "byterange", lambda path: AuxByteRangeStoreReference(byterange_fs_path=path)
)


@edataclass
class Subset:
    """
    A subset range to be applied to a dataset. The range is always consecutive.

    The range is a tuple of two values, where the first value is the start of the subset and the second value is the end of the subset (end not included).
    The range can either be an absolute range with sample indices, or a ratio of the dataset size.
    Relative range example: [25%, 75%]. This would limit the subset to the middle 50% of the dataset.
    Absolute range example: [100, 200]. This would limit the subset to the 100 samples with indices 100-199.
    For absolute ranges, the end can be set to "end" to indicate the end of the dataset, for example [100, end].

    Since subsets can be specified at multiple levels of a hierarchy, for example in a blend,
    their effects can be merged to a single subset.
    Note however, that absolute ranges are only allowed for leaf datasets, while relative ranges
    can be applied at any level.
    """

    range: tuple[str | int, str | int]

    def as_dataset_subset(self) -> DatasetSubset:
        """Convert the subset with string values to a DatasetSubset object with `range` and `absolute_range`."""

        start, end = self.range

        def _conv(value: str | int) -> float | int | None:
            if isinstance(value, int):
                return value
            else:
                assert isinstance(value, str), "Range must be a string if it's not an integer"
                if value.strip() == "end":
                    return None
                assert value.endswith("%"), "Range must be a percentage"
                percentage = float(value.removesuffix("%"))
                assert 0 <= percentage <= 100, "Percentage must be between 0 and 100"
                return percentage / 100.0

        start = _conv(start)
        end = _conv(end)

        if isinstance(start, int):
            assert isinstance(end, int) or end is None, (
                "End must be an integer if start is an integer"
            )
            return DatasetSubset(absolute_range=(start, end), range=(0, 1))
        else:
            assert isinstance(start, float), "Range start must be a float if it's not an integer"
            assert isinstance(end, float) or end is None, "End must be a float if start is a float"
            assert 0 <= start <= 1, "Start must be between 0 and 1"
            assert 0 <= end <= 1, "End must be between 0 and 1"
            assert start <= end, "Start must be less than end"
            return DatasetSubset(range=(start, end), absolute_range=None)

    def merge(self, parent_subset: DatasetSubset | None) -> DatasetSubset:
        """Merge this subset with a parent subset.

        If the parent subset is None, return the subset.
        If the parent subset is an absolute range, fail, because that's not allowed.
        If the parent subset is a ratio, merge it with the subset.

        Merging a child absolute range with a parent relative range:
        In this case, both are kept in the DatasetSubset object and applies in "absolute first" order later.

        Merging a child relative range with a parent relative range:
        In this case, the relative parent range is applied to the child's relative range.
        The absolute range is not affected.

        For details on how this is applied, see `DatasetSubset.compute_subset`.
        """

        assert parent_subset is None or parent_subset.absolute_range is None, (
            f"Cannot merge absolute subset ranges. Absolute ranges are only allowed for a leaf dataset. {self.absolute_range=} {self.range=}"
        )
        my_subset = self.as_dataset_subset()
        if parent_subset is None or parent_subset.range is None:
            return my_subset

        # Assuming inner ratio: [0.25, 0.75] and outer ratio: [0, 0.5]
        # Then the total ratio is supposed to be: [0.25 + 0*0.5, 0.25 + 0.5 * 0.5] = [0.25, 0.5]
        total = my_subset.range[1] - my_subset.range[0]
        return DatasetSubset(
            range=(
                my_subset.range[0] + parent_subset.range[0] * total,
                my_subset.range[0] + parent_subset.range[1] * total,
            ),
            absolute_range=my_subset.absolute_range,
        )


@dataclass(kw_only=True, eq=False)
class SubsetRatioMixin:
    subset: Optional[Subset] = None

    def _get_subset(self, parent_subset: Optional[DatasetSubset]) -> Optional[DatasetSubset]:
        if parent_subset is not None:
            assert parent_subset.absolute_range is None, (
                f"Can only use absolute subset ranges for a leaf dataset (Range {parent_subset.absolute_range=})"
            )
            if self.subset is not None:
                return self.subset.merge(parent_subset)
            else:
                return parent_subset
        elif self.subset is not None:
            return self.subset.merge(None)
        return None


@dataclass(kw_only=True, eq=False)
class ShuffleOverEpochsMultiplierMixin:
    shuffle_over_epochs_multiplier: Optional[int] = 1

    def _merge_shuffle_over_epochs_multiplier(
        self, inherited_shuffle_over_epochs_multiplier: Optional[int]
    ) -> Optional[int]:
        if (
            inherited_shuffle_over_epochs_multiplier is None
            or self.shuffle_over_epochs_multiplier is None
        ):
            # If no shuffling is requested, this has override priority.
            return None
        elif (
            inherited_shuffle_over_epochs_multiplier == -1
            or self.shuffle_over_epochs_multiplier == -1
        ):
            # Next priority is sampling with replacement.
            return -1
        else:
            # Otherwise, multiply the shuffle over epochs multiplier.
            return inherited_shuffle_over_epochs_multiplier * self.shuffle_over_epochs_multiplier


@dataclass(kw_only=True, eq=False)
class TagsMixin(TagsAlias):
    tags: Optional[Dict[str, Any]] = None

    def _merge_tags(self, inherited_tags: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge this reference's tags with the inherited traversal tags.

        The merge order mirrors `get_datasets(...)`: this reference contributes the base mapping,
        and inherited outer-hierarchy tags override on key conflicts.


        Args:
            inherited_tags: Effective tags accumulated from outer recipe
                references during traversal.

        Returns:
            The effective tag mapping for this reference, after applying outer-overrides-inner
            merge semantics.
        """
        if self.tags is not None:
            return {**self.tags, **(inherited_tags or {})}
        return dict(inherited_tags or {})


@edataclass
class DatasetReference(
    SubsetRatioMixin,
    ShuffleOverEpochsMultiplierMixin,
    TagsMixin,
    DatasetLoaderInterface,
):
    path: Union[str, EPath]

    split_part: Optional[str] = None
    dataset_config: Optional[str] = None
    split_config: Optional[str] = None
    filter: Optional[str] = None

    #: Auxiliary datasets. May only be specified for crude datasets for cooking. Cooking will get
    # these references to load data from. If specified as string, it will be interpreted as a
    # dataset path.
    aux: Optional[Dict[str, Union[str, AuxReference]]] = None

    _dataset: Optional[DatasetLoaderInterface] = None

    def _resolve_path(self, recipe_path: Optional[EPath]) -> EPath:
        assert recipe_path is not None
        if not isinstance(self.path, EPath):
            self.path = recipe_path.parent / self.path
        return self.path

    @staticmethod
    def _normalize_aux_reference(
        reference: Union[str, AuxReference],
    ) -> AuxReference:
        if isinstance(reference, (AuxDatasetReference, AuxFileStoreReference)):
            return reference
        if m := url_regex.match(reference):
            prot = m.group("protocol")
            if prot.count("+") == 1:
                # aux_filestore_protocol+fs_prot://...
                aux_protocol, fs_prot = prot.split("+")
                factory = _get_aux_filestore_protocol_factory(aux_protocol)
                if factory is not None:
                    return factory(f"{fs_prot}://{m.group('path')}")
            else:
                factory = _get_aux_filestore_protocol_factory(prot)
                if factory is not None:
                    # registered_protocol://... may be relative or absolute.
                    return factory(m.group("path"))

            # msc:// or any other unregistered protocol remains a prepared aux dataset path.
            return AuxDatasetReference(path=reference)
        return AuxDatasetReference(path=reference)

    def _normalize_aux_references(self, recipe_path: Optional[EPath], *, validate: bool) -> None:
        if self.aux is None:
            return
        new_aux: Dict[str, AuxReference] = {}
        for key, value in self.aux.items():
            normalized = self._normalize_aux_reference(value)
            if validate:
                normalized.post_initialize(recipe_path)
            else:
                normalized._resolve_path(recipe_path)
            new_aux[key] = normalized
        self.aux = new_aux

    def _get_traversed_aux_references(self) -> dict[str, EPath]:
        if self.aux is None:
            return {}
        traversed_aux: dict[str, EPath] = {}
        for key, value in self.aux.items():
            if isinstance(value, AuxDatasetReference):
                assert isinstance(value.path, EPath)
                traversed_aux[key] = value.path
            else:
                assert isinstance(value, AuxFileStoreReference)
                traversed_aux[key] = value.get_traversed_path()
        return traversed_aux

    def _load_nested_recipe(self) -> DatasetLoaderInterface:
        assert isinstance(self.path, EPath)
        assert self.aux is None, "Cannot specify auxiliary datasets for crude datasets"
        assert self.dataset_config is None, "Must not set dataset_config"
        assert self.split_config is None, "Must not set split_config"
        assert self.filter is None, "Must not set filter"
        return load_config(
            self.path,
            default_type=Recipe,
            default_kwargs=dict(path=self.path),
        )

    def post_initialize(self, recipe_path: Optional[EPath] = None) -> None:
        self._resolve_path(recipe_path)
        ds_type = get_dataset_type(self.path)
        if ds_type == EnergonDatasetType.RECIPE:
            self._dataset = self._load_nested_recipe()
            self._dataset.post_initialize()
        elif ds_type in (
            EnergonDatasetType.MANIFEST_DATASET,
            EnergonDatasetType.JSONL,
            EnergonDatasetType.BINIDX,
            EnergonDatasetType.PARQUET,
        ):
            self._dataset = DatasetLoader(
                path=self.path,
                split_config=self.split_config,
                dataset_config=self.dataset_config,
                filter_name=self.filter,
            )
            self._dataset.post_initialize()
            self._normalize_aux_references(recipe_path, validate=True)
        elif ds_type == EnergonDatasetType.FILESYSTEM:
            raise ValueError(
                "Filesystem datasets are not supported within recipes except as auxiliary datasets."
            )
        else:
            raise FileNotFoundError(self.path)

    def traverse(
        self,
        recipe_path: Optional[EPath] = None,
        *,
        split_part: Union[Literal["train", "val", "test"], str],
        _shuffle_over_epochs_multiplier: Optional[int] = 1,
        _tags: Optional[Dict[str, Any]] = None,
    ) -> List[TraversedDatasetReference]:
        self._resolve_path(recipe_path)
        _tags = self._merge_tags(_tags)
        _shuffle_over_epochs_multiplier = self._merge_shuffle_over_epochs_multiplier(
            _shuffle_over_epochs_multiplier
        )
        ds_type = get_dataset_type(self.path)
        if ds_type == EnergonDatasetType.RECIPE:
            return self._load_nested_recipe().traverse(
                split_part=self.split_part or split_part,
                _shuffle_over_epochs_multiplier=_shuffle_over_epochs_multiplier,
                _tags=_tags,
            )
        self._normalize_aux_references(recipe_path, validate=False)
        return [
            TraversedDatasetReference(
                path=self.path,
                split_part=self.split_part or split_part,
                aux=self._get_traversed_aux_references(),
                tags=_tags,
                shuffle_over_epochs_multiplier=_shuffle_over_epochs_multiplier,
            )
        ]

    def prepare(self, split_part: Optional[str] = None) -> Sequence[EPath]:
        assert self._dataset is not None
        return self._dataset.prepare(split_part=split_part)

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        tags: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        subset: Optional[DatasetSubset] = None,
        **kwargs,
    ) -> LoadedDatasetList:
        tags = resolve_tags(tags, kwargs.pop("subflavors", None))
        assert self._dataset is not None

        result = self._dataset.get_datasets(
            training=training,
            split_part=self.split_part or split_part,
            worker_config=worker_config,
            tags=self._merge_tags(tags),
            shuffle_over_epochs_multiplier=self._merge_shuffle_over_epochs_multiplier(
                shuffle_over_epochs_multiplier
            ),
            subset=self._get_subset(subset),
            **kwargs,
        )
        if self.aux is not None:
            aux = {k: v.get_file_store() for k, v in self.aux.items()}
            for loaded_dataset in result.datasets:
                if loaded_dataset.aux is None:
                    loaded_dataset.aux = aux
                else:
                    loaded_dataset.aux.update(aux)
        return result


@dataclass
class BlendWeightMixin:
    weight: float = 1.0


@edataclass
class BlendDatasetReference(BlendWeightMixin, DatasetReference):
    pass


@edataclass
class RecipeBlend(
    SubsetRatioMixin,
    ShuffleOverEpochsMultiplierMixin,
    TagsMixin,
    DatasetLoaderInterface,
):
    """Blending of datasets by specifying the sampling weight for the inner datasets."""

    blend: List[Union[BlendDatasetReference, "RecipeBlend"]]
    blend_weight_unit: str = "samples"

    def post_initialize(self, recipe_path: Optional[EPath] = None):
        assert recipe_path is not None
        for dataset in self.blend:
            dataset.post_initialize(recipe_path)

    def traverse(
        self,
        recipe_path: Optional[EPath] = None,
        *,
        split_part: Union[Literal["train", "val", "test"], str],
        _shuffle_over_epochs_multiplier: Optional[int] = 1,
        _tags: Optional[Dict[str, Any]] = None,
    ) -> List[TraversedDatasetReference]:
        assert recipe_path is not None
        _shuffle_over_epochs_multiplier = self._merge_shuffle_over_epochs_multiplier(
            _shuffle_over_epochs_multiplier
        )
        _tags = self._merge_tags(_tags)
        flattened: List[TraversedDatasetReference] = []
        for dataset in self.blend:
            flattened.extend(
                dataset.traverse(
                    recipe_path,
                    split_part=split_part,
                    _shuffle_over_epochs_multiplier=_shuffle_over_epochs_multiplier,
                    _tags=_tags,
                )
            )
        return flattened

    def prepare(self, split_part: Optional[str] = None) -> Sequence[EPath]:
        files = []
        for dataset in self.blend:
            files.extend(dataset.prepare(split_part=split_part))
        return files

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        tags: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        subset: Optional[DatasetSubset] = None,
        **kwargs,
    ) -> LoadedDatasetList:
        tags = resolve_tags(tags, kwargs.pop("subflavors", None))
        subset = self._get_subset(subset)
        tags = self._merge_tags(tags)
        shuffle_over_epochs_multiplier = self._merge_shuffle_over_epochs_multiplier(
            shuffle_over_epochs_multiplier
        )
        sum_weight = sum(dataset.weight for dataset in self.blend)
        datasets = []
        for dataset in self.blend:
            inner_result = dataset.get_datasets(
                training=training,
                split_part=split_part,
                worker_config=worker_config,
                tags=tags,
                shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
                subset=subset,
                **kwargs,
            )
            if inner_result.blend_mode not in (
                DatasetBlendMode.NONE,
                DatasetBlendMode.DATASET_WEIGHT,
            ):
                raise ValueError(
                    "Can only blend datasets which are of the same blend mode. Cannot mix blend with blend_epochized."
                )
            if (
                inner_result.blend_mode == DatasetBlendMode.DATASET_WEIGHT
                and inner_result.blend_weight_unit != self.blend_weight_unit
            ):
                raise ValueError(
                    "Nested dataset-weight blends must use the same blend_weight_unit. "
                    f"Got {self.blend_weight_unit!r} and {inner_result.blend_weight_unit!r}."
                )
            for loaded_dataset in inner_result.datasets:
                if inner_result.blend_mode == DatasetBlendMode.DATASET_WEIGHT:
                    assert isinstance(loaded_dataset.weight, float)
                else:
                    assert inner_result.blend_mode == DatasetBlendMode.NONE
                    assert loaded_dataset.weight is None
                    assert loaded_dataset.repetitions is None
                    loaded_dataset.weight = 1.0
                loaded_dataset.weight = loaded_dataset.weight * dataset.weight / sum_weight
                datasets.append(loaded_dataset)
        return LoadedDatasetList(
            blend_mode=DatasetBlendMode.DATASET_WEIGHT,
            blend_weight_unit=self.blend_weight_unit,
            datasets=datasets,
        )


@dataclass
class BlendRepetitionsMixin:
    repetitions: Union[int, float] = 1


@edataclass
class BlendEpochizedDatasetReference(BlendRepetitionsMixin, DatasetReference):
    pass


@edataclass
class RecipeBlendEpochized(
    SubsetRatioMixin,
    ShuffleOverEpochsMultiplierMixin,
    TagsMixin,
    DatasetLoaderInterface,
):
    """Blending of datasets, by specifying the number of repetitions for samples from the inner
    datasets. Ensures that the constraint, that samples are seen exactly this many times before
    repeating the "epoch" (i.e. one epoch contains the total number of repetitions for each inner
    dataset)."""

    blend_epochized: List[
        Union[
            BlendEpochizedDatasetReference,
            "RecipeBlendEpochized",
        ]
    ]

    def post_initialize(self, recipe_path: Optional[EPath] = None):
        assert recipe_path is not None
        for dataset in self.blend_epochized:
            dataset.post_initialize(recipe_path)

    def traverse(
        self,
        recipe_path: Optional[EPath] = None,
        *,
        split_part: Union[Literal["train", "val", "test"], str],
        _shuffle_over_epochs_multiplier: Optional[int] = 1,
        _tags: Optional[Dict[str, Any]] = None,
    ) -> List[TraversedDatasetReference]:
        assert recipe_path is not None
        flattened: List[TraversedDatasetReference] = []
        _shuffle_over_epochs_multiplier = self._merge_shuffle_over_epochs_multiplier(
            _shuffle_over_epochs_multiplier
        )
        _tags = self._merge_tags(_tags)
        for dataset in self.blend_epochized:
            flattened.extend(
                dataset.traverse(
                    recipe_path,
                    split_part=split_part,
                    _shuffle_over_epochs_multiplier=_shuffle_over_epochs_multiplier,
                    _tags=_tags,
                )
            )
        return flattened

    def prepare(self, split_part: Optional[str] = None) -> Sequence[EPath]:
        files = []
        for dataset in self.blend_epochized:
            files.extend(dataset.prepare(split_part=split_part))
        return files

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        tags: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        subset: Optional[DatasetSubset] = None,
        **kwargs,
    ) -> LoadedDatasetList:
        tags = resolve_tags(tags, kwargs.pop("subflavors", None))
        subset = self._get_subset(subset)
        shuffle_over_epochs_multiplier = self._merge_shuffle_over_epochs_multiplier(
            shuffle_over_epochs_multiplier
        )
        tags = self._merge_tags(tags)
        datasets = []
        for dataset in self.blend_epochized:
            inner_result = dataset.get_datasets(
                training=training,
                split_part=split_part,
                worker_config=worker_config,
                tags=tags,
                shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
                subset=subset,
                **kwargs,
            )
            if inner_result.blend_mode not in (
                DatasetBlendMode.NONE,
                DatasetBlendMode.SAMPLE_REPETITIONS,
            ):
                raise ValueError(
                    "Can only blend datasets which are of the same blend mode. Cannot mix blend with blend_epochized."
                )
            for loaded_dataset in inner_result.datasets:
                if inner_result.blend_mode == DatasetBlendMode.SAMPLE_REPETITIONS:
                    assert isinstance(loaded_dataset.repetitions, (int, float))
                else:
                    assert loaded_dataset.weight is None
                    assert loaded_dataset.repetitions is None
                    loaded_dataset.repetitions = 1
                loaded_dataset.repetitions = dataset.repetitions * loaded_dataset.repetitions
                datasets.append(loaded_dataset)
        return LoadedDatasetList(
            blend_mode=DatasetBlendMode.SAMPLE_REPETITIONS,
            datasets=datasets,
        )


@edataclass
class Recipe(DatasetLoaderInterface):
    path: EPath
    splits: Dict[str, Union[RecipeBlend, RecipeBlendEpochized, DatasetReference]]

    def post_initialize(self, recipe_path: Optional[EPath] = None):
        assert recipe_path is None
        for split in self.splits.values():
            split.post_initialize(self.path)

    def traverse(
        self,
        recipe_path: Optional[EPath] = None,
        *,
        split_part: Union[Literal["train", "val", "test"], str],
        _shuffle_over_epochs_multiplier: Optional[int] = 1,
        _tags: Optional[Dict[str, Any]] = None,
    ) -> List[TraversedDatasetReference]:
        """Traverse the selected recipe split and flatten all reachable leaf references.

        Args:
            recipe_path: Unused for top-level recipes. Present to satisfy the shared interface.
            split_part: Split to traverse.

        Returns:
            The flattened list of traversed leaf dataset references for `split_part`.
        """
        assert recipe_path is None
        return self.splits[split_part].traverse(
            self.path,
            split_part=split_part,
            _shuffle_over_epochs_multiplier=_shuffle_over_epochs_multiplier,
            _tags=_tags,
        )

    def prepare(self, split_part: Optional[str] = None) -> Sequence[EPath]:
        # In the case of prepare for Recipe, we ignore the passed cache_path
        # and instead use the own path.
        # If someone runs energon prepare on a recipe that refers to another recipe,
        # any actions concerning the inner recipe will be done on the inner recipe's path.

        if split_part is None:
            files = []
            for split_part, split in self.splits.items():
                files.extend(split.prepare(split_part=split_part))
        else:
            files = self.splits[split_part].prepare(split_part=split_part)
        # Cleanup paths here
        cache_path = EPath(self.path.parent / f"{self.path.name}.cache")
        remove_files = set(cache_path.glob("*")) - set(files)
        for file in remove_files:
            print(f"Cleanup cache file {file}")
            file.unlink()
        # Do not return paths. This is a barrier
        return ()

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        tags: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        subset: Optional[DatasetSubset] = None,
        **kwargs,
    ) -> LoadedDatasetList:
        tags = resolve_tags(tags, kwargs.pop("subflavors", None))
        return self.splits[split_part].get_datasets(
            training=training,
            split_part=split_part,
            worker_config=worker_config,
            tags=tags,
            shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
            subset=subset,
            **kwargs,
        )


@edataclass
class MetadatasetV2(Recipe):
    def __post_init__(self):
        warn_deprecated(
            "MetadatasetV2 is deprecated and will be removed in a future release. Use Recipe instead.",
            stacklevel=3,
        )
