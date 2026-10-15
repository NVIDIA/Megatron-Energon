# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import importlib.abc
import importlib.machinery
import sys
import types
import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class DeprecatedSymbol:
    target_module: str
    target_name: str


DeprecatedModuleMap = dict[str, dict[str, DeprecatedSymbol]]


DEPRECATED_MODULES: DeprecatedModuleMap = {
    "megatron.energon.flavors.base_shard_list_dataset": {
        "DEBUG_SHARD_PRINT": DeprecatedSymbol(
            "megatron.energon.flavors.base_indexed_dataset", "DEBUG_SHARD_PRINT"
        ),
        "BaseIndexedDatasetFactory": DeprecatedSymbol(
            "megatron.energon.flavors.base_indexed_dataset", "BaseIndexedDatasetFactory"
        ),
        "BaseManifestShardListDatasetFactory": DeprecatedSymbol(
            "megatron.energon.flavors.base_manifest_dataset",
            "BaseManifestShardListDatasetFactory",
        ),
        "BaseShardListDatasetFactory": DeprecatedSymbol(
            "megatron.energon.flavors.base_indexed_dataset", "BaseIndexedDatasetFactory"
        ),
        "ShardListMeta": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.io", "ShardListMeta"
        ),
        "_print_shard_slices": DeprecatedSymbol(
            "megatron.energon.flavors.base_indexed_dataset", "_print_shard_slices"
        ),
    },
    "megatron.energon.flavors.filter_index": {
        "FILTER_FILENAME_PREFIX": DeprecatedSymbol(
            "megatron.energon.flavors.common.filter_index", "FILTER_FILENAME_PREFIX"
        ),
        "FILTER_INDEX_DTYPE": DeprecatedSymbol(
            "megatron.energon.flavors.common.filter_index", "FILTER_INDEX_DTYPE"
        ),
        "FILTER_INDEX_SUFFIX": DeprecatedSymbol(
            "megatron.energon.flavors.common.filter_index", "FILTER_INDEX_SUFFIX"
        ),
        "FILTER_INDEX_VERSION": DeprecatedSymbol(
            "megatron.energon.flavors.common.filter_index", "FILTER_INDEX_VERSION"
        ),
        "FILTER_JSON_SUFFIX": DeprecatedSymbol(
            "megatron.energon.flavors.common.filter_index", "FILTER_JSON_SUFFIX"
        ),
        "FilterIndex": DeprecatedSymbol(
            "megatron.energon.flavors.common.filter_index", "FilterIndex"
        ),
        "FilterIndexWriter": DeprecatedSymbol(
            "megatron.energon.flavors.common.filter_index", "FilterIndexWriter"
        ),
        "FilterMetadata": DeprecatedSymbol(
            "megatron.energon.flavors.common.filter_index", "FilterMetadata"
        ),
        "TranslatedIndexReader": DeprecatedSymbol(
            "megatron.energon.flavors.common.filter_index", "TranslatedIndexReader"
        ),
        "build_filter_index": DeprecatedSymbol(
            "megatron.energon.flavors.common.filter_index", "build_filter_index"
        ),
        "build_filter_index_from_global_indexes": DeprecatedSymbol(
            "megatron.energon.flavors.common.filter_index",
            "build_filter_index_from_global_indexes",
        ),
        "build_filter_index_from_shard_indexes": DeprecatedSymbol(
            "megatron.energon.flavors.common.filter_index",
            "build_filter_index_from_shard_indexes",
        ),
        "filter_index_paths": DeprecatedSymbol(
            "megatron.energon.flavors.common.filter_index", "filter_index_paths"
        ),
    },
    "megatron.energon.flavors.manifest": {
        "build_split_parts": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.write", "build_split_parts"
        ),
        "write_manifest_dataset_metadata": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.write",
            "write_manifest_dataset_metadata",
        ),
    },
    "megatron.energon.flavors.webdataset.empty_dataset_error": {
        "EmptyDatasetError": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.empty_dataset_error",
            "EmptyDatasetError",
        ),
    },
    "megatron.energon.flavors.webdataset.metadata": {
        "EnergonDatasetType": DeprecatedSymbol(
            "megatron.energon.flavors.dataset_type", "EnergonDatasetType"
        ),
        "check_dataset_info_present": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.io", "check_dataset_info_present"
        ),
        "get_dataset_info": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.io", "get_dataset_info"
        ),
        "get_dataset_type": DeprecatedSymbol(
            "megatron.energon.flavors.dataset_type", "get_dataset_type"
        ),
        "get_info_shard_files": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.io", "get_info_shard_files"
        ),
    },
    "megatron.energon.flavors.webdataset.sample_loader": {
        "DatasetSampler": DeprecatedSymbol(
            "megatron.energon.flavors.common.dataset_sampler", "DatasetSampler"
        ),
        "RawSampleData": DeprecatedSymbol(
            "megatron.energon.flavors.common.dataset_sampler", "RawSampleData"
        ),
        "SliceState": DeprecatedSymbol(
            "megatron.energon.flavors.common.dataset_sampler", "SliceState"
        ),
        "WebdatasetSampleLoaderDataset": DeprecatedSymbol(
            "megatron.energon.flavors.common.dataset_sampler", "DatasetSampler"
        ),
    },
    "megatron.energon.flavors.webdataset.sample_decoder": {
        "AVDecoderType": DeprecatedSymbol("megatron.energon.decoders", "AVDecoderType"),
        "DEFAULT_DECODER": DeprecatedSymbol("megatron.energon.decoders", "DEFAULT_DECODER"),
        "GuessingHandlerWrapper": DeprecatedSymbol(
            "megatron.energon.decoders", "GuessingHandlerWrapper"
        ),
        "ImageDecoderType": DeprecatedSymbol("megatron.energon.decoders", "ImageDecoderType"),
        "SampleDecoder": DeprecatedSymbol("megatron.energon.decoders", "SampleDecoder"),
    },
    "megatron.energon.flavors.webdataset.sharder": {
        "Sharder": DeprecatedSymbol("megatron.energon.flavors.common.manifest.sharder", "Sharder"),
    },
    "megatron.energon.flavors.webdataset.structs": {
        "DatasetSubset": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.types", "DatasetSubset"
        ),
        "FilteredSample": DeprecatedSymbol(
            "megatron.energon.flavors.common.sample_record", "SampleRecord"
        ),
        "ManifestDatasetInfo": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.types", "ManifestDatasetInfo"
        ),
        "ManifestSplits": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.types", "ManifestSplits"
        ),
        "SampleRecord": DeprecatedSymbol(
            "megatron.energon.flavors.common.sample_record", "SampleRecord"
        ),
        "ShardInfo": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.types", "ShardInfo"
        ),
        "WebdatasetInfo": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.types", "ManifestDatasetInfo"
        ),
        "WebdatasetSplits": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.types", "ManifestSplits"
        ),
    },
    "megatron.energon.flavors.webdataset.thread_local_sqlite": {
        "ThreadLocalSqlite": DeprecatedSymbol(
            "megatron.energon.flavors.common.sqlite", "ThreadLocalSqlite"
        ),
        "ThreadLocalStorage": DeprecatedSymbol(
            "megatron.energon.flavors.common.sqlite", "ThreadLocalStorage"
        ),
    },
}


DEPRECATED_ATTRS: DeprecatedModuleMap = {
    "megatron.energon": {
        "MetadatasetV2": DeprecatedSymbol("megatron.energon.recipe.recipe", "MetadatasetV2"),
        "prepare_metadataset": DeprecatedSymbol("megatron.energon.recipe", "prepare_recipe"),
        "traverse_metadataset": DeprecatedSymbol("megatron.energon.recipe", "traverse_recipe"),
    },
    "megatron.energon.recipe": {
        "prepare_metadataset": DeprecatedSymbol("megatron.energon.recipe", "prepare_recipe"),
        "traverse_metadataset": DeprecatedSymbol("megatron.energon.recipe", "traverse_recipe"),
    },
    "megatron.energon.flavors.dataset_type": {
        "is_metadataset": DeprecatedSymbol("megatron.energon.flavors.dataset_type", "is_recipe"),
    },
    "megatron.energon.flavors.webdataset.config": {
        "INDEX_BATCH_SIZE": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.paths", "INDEX_BATCH_SIZE"
        ),
        "INDEX_SQLITE_FILENAME": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.paths", "INDEX_SQLITE_FILENAME"
        ),
        "INDEX_UUID_FILENAME": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.paths", "INDEX_UUID_FILENAME"
        ),
        "INFO_JSON_FILENAME": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.paths", "INFO_JSON_FILENAME"
        ),
        "INFO_YAML_FILENAME": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.paths", "INFO_YAML_FILENAME"
        ),
        "MAIN_FOLDER_NAME": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.paths", "MAIN_FOLDER_NAME"
        ),
        "SPLIT_YAML_FILENAME": DeprecatedSymbol(
            "megatron.energon.flavors.common.manifest.paths", "SPLIT_YAML_FILENAME"
        ),
    },
}


def _warn_deprecated(
    old_module: str, old_name: str | None, symbol: DeprecatedSymbol | None
) -> None:
    target = (
        f"{symbol.target_module}.{symbol.target_name}"
        if symbol is not None
        else "the new module path"
    )
    old = old_module if old_name is None else f"{old_module}.{old_name}"
    warnings.warn(
        f"{old} is deprecated; use {target} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _resolve(symbol: DeprecatedSymbol):
    module = importlib.import_module(symbol.target_module)
    return getattr(module, symbol.target_name)


def deprecated_getattr(module_name: str, name: str):
    symbol = DEPRECATED_ATTRS.get(module_name, {}).get(name)
    if symbol is None:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    _warn_deprecated(module_name, name, symbol)
    return _resolve(symbol)


def deprecated_dir(module_name: str, names: list[str]) -> list[str]:
    return sorted(set(names) | set(DEPRECATED_ATTRS.get(module_name, {})))


class _DeprecatedModuleLoader(importlib.abc.Loader):
    def __init__(self, module_name: str):
        self.module_name = module_name

    @property
    def symbols(self) -> dict[str, DeprecatedSymbol]:
        return DEPRECATED_MODULES[self.module_name]

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__loader__ = self
        module.__package__ = spec.name.rpartition(".")[0]
        module.__all__ = list(self.symbols)
        module.__deprecated_module__ = True

        def __getattr__(name: str):
            symbol = self.symbols.get(name)
            if symbol is None:
                raise AttributeError(f"module {spec.name!r} has no attribute {name!r}")
            _warn_deprecated(spec.name, name, symbol)
            value = _resolve(symbol)
            setattr(module, name, value)
            return value

        def __dir__():
            return sorted(set(module.__dict__) | set(self.symbols))

        module.__getattr__ = __getattr__
        module.__dir__ = __dir__
        return module

    def exec_module(self, module) -> None:
        _warn_deprecated(module.__name__, None, None)


class _DeprecatedModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        if fullname not in DEPRECATED_MODULES:
            return None
        return importlib.machinery.ModuleSpec(
            fullname,
            _DeprecatedModuleLoader(fullname),
            origin="deprecated",
        )


def install_deprecated_imports() -> None:
    if any(isinstance(finder, _DeprecatedModuleFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _DeprecatedModuleFinder())
