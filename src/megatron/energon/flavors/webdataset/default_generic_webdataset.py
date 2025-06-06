# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.flavors.webdataset.base_webdataset import BaseWebdatasetFactory
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.field_access import field_access, split_field_access
from megatron.energon.flavors.webdataset.structs import FilteredSample
from megatron.energon.module_loader import ModuleLoader

T_sample = TypeVar("T_sample", covariant=True)


class DefaultGenericWebdatasetFactory(BaseWebdatasetFactory[T_sample], Generic[T_sample]):
    """
    Default implementation of webdataset for generic samples and the generic config interface for use with dataset.yaml.
    """

    _sample_loader: Callable[[Dict[str, Any]], Dict[str, Any]]

    def __init__(
        self,
        path: EPath,
        *,
        subflavors: Optional[Dict[str, Any]] = None,
        field_map: Optional[Dict[str, str]] = None,
        sample_loader: Optional[Union[str, Callable[[dict], dict]]] = None,
        part_filter: Optional[Union[str, List[str], Callable[[str], bool]]] = None,
        **kwargs,
    ):
        """
        Factory for the webdataset sample loader and basic configuration options.

        Args:
            subflavors: Subflavors dictionary to set for all loaded samples.
            field_map: Mapping from the webdataset fields to the sample fields.
            sample_loader: Function to load the sample from the webdataset fields. May be a string
                in order to load a function from a module, or a callable directly.
            part_filter: Filter for the parts to load. May be a string in order to load a function
                from a module, or a callable directly.
            **kwargs: Args passed to parent constructor.
        """
        assert (field_map is None) != (sample_loader is None), (
            "Either field_map or sample_loader must be provided."
        )
        if sample_loader is not None:
            assert part_filter is not None, (
                "part_filter must be provided if sample_loader is provided."
            )
            module_loader = ModuleLoader()
            if isinstance(sample_loader, str):
                sample_loader = module_loader.get_function(
                    sample_loader, "sample_loader", relative_path=path / MAIN_FOLDER_NAME
                )
            else:
                assert callable(sample_loader)
                sample_loader = sample_loader
            if isinstance(part_filter, list):
                parts = set(part_filter)
                part_filter = lambda part: part in parts
            elif isinstance(part_filter, str):
                part_filter = module_loader.get_function(
                    part_filter, "part_filter", relative_path=path / MAIN_FOLDER_NAME
                )
            else:
                assert callable(part_filter)
            self._sample_loader = sample_loader
        else:
            assert field_map is not None
            assert part_filter is None
            # Split field map fields by json[field][field]
            fields = {key: split_field_access(field) for key, field in field_map.items()}
            assert set(field.name for field in dataclasses.fields(self.__sample_type__)).issuperset(
                fields.keys()
            ) and set(
                field.name
                for field in dataclasses.fields(self.__sample_type__)
                if field.default is not dataclasses.MISSING
                and field.default_factory is not dataclasses.MISSING
            ).issubset(field_map.keys()), (
                f"field_map does not map to type {self.__sample_type__.__name__} fields"
            )
            self._sample_loader = lambda sample: {
                k: field_access(sample, v) for k, v in fields.items()
            }
            parts = set(access[0] for options in fields.values() for access in options)
            part_filter = lambda part: part in parts
        inner_sample_loader = self._sample_loader
        self._sample_loader = lambda sample: {
            "__key__": sample["__key__"],
            **inner_sample_loader(sample),
            "__restore_key__": sample["__restore_key__"],
            "__subflavors__": self.subflavors,
            "__sources__": sample["__sources__"],
        }
        super().__init__(path, **kwargs, part_filter=part_filter)
        self.subflavors = subflavors or {}

    def load_sample(self, sample: FilteredSample) -> T_sample:
        return self.__sample_type__(**self._sample_loader(sample))

    def config(self) -> Dict[str, Any]:
        return dict(
            **super().config(),
            subflavors=self.subflavors,
            sample_loader=SavableDataset._function_config(self._sample_loader),
        )
