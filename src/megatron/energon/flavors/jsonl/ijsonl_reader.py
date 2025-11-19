# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from typing import (
    Callable,
    Generator,
    Optional,
    Tuple,
    TypeVar,
)

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.jsonl.ijsonl import (
    CachedIJsonlOffsetReader,
    IJsonlFile,
    IJsonlIndexReader,
    IJsonlSamplePointer,
)
from megatron.energon.flavors.webdataset.structs import FilteredSample
from megatron.energon.source_info import SourceInfo

T_index = TypeVar("T_index", covariant=False)


class IJsonlReader(ABC):
    """
    Class for reading indexed jsonl files containing json samples.

    The common usage patterns and random-access interfaces are provided here.

    Args:
        base_path: The path to the dataset.
        jsonl_path: The path to the jsonl file.
        jsonl_filename: The jsonl file name.
        sample_filter: An optional filter function to select samples by their key.
        index_cache_size: The size of the index cache.
    """

    jsonl_path: EPath
    sample_filter: Optional[Callable[[str], bool]]

    cached_offset_reader: CachedIJsonlOffsetReader
    ijsonl_file: IJsonlFile | None = None

    def __init__(
        self,
        jsonl_path: EPath,
        sample_filter: Optional[Callable[[str], bool]] = None,
        index_cache_size: int = 5,
    ):
        self.jsonl_path = jsonl_path
        self.sample_filter = sample_filter
        self.cached_offset_reader = CachedIJsonlOffsetReader(
            jsonl_path, cache_size=index_cache_size
        )

    def __len__(self) -> int:
        return len(self.cached_offset_reader)

    def __str__(self) -> str:
        return f"IJsonlReader(jsonl_path={self.jsonl_path})"

    def _get_item_by_sample_pointer(
        self,
        sample_pointer: IJsonlSamplePointer,
    ) -> FilteredSample | None:
        """
        Get a sample from the dataset or slice it.

        Args:
            sample_pointer: The sample pointer to get the sample from.
            sample_index: The global index of the sample in the dataset.

        Returns:
            The sample or None if the sample is invalid.
        """

        key = str(sample_pointer.index)
        if self.sample_filter is not None and not self.sample_filter(key):
            return None

        if self.ijsonl_file is None:
            self.ijsonl_file = IJsonlFile(self.jsonl_path.open("rb"))

        json_data = self.ijsonl_file.next(sample_pointer.byte_offset, sample_pointer.byte_size)
        if json_data is None:
            return None

        return FilteredSample(
            __key__=key,
            __shard__=self.jsonl_path.name,
            __restore_key__=("Webdataset", sample_pointer.index),
            __sources__=(
                SourceInfo(
                    dataset_path=str(self.jsonl_path),
                    index=sample_pointer.index,
                    shard_name=self.jsonl_path.name,
                    file_names=(f"{key}.json",),
                ),
            ),
            json=json_data,
        )

    def __getitem__(self, idx: int | str) -> FilteredSample | tuple[bytes, SourceInfo] | None:
        """
        Get a sample from the dataset.
        """

        assert isinstance(idx, (int, str)), f"Invalid argument type for __getitem__: {type(idx)}"
        full_entry_name = False
        if isinstance(idx, str):
            if idx.endswith(".json"):
                num_idx = idx.removesuffix(".json")
                full_entry_name = True
            try:
                idx = int(num_idx)
            except ValueError:
                raise ValueError(f"Invalid JSONL sample key: {idx}")

        byte_offset, byte_size = self.cached_offset_reader.get_ijsonl_byte_offset(idx)
        sample: FilteredSample | None = self._get_item_by_sample_pointer(
            IJsonlSamplePointer(
                index=idx,
                byte_offset=byte_offset,
                byte_size=byte_size,
            )
        )

        if sample is None:
            return None

        if full_entry_name:
            assert len(sample["__sources__"]) == 1
            return sample["json"], sample["__sources__"][0]
        else:
            return sample

    def list_all_samples(self) -> Generator[Tuple[str, int, int], None, None]:
        """List all samples in the jsonl file.

        Returns:
            A generator of tuples of (sample_key, size, tar_file_id)
        """
        last_byte_offset = 0
        with IJsonlIndexReader(self.jsonl_path) as ijsonl_index_reader:
            for sample_idx, byte_offset in enumerate(ijsonl_index_reader):
                if last_byte_offset == byte_offset:
                    continue
                yield str(sample_idx), byte_offset - last_byte_offset, 0
                last_byte_offset = byte_offset

    def list_all_sample_parts(self) -> Generator[Tuple[str, int, int], None, None]:
        """List all sample parts in the jsonl file.

        Returns:
            A generator of tuples of (sample_key + "." + part_name, size, tar_file_id)
        """
        last_byte_offset = 0
        with IJsonlIndexReader(self.jsonl_path) as ijsonl_index_reader:
            for sample_idx, byte_offset in enumerate(ijsonl_index_reader):
                if last_byte_offset == byte_offset:
                    continue
                yield f"{sample_idx}.json", byte_offset - last_byte_offset, 0
                last_byte_offset = byte_offset

    def list_sample_parts(self, sample_key: str) -> Generator[Tuple[str, int, int], None, None]:
        """Given a sample key, list all its parts. (E.g. given 1, list 1.jpg, 1.json, etc.)

        Args:
            sample_key: The sample key to list the parts of.

        Returns:
            A generator of tuples of (part_name, size, tar_file_id)
        """
        try:
            sample_idx = int(sample_key)
        except ValueError:
            raise ValueError(f"Invalid JSONL sample key: {sample_key}")

        _, byte_size = self.cached_offset_reader.get_ijsonl_byte_offset(sample_idx)
        yield f"{sample_key}.json", byte_size, 0

    def get_total_size(self) -> int:
        return self.cached_offset_reader.get_total_size()

    def close(self):
        if self.ijsonl_file is not None:
            self.ijsonl_file.close()
        self.cached_offset_reader.close()
