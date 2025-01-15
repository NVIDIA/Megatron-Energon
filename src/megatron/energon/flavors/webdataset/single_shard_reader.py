# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import tarfile
import warnings
from typing import Callable, Iterator, Optional, Set, Tuple

from megatron.energon.errors import SYSTEM_EXCEPTIONS, FatalSampleError
from megatron.energon.flavors.webdataset.config import skip_meta_re, split_name_re
from megatron.energon.flavors.webdataset.itar import get_itar_byte_offset, open_itar
from megatron.energon.flavors.webdataset.structs import FilteredSample, ShardState
from megatron.energon.worker import WorkerConfig


class SingleShardReader:
    """
    Reads a single shard, possibly resuming from a saved state. The reader keeps track of the
    current offset and byte offset in the shard and yields one sample at a time calling the
    `read_next` function. Optionally reads sample columns from extra column shards.
    """

    # Internal empty iterator, will always raise StopIteration on next
    _EMPTY_ITER = iter(())

    #: The data parallel worker config
    worker_config: WorkerConfig
    # Sample keys to ignore
    exclude: Set[str]
    # State of the shard to load
    shard_state: ShardState
    # Filter out wds files.
    part_filter: Optional[Callable[[str], bool]]
    # Error handler
    handler: Callable[[Exception, Optional[str]], None]

    # The internal state variables
    _ctx: contextlib.ExitStack
    _tar_file: tarfile.TarFile
    _stream: Iterator[tarfile.TarInfo] = _EMPTY_ITER
    _cur_tarinfo: Optional[tarfile.TarInfo] = None
    _cur_base_name: Optional[str] = None
    _cur_ext: Optional[str] = None
    _cur_key: Optional[str] = None

    def __init__(
        self,
        worker_config: WorkerConfig,
        exclude: Set[str],
        shard_state: ShardState,
        part_filter: Optional[Callable[[str], bool]],
        handler: Callable[[Exception, Optional[str]], None],
    ):
        self.worker_config = worker_config
        self.exclude = exclude
        self.shard_state = shard_state
        self.part_filter = part_filter
        self.handler = handler

        if shard_state.offset == shard_state.shard.count:
            # Empty shard, return immediately
            self._ctx = contextlib.ExitStack()
            return

        shard = shard_state.shard
        if shard.byte_offset is None:
            shard.byte_offset = get_itar_byte_offset(shard.path, shard.offset)
        if shard.byte_size is None:
            shard.byte_size = (
                get_itar_byte_offset(shard.path, shard.offset + shard.count) - shard.byte_offset
            )

        # If the shard is not empty, the absolute byte offset must be smaller than the shard size
        assert self.shard_state.byte_offset <= shard.byte_size

        # Given the shard offset (e.g. sub-shard) and the relative byte offset from the stored state, compute the absolute byte offset
        self._absolute_offset = shard.offset + self.shard_state.offset
        self._absolute_byte_offset = shard.byte_offset + self.shard_state.byte_offset
        self._sub_tar_byte_size = shard.byte_size - self.shard_state.byte_offset
        self._orig_shard_state_byte_offset = self.shard_state.byte_offset

        if worker_config.should_log(level=2):
            worker_config.worker_log(
                {
                    "t": "WebdatasetSampleLoaderDataset._shard_iter",
                    "r": worker_config.rank,
                    "w": worker_config.rank_worker_id(),
                    "shard": {
                        "name": shard_state.shard.name,
                        "path": str(shard_state.shard.path),
                        "offset": shard_state.shard.offset,
                        "count": shard_state.shard.count,
                    },
                    "offset": shard_state.offset,
                }
            )

        self._ctx = contextlib.ExitStack()
        if self.shard_state.byte_offset < shard.byte_size:
            # Non-empty shard (empty cannot be handled by the open_itar function)
            self._tar_file = self._ctx.enter_context(
                open_itar(
                    shard.path,
                    byte_offset=self._absolute_byte_offset,
                    byte_size=self._sub_tar_byte_size,
                )
            )
            self._stream = iter(self._tar_file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._stream and hasattr(self._stream, "close"):
            self._stream.close()
        self._ctx.close()

    def _next_tarinfo(self) -> None:
        """Internally loads the nest tar entry metadata. Or raises StopIteration."""
        while True:
            try:
                self._cur_tarinfo = next(self._stream)
            except StopIteration:
                self._cur_base_name = self._cur_ext = self._cur_key = self._cur_tarinfo = None
                raise
            fname = self._cur_tarinfo.name
            if not self._cur_tarinfo.isreg():
                continue
            if fname is None:
                continue
            if skip_meta_re.match(fname):
                continue

            # Get base_name and extension if available
            m = split_name_re.match(fname)
            if not m:
                continue
            self._cur_base_name, self._cur_ext = m.groups()

            self._cur_key = f"{self.shard_state.shard.name}/{self._cur_base_name}"
            if self._cur_key in self.exclude:
                continue
            return

    def read_next(
        self, for_group_name: Optional[str] = None, must_match: bool = False
    ) -> Tuple[str, FilteredSample]:
        """Gets the next sample, or raises StopIteration if exhausted."""
        if self._cur_tarinfo is None:
            self._next_tarinfo()
        assert self._cur_tarinfo is not None
        assert self._cur_key is not None

        if for_group_name is None:
            group_name = self._cur_base_name
        else:
            group_name = for_group_name
            assert (
                not must_match or group_name == self._cur_base_name
            ), f"Sample key mismatch: {group_name} != {self._cur_base_name}"
        group: FilteredSample = dict(
            __key__=self._cur_key,
            __shard__=self.shard_state.shard.name,
            __restore_key__=(
                "Webdataset",
                self.shard_state.shard.name,
                self.shard_state.offset + self.shard_state.shard.offset,
            ),
        )
        while True:
            try:
                if group_name != self._cur_base_name:
                    # Either the group has content, it it's only for a specific key
                    next_sample_offset_in_sub_tar = self._cur_tarinfo.offset
                    # NOTE: The next_sample_offset_in_sub_tar (tarinfo.offset) is relative to
                    # absolute_byte_offset, since open_itar crops a part out of the file.
                    # But we want to compute the offset relative to shard.byte_offset
                    self.shard_state.byte_offset = (
                        next_sample_offset_in_sub_tar + self._orig_shard_state_byte_offset
                    )
                    assert self.shard_state.byte_offset <= self.shard_state.shard.byte_size
                    self.shard_state.offset += 1

                    if self.worker_config.should_log(level=3):
                        self.worker_config.worker_log(
                            {
                                "t": "WebdatasetSampleLoaderDataset._shard_iter.yield",
                                "r": self.worker_config.rank,
                                "w": self.worker_config.rank_worker_id(),
                                "shard": {
                                    "name": self.shard_state.shard.name,
                                    "path": str(self.shard_state.shard.path),
                                    "offset": self.shard_state.shard.offset,
                                    "count": self.shard_state.shard.count,
                                },
                                "offset": self.shard_state.offset,
                                "group_name": group_name,
                            }
                        )

                    return group_name, group

                if self.part_filter is None or self.part_filter(self._cur_ext):
                    group[self._cur_ext] = self._tar_file.extractfile(self._cur_tarinfo).read()
            except SYSTEM_EXCEPTIONS:
                raise FatalSampleError.from_sample_key(self._cur_key)
            except Exception as e:
                self.handler(e, self._cur_key)

            try:
                self._next_tarinfo()
                assert self._cur_tarinfo is not None
            except StopIteration:
                break

        # shard_state.byte_offset = (absolute_tar_begin_offset - shard_info.byte_offset)
        # Next state
        next_sample_offset_in_sub_tar = self.shard_state.shard.byte_size - (
            self._absolute_byte_offset - self.shard_state.shard.byte_offset
        )
        # NOTE: The next_sample_offset_in_sub_tar (tarinfo.offset) is relative to
        # absolute_byte_offset, since open_itar crops a part out of the file.
        # But we want to compute the offset relative to shard.byte_offset
        self.shard_state.byte_offset = (
            next_sample_offset_in_sub_tar + self._orig_shard_state_byte_offset
        )
        assert self.shard_state.byte_offset <= self.shard_state.shard.byte_size
        self.shard_state.offset += 1

        if self.worker_config.should_log(level=3):
            self.worker_config.worker_log(
                {
                    "t": "WebdatasetSampleLoaderDataset._shard_iter.yield",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "shard": {
                        "name": self.shard_state.shard.name,
                        "path": str(self.shard_state.shard.path),
                        "offset": self.shard_state.shard.offset,
                        "count": self.shard_state.shard.count,
                    },
                    "offset": self.shard_state.offset,
                    "group_name": group_name,
                }
            )

        if (
            self.shard_state.offset != self.shard_state.shard.count
            or self.shard_state.byte_offset != self.shard_state.shard.byte_size
        ):
            warnings.warn(
                f"shard_state.offset({self.shard_state.offset}) != shard.count({self.shard_state.shard.count}) or "
                f"shard_state.byte_offset({self.shard_state.byte_offset}) != shard.byte_size({self.shard_state.shard.byte_size})"
                f"; this indicates an internal bug. Shard might not have been iterated completely, samples may be missing."
            )

        return group_name, group
