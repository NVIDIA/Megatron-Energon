# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import struct
from typing import Tuple

_SAMPLE_VALUE_FMT = ">iqqq"
_SAMPLE_VALUE_SIZE = struct.calcsize(_SAMPLE_VALUE_FMT)

_PART_VALUE_FMT = ">qq"
_PART_VALUE_SIZE = struct.calcsize(_PART_VALUE_FMT)

_LOC_KEY_FMT = ">ii"
_LOC_KEY_SIZE = struct.calcsize(_LOC_KEY_FMT)


def pack_sample_value(
    *,
    tar_file_id: int,
    sample_index: int,
    byte_offset: int,
    byte_size: int,
) -> bytes:
    return struct.pack(
        _SAMPLE_VALUE_FMT,
        tar_file_id,
        sample_index,
        byte_offset,
        byte_size,
    )


def unpack_sample_value(data: bytes) -> Tuple[int, int, int, int]:
    return struct.unpack(_SAMPLE_VALUE_FMT, data)


def pack_part_key(tar_file_id: int, sample_index: int, part_name: str) -> bytes:
    return struct.pack(_LOC_KEY_FMT, tar_file_id, sample_index) + part_name.encode("utf-8")


def pack_loc_key(tar_file_id: int, sample_index: int) -> bytes:
    return struct.pack(_LOC_KEY_FMT, tar_file_id, sample_index)


def pack_part_value(content_byte_offset: int, content_byte_size: int) -> bytes:
    return struct.pack(_PART_VALUE_FMT, content_byte_offset, content_byte_size)


def unpack_part_value(data: bytes) -> Tuple[int, int]:
    return struct.unpack(_PART_VALUE_FMT, data)


def pack_media_metadata_value(metadata_type: str, metadata_json: str) -> bytes:
    type_bytes = metadata_type.encode("utf-8")
    json_bytes = metadata_json.encode("utf-8")
    return struct.pack(">II", len(type_bytes), len(json_bytes)) + type_bytes + json_bytes


def unpack_media_metadata_value(data: bytes) -> Tuple[str, str]:
    type_len, json_len = struct.unpack(">II", data[:8])
    offset = 8
    metadata_type = data[offset : offset + type_len].decode("utf-8")
    offset += type_len
    metadata_json = data[offset : offset + json_len].decode("utf-8")
    return metadata_type, metadata_json
