# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import traceback
from typing import Optional

from megatron.energon.source_info import SourceInfo


def log_exception(_e: Exception, sample, sources: Optional[list[SourceInfo]] = None):
    traceback.print_exc()
    print("-" * 10)

    if sources:
        print("Sources:")
        for source in sources:
            print(
                f" - {source.dataset_path}[{source.index}] {source.shard_name}{source.file_names!r}"
            )
        print("-" * 10)

    sample_str = str(sample)
    if len(sample_str) > 400:
        sample_str = sample_str[:200] + "..." + sample_str[-200:]

    print(sample_str)

    print("-" * 10)
