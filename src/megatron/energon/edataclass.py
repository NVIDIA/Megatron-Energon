# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from functools import partial

# We define an alias for `@dataclass(slots=True, kw_only=True)`,
# because we want to use this combination almost everywhere.
edataclass = partial(dataclass, slots=True, kw_only=True)
