# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from megatron.energon.flavors.base_dataset import Sample


@dataclass
class Cooker:
    """A cooker transforms a crude sample (simple dict) into a
    specific sample type inheriting from `Sample`.

    The `cook` method performs the transformation,
    the other fields are used to select the samples which this cooker
    can transform. If no filters are provided, the cooker will
    transform any sample.
    """

    cook: Callable[[dict], Sample]

    # If multiple of the following conditions are provided
    # then the sample must satisfy all of them.
    has_subflavors: Optional[dict] = None
    condition: Optional[Callable[[dict], bool]] = None

    def is_match(self, subflavors: Dict[str, Any]) -> bool:
        if self.has_subflavors is not None:
            # Checks if the dict entries provided as a filter all match
            # the ones in the sample. The sample may have additional entries.
            for k, v in self.has_subflavors.items():
                if k not in subflavors or subflavors[k] != v:
                    return False

        if self.condition is not None:
            if not self.condition(subflavors):
                return False

        return True


def basic_sample_keys(crude_sample: dict) -> dict:
    """A convenience helper to extract the basic keys from a crude sample,
    which you will always need to forward to the cooked sample."""

    return {
        field.name: crude_sample[field.name]
        for field in dataclasses.fields(Sample)
        if field.name in crude_sample
    }
