# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod

from megatron.energon.state import FlexState


class Savable(ABC):
    @abstractmethod
    def save_state(self) -> FlexState:
        """Dumps the state of the object into some serializable format.
        Should only dump the part that is persistable for energon usage.
        """
        ...

    @abstractmethod
    def restore_state(self, state: FlexState):
        """Restores the state of the object.
        If state is None, the object is restored to its default state."""
        ...
