# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import random
from typing import Any, List, Mapping, Optional, Sequence, TypeVar

import numpy
import torch
import torch.distributed
import torch.utils.data

from megatron.energon.edataclass import edataclass
from megatron.energon.savable import FlexState, Savable
from megatron.energon.worker import WorkerConfig

T = TypeVar("T")


class WorkerRng(Savable):
    """Helper class for getting a worker random generator, which is still in itself deterministic.
    If not in a worker, uses the global random generator's seed to initialize a new rng."""

    worker_config: WorkerConfig

    _rng: Optional[torch.Generator] = None

    _restore_state: Optional[bytes] = None

    def __init__(self, worker_config: WorkerConfig):
        self.worker_config = worker_config

    @property
    def rng(self) -> torch.Generator:
        if self._rng is None or self._restore_state is not None:
            self.worker_config.assert_worker()
            self._rng = torch.Generator()
            if self._restore_state is not None:
                self._rng.set_state(
                    torch.frombuffer(
                        bytearray(self._restore_state),
                        dtype=torch.uint8,
                    ).clone()
                )
            else:
                # Restore to initial state (either due to zero sized states, or just initial state)
                self._rng.manual_seed(self.worker_config.worker_seed())
            self._restore_state = None
        return self._rng

    def randbelow(self, n: int) -> int:
        return torch.randint(0, n, (), generator=self.rng).item()

    def choice_idx(self, probs: torch.Tensor) -> int:
        if len(probs) == 1:
            return 0
        else:
            # Custom implementation of multinomial to ensure consistency
            # Torch changed their implementation of torch.multinomial in 2.7.0 and to be
            # consistent with any torch version, we use a custom implementation here instead.
            # This is anyways just a very simple case of multinomial, thus this should be fine.
            # Actually, benchmarks show that this is faster than torch.multinomial by a factor of
            # 10 even on CPU.
            cdf = torch.cumsum(probs, dim=0)
            val = torch.rand(1, generator=self.rng) * cdf[-1]
            return torch.searchsorted(cdf, val).item()

    def choice(self, l: List[T], probs: Optional[torch.Tensor] = None) -> T:
        if probs is None:
            return l[self.randbelow(len(l))]
        assert len(l) == len(probs)
        return l[self.choice_idx(probs)]

    def shuffle(self, l: List[T]) -> List[T]:
        """Returns a new list with shuffled entries"""
        p = torch.randperm(len(l), generator=self.rng)
        return [l[p[i]] for i in range(len(l))]

    def rand_pop(self, l: List[T]) -> T:
        return l.pop(self.randbelow(len(l)))

    def save_state(self) -> FlexState:
        return FlexState(rng=None if self.rng is None else bytes(self.rng.get_state().tolist()))

    def restore_state(self, state: FlexState):
        if state["rng"] is None:
            self._restore_state = None
        else:
            self._restore_state = state["rng"]


@edataclass
class SystemRngState:
    """The state of the global random generators.

    Note that the data types of the internal RNG states are implementation details of the
    respective libraries and may change in the future.

    Python does not even specify the type in their docs. Hence we will allow arbitrary types,
    because all that matters is that we can save and restore them. We will not use the data
    anywhere else.
    """

    torch: Any  # Currently `torch.Tensor`
    numpy: Any  # Currently `dict[str, Any] | tuple[str, NDArray[uint32], int, int, float]`
    random: Any  # Currently a nested tuple

    def _hashable_value(self, value: Any) -> Any:
        if isinstance(value, (int, float, bool, str)) or value is None:
            return value
        elif isinstance(value, torch.Tensor):
            return self._hashable_value(value.tolist())
        elif isinstance(value, numpy.ndarray):
            return self._hashable_value(value.tolist())
        elif isinstance(value, Mapping):
            return tuple(
                (self._hashable_value(k), self._hashable_value(v)) for k, v in value.items()
            )
        elif isinstance(value, Sequence):
            return tuple(self._hashable_value(v) for v in value)
        else:
            raise ValueError(f"Cannot hash value of type {type(value)}: {value!r}")

    def __repr__(self):
        # If the hash is the same, the state is the same. Should suffice to identify the state.
        return f"SystemRngState(hash={hash(self._hashable_value((self.torch, self.numpy, self.random)))})"


class SystemRng:
    """A class to seed, save or restore the global random generators.
    This affects torch, numpy and the standard library random module."""

    @staticmethod
    def seed(seed: int) -> None:
        """Seeds the global random generators."""
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def save_state() -> SystemRngState:
        """Saves the global rng state for torch, numpy and random."""
        return SystemRngState(
            torch=torch.get_rng_state(),
            numpy=numpy.random.get_state(),
            random=random.getstate(),
        )

    @staticmethod
    def restore_state(state: SystemRngState) -> None:
        """Restores the global rng state for torch, numpy and random."""
        torch.set_rng_state(state.torch)
        numpy.random.set_state(state.numpy)
        random.setstate(state.random)

    @staticmethod
    def get_seed_from_args(*args: Any) -> int:
        """Deterministically generates a seed from the given arguments.
        The str() representation of each arg is used."""

        # Use a deterministic hash function to compute the seed
        hash_digest = hashlib.sha1("|".join([str(obj) for obj in args]).encode("utf-8")).digest()

        # We use the first 4 bytes of the hash as the seed and fix the endianness
        seed_value = int.from_bytes(hash_digest[:4], byteorder="big")

        return seed_value

    @staticmethod
    def seed_args(*args: Any) -> None:
        """Seeds the global random generators deterministically from the given arguments."""
        SystemRng.seed(SystemRng.get_seed_from_args(*args))
