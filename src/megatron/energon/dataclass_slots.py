# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

# from dataclasses import dataclass
import sys
from typing import Callable, TypeVar, overload

from typing_extensions import dataclass_transform

if sys.version_info >= (3, 10):
    from dataclasses import dataclass
else:
    # Fallback for 3.9 and below
    from dataslots import dataclass

T = TypeVar("T", bound=type)


@overload
def dataclass_slots(cls: T) -> T: ...


@overload
def dataclass_slots(**kwargs) -> Callable[[T], T]: ...


@dataclass_transform(slots_default=True)
def dataclass_slots(cls=None, **kwargs):
    """
    A decorator that is a combination of `dataclass` and `__slots__` for python <3.10.
    As soon as we bump the minimum python version to 3.10, we can remove this function and set
    slots=True.

    Kwargs are the same as for dataclass if passed.
    """
    if cls is None:

        def wrap(cls):
            new_cls = dataclass(cls, **kwargs, slots=True)
            return new_cls

        return wrap

    assert not kwargs
    new_cls = dataclass(cls, **kwargs, slots=True)
    return new_cls
