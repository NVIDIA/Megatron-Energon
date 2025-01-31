from dataclasses import dataclass
from typing import Callable, TypeVar, dataclass_transform, overload

from typing_extensions import dataclass_transform

T = TypeVar("T", bound=type)


@overload
def dataclass_slots(cls: T) -> T: ...


@overload
def dataclass_slots(**kwargs) -> Callable[[T], T]: ...


@dataclass_transform(eq_default=True, order_default=False, kw_only_default=False)
def dataclass_slots(cls=None, **kwargs):
    """
    A decorator that is a combination of `dataclass` and `__slots__` for python <3.10.
    As soon as we bump the minimum python version to 3.10, we can remove this function and set
    slots=True.

    Kwargs are the same as for dataclass if passed.
    """
    if cls is None:
        return lambda cls: dataclass(cls, **kwargs)
    assert not kwargs
    return dataclass(cls)
