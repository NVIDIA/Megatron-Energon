# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from abc import abstractmethod
from typing import Any, Callable, Generic, TypeVar

R = TypeVar("R", covariant=True)
T = TypeVar("T", covariant=True)


class Future(Generic[R]):
    """Base class for abstract futures."""

    @abstractmethod
    def get(self) -> R: ...


class DoneFuture(Future[R]):
    """Future that is already done."""

    def __init__(self, result: R):
        self._result = result

    def get(self) -> R:
        return self._result


class CallableFuture(Future[R]):
    """Future that calls a callable to get the result."""

    _callable: Callable[[], R]
    _value: R
    _exception: Exception

    def __init__(self, callable: Callable[[], R]):
        self._callable = callable

    def get(self) -> R:
        if not hasattr(self, "_value") and not hasattr(self, "_exception"):
            try:
                self._value = self._callable()
            except Exception as e:
                self._exception = e
        if hasattr(self, "_exception"):
            raise self._exception
        return self._value

    @staticmethod
    def chain(future: Future[T], fn: Callable[[Future[T]], R]) -> Future[R]:
        """
        Chain a function to a future.

        Args:
            future: The future which provides the input for the function.
            fn: The function to call on the result of the future, to transform the result.

        Returns:
            A future that will be resolved to the result of the function given the result of the future.
        """
        return CallableFuture(lambda: fn(future))


class ExceptionFuture(Future[Any]):
    """Future that raises an exception."""

    def __init__(self, exception: Exception):
        self._exception = exception

    def get(self) -> Any:
        raise self._exception
