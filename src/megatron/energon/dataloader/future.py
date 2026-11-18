# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from abc import abstractmethod
from typing import Any, Callable, Generic, TypeVar

R = TypeVar("R", covariant=True)
T = TypeVar("T", covariant=True)


class CancelledError(Exception):
    """Exception raised when a future was cancelled."""

    @classmethod
    def with_current_traceback(cls):
        try:
            raise cls()
        except cls as e:
            if e.__traceback__ is not None and e.__traceback__.tb_next is not None:
                return e.with_traceback(e.__traceback__.tb_next)
            return e


class Future(Generic[R]):
    """Base class for abstract futures."""

    @abstractmethod
    def get(self) -> R:
        """Get the result of the future. Waits until the future is done."""
        ...

    @abstractmethod
    def cancel(self) -> bool:
        """Cancel the future.

        Returns:
            True if the future was cancelled, False if already done.
        """
        ...


class DoneFuture(Future[R]):
    """Future that is already done."""

    def __init__(self, result: R):
        self._result = result

    def get(self) -> R:
        return self._result

    def cancel(self) -> bool:
        return False


class CallableFuture(Future[R]):
    """Future that calls a callable to get the result."""

    __slots__ = ("_callable", "_value", "_exception", "_cancelled")

    _callable: Callable[[], R]
    _value: R
    _exception: Exception
    _cancelled: bool

    def __init__(self, callable: Callable[[], R]):
        self._callable = callable

    def get(self) -> R:
        if getattr(self, "_cancelled", False):
            raise CancelledError("Future was cancelled")
        if not hasattr(self, "_value") and not hasattr(self, "_exception"):
            try:
                self._value = self._callable()
            except Exception as e:
                self._exception = e
        if hasattr(self, "_exception"):
            raise self._exception
        return self._value

    def cancel(self) -> bool:
        if getattr(self, "_cancelled", False):
            return True
        if hasattr(self, "_value") or hasattr(self, "_exception"):
            return False
        self._cancelled = True
        return True

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

    def cancel(self) -> bool:
        return False
