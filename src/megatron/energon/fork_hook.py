# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import os
import weakref
from dataclasses import dataclass
from typing import Callable


def _cleanup(hooks, key, wr):
    hooks.pop(key)


class WeakCallbacks:
    """
    A class that manages weak references to callback functions.
    """

    # A dictionary of weak (or strong) references to functions.
    _hooks: dict[int, Callable[[], Callable[..., None] | None]]

    def __init__(self):
        """
        Initialize the registry.
        """
        self._hooks: dict[int, Callable[[], Callable[..., None] | None]] = {}

    def add_hook(self, callable: Callable[..., None], make_persistent: bool = False) -> None:
        """
        Add a callback to the registry.

        Args:
            callable: The function to run before the fork of a worker process.
            make_persistent: If True, the function will be stored as a strong reference, otherwise a weak reference is used.
        """
        if make_persistent:
            # Not a weakref, but always return the callable.
            self._hooks[id(callable)] = lambda: callable
        elif getattr(callable, "__self__", None):
            # Add a method reference to the hooks
            key = id(callable.__self__)
            self._hooks[key] = weakref.WeakMethod(
                callable, functools.partial(_cleanup, self._hooks, key)
            )
        else:
            # Add a function reference to the hooks
            key = id(callable)
            self._hooks[key] = weakref.ref(callable, functools.partial(_cleanup, self._hooks, key))

    def run(self, *args, **kwargs) -> None:
        """
        Run all the callbacks in the registry, passing the given arguments.
        """
        for hook in self._hooks.values():
            ref = hook()
            if ref is not None:
                ref(*args, **kwargs)


_after_in_child_fork_hooks = WeakCallbacks()
_after_in_parent_fork_hooks = WeakCallbacks()
_before_fork_hooks = WeakCallbacks()


def before_fork_hook(callable: Callable[[], None], make_persistent: bool = False):
    """
    Run function before the fork of a worker process.
    The function must be persistent (i.e. not a lambda) or an instance method.

    Args:
        callable: The function to run before the fork of a worker process.
        make_persistent: If True, the function will be stored as a strong reference, otherwise a weak reference is used.
    """
    _before_fork_hooks.add_hook(callable, make_persistent)


def after_in_parent_fork_hook(callable: Callable[[], None], make_persistent: bool = False):
    """
    Run function after the fork of a worker process.
    The function must be persistent (i.e. not a lambda) or an instance method.

    Args:
        callable: The function to run after the fork of a worker process.
        make_persistent: If True, the function will be stored as a strong reference, otherwise a weak reference is used.
    """
    _after_in_parent_fork_hooks.add_hook(callable, make_persistent)


def after_in_child_fork_hook(callable: Callable[[], None], make_persistent: bool = False):
    """
    Run function after the fork of a worker process.
    The function must be persistent (i.e. not a lambda) or an instance method.

    Args:
        callable: The function to run after the fork of a worker process.
        make_persistent: If True, the function will be stored as a strong reference, otherwise a weak reference is used.
    """
    _after_in_child_fork_hooks.add_hook(callable, make_persistent)


class ForkMixin:
    """
    A mixin that runs a method after the fork of a worker process.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__post_init__()

    def __post_init__(self):
        if getattr(self.__before_fork__, "__func__", None) is not ForkMixin.__before_fork__:
            before_fork_hook(self.__before_fork__)
        if (
            getattr(self.__after_in_child_fork__, "__func__", None)
            is not ForkMixin.__after_in_child_fork__
        ):
            after_in_child_fork_hook(self.__after_in_child_fork__)
        if (
            getattr(self.__after_in_parent_fork__, "__func__", None)
            is not ForkMixin.__after_in_parent_fork__
        ):
            after_in_parent_fork_hook(self.__after_in_parent_fork__)

    def __after_in_child_fork__(self):
        """
        A method that runs after the fork in the child process.
        """
        pass

    def __after_in_parent_fork__(self):
        """
        A method that runs after the fork in the parent process.
        """
        pass

    def __before_fork__(self):
        """
        A method that runs before the fork of a worker process.
        """
        pass


@dataclass
class DataclassForkMixin:
    """
    A mixin that runs a method after the fork of a worker process.
    """

    def __post_init__(self):
        if (
            getattr(self.__before_fork__, "__func__", None)
            is not DataclassForkMixin.__before_fork__
        ):
            before_fork_hook(self.__before_fork__)
        if (
            getattr(self.__after_in_child_fork__, "__func__", None)
            is not DataclassForkMixin.__after_in_child_fork__
        ):
            after_in_child_fork_hook(self.__after_in_child_fork__)
        if (
            getattr(self.__after_in_parent_fork__, "__func__", None)
            is not DataclassForkMixin.__after_in_parent_fork__
        ):
            after_in_parent_fork_hook(self.__after_in_parent_fork__)

    def __after_in_child_fork__(self):
        """
        A method that runs after the fork in the child process.
        """
        pass

    def __after_in_parent_fork__(self):
        """
        A method that runs after the fork in the parent process.
        """
        pass

    def __before_fork__(self):
        """
        A method that runs before the fork of a worker process.
        """
        pass


os.register_at_fork(
    before=_before_fork_hooks.run,
    after_in_child=_after_in_child_fork_hooks.run,
    after_in_parent=_after_in_parent_fork_hooks.run,
)
