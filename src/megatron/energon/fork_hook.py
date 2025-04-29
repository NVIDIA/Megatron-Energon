# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import os
import weakref
from typing import Callable, Protocol, Type, TypeVar

_after_in_child_fork_hooks = weakref.WeakSet()
_after_in_parent_fork_hooks = weakref.WeakSet()
_before_fork_hooks = weakref.WeakSet()

T = TypeVar("T", bound=Callable[[], None])


def after_in_child_fork_hook(callable: T):
    """
    Run function after the fork of a worker process.
    """
    _after_in_child_fork_hooks.add(callable)


def before_fork_hook(callable: T):
    """
    Run function before the fork of a worker process.
    """
    _before_fork_hooks.add(callable)


def after_in_parent_fork_hook(callable: T):
    """
    Run function after the fork of a worker process.
    """
    _after_in_parent_fork_hooks.add(callable)


class ForkMixin:
    """
    A mixin that runs a method after the fork of a worker process.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        before_fork_hook(self.__before_fork__)
        after_in_child_fork_hook(self.__after_in_child_fork__)
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


class ForkHookProtocol(Protocol):
    """
    A protocol that defines a method that runs before and after the fork of a worker process.
    """

    def __after_in_child_fork__(self):
        """
        A method that runs after the fork in the child process.
        """
        ...

    def __after_in_parent_fork__(self):
        """
        A method that runs after the fork in the parent process.
        """
        ...

    def __before_fork__(self):
        """
        A method that runs before the fork of a worker process.
        """
        ...


T_CLS = TypeVar("T_CLS", bound=Type[ForkHookProtocol])


def fork_hook_class(cls: T_CLS) -> T_CLS:
    """
    A decorator that runs a function after the fork of a worker process.
    """
    if hasattr(cls, "__init__"):
        orig_init = cls.__init__

        @functools.wraps(orig_init)
        def __init__(self, *args, **kwargs):
            _after_in_child_fork_hooks.add(self.__after_in_child_fork__)
            _after_in_parent_fork_hooks.add(self.__after_in_parent_fork__)
            _before_fork_hooks.add(self.__before_fork__)
            orig_init(self, *args, **kwargs)

        cls.__init__ = __init__
    else:

        def __init__(self, *args, **kwargs):
            _after_in_child_fork_hooks.add(cls.__after_in_child_fork__)
            _after_in_parent_fork_hooks.add(cls.__after_in_parent_fork__)
            _before_fork_hooks.add(cls.__before_fork__)
            cls(*args, **kwargs)

        cls.__init__ = __init__
    return cls


def _run_before_fork_hooks():
    """
    Run all the functions that were registered with the before_fork_hook decorator.
    """
    # print(f"Running before_fork_hooks for pid={os.getpid()}")
    for hook in _before_fork_hooks:
        hook()


def _run_after_in_child_fork_hooks():
    """
    Run all the functions that were registered with the after_in_child_fork_hook decorator.
    """
    # print(f"Running after_in_child_fork_hooks for pid={os.getpid()}")
    for hook in _after_in_child_fork_hooks:
        hook()


def _run_after_in_parent_fork_hooks():
    """
    Run all the functions that were registered with the after_in_parent_fork_hook decorator.
    """
    # print(f"Running after_in_parent_fork_hooks for pid={os.getpid()}")
    for hook in _after_in_parent_fork_hooks:
        hook()


os.register_at_fork(
    before=_run_before_fork_hooks,
    after_in_child=_run_after_in_child_fork_hooks,
    after_in_parent=_run_after_in_parent_fork_hooks,
)
