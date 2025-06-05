# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import os
import weakref
from typing import Callable, Protocol, Type, TypeVar

_after_in_child_fork_hooks = weakref.WeakKeyDictionary()
_after_in_parent_fork_hooks = weakref.WeakKeyDictionary()
_before_fork_hooks = weakref.WeakKeyDictionary()


T = TypeVar("T", bound=Callable[[], None])


def before_fork_hook(callable: Callable[[], None]):
    """
    Run function before the fork of a worker process. The function must be persistent.
    """
    # Make sure, that callable is a method of object
    assert getattr(callable, "__self__", None) is None, (
        f"Callable must not be a method: {callable.__name__}"
    )
    # print(f"Adding before_fork_hook for {callable.__name__}\n", end="")
    _before_fork_hooks[callable] = callable


def after_in_parent_fork_hook(callable: T):
    """
    Run function after the fork of a worker process. The function must be persistent.
    """
    # print(f"Adding after_in_child_fork_hook for {callable.__name__}\n", end="")
    assert getattr(callable, "__self__", None) is None, (
        f"Callable must not be a method: {callable.__name__}"
    )
    _after_in_parent_fork_hooks[callable] = callable


def after_in_child_fork_hook(callable: T):
    """
    Run function after the fork of a worker process. The function must be persistent.
    """
    # print(f"Adding after_in_child_fork_hook for {callable.__name__}\n", end="")
    assert getattr(callable, "__self__", None) is None, (
        f"Callable must not be a method: {callable.__name__}"
    )
    _after_in_child_fork_hooks[callable] = callable


class ForkMixin:
    """
    A mixin that runs a method after the fork of a worker process.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.__before_fork__, "__func__", None) is not ForkMixin.__before_fork__:
            _before_fork_hooks[self] = "__before_fork__"
        if (
            getattr(self.__after_in_child_fork__, "__func__", None)
            is not ForkMixin.__after_in_child_fork__
        ):
            _after_in_child_fork_hooks[self] = "__after_in_child_fork__"
        if (
            getattr(self.__after_in_parent_fork__, "__func__", None)
            is not ForkMixin.__after_in_parent_fork__
        ):
            _after_in_parent_fork_hooks[self] = "__after_in_parent_fork__"

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
            _after_in_child_fork_hooks[self] = "__after_in_child_fork__"
            _after_in_parent_fork_hooks[self] = "__after_in_parent_fork__"
            _before_fork_hooks[self] = "__before_fork__"
            orig_init(self, *args, **kwargs)

        cls.__init__ = __init__
    else:

        def __init__(self, *args, **kwargs):
            _after_in_child_fork_hooks[cls] = "__after_in_child_fork__"
            _after_in_parent_fork_hooks[cls] = "__after_in_parent_fork__"
            _before_fork_hooks[cls] = "__before_fork__"
            cls(*args, **kwargs)

        cls.__init__ = __init__
    return cls


def _run_before_fork_hooks():
    """
    Run all the functions that were registered with the before_fork_hook decorator.
    """
    # print(f"Running before_fork_hooks for pid={os.getpid()}")
    for obj, hook in _before_fork_hooks.items():
        # print(f"Running before_fork_hook for {hook}\n", end="")
        if callable(hook):
            hook()
        else:
            getattr(obj, hook)()


def _run_after_in_child_fork_hooks():
    """
    Run all the functions that were registered with the after_in_child_fork_hook decorator.
    """
    # print(f"Running after_in_child_fork_hooks for pid={os.getpid()}")
    for obj, hook in _after_in_child_fork_hooks.items():
        # print(f"Running after_in_child_fork_hook for {hook}\n", end="")
        if callable(hook):
            hook()
        else:
            getattr(obj, hook)()


def _run_after_in_parent_fork_hooks():
    """
    Run all the functions that were registered with the after_in_parent_fork_hook decorator.
    """
    # print(f"Running after_in_parent_fork_hooks for pid={os.getpid()}")
    for obj, hook in _after_in_parent_fork_hooks.items():
        # print(f"Running after_in_parent_fork_hook for {hook}\n", end="")
        if callable(hook):
            hook()
        else:
            getattr(obj, hook)()


os.register_at_fork(
    before=_run_before_fork_hooks,
    after_in_child=_run_after_in_child_fork_hooks,
    after_in_parent=_run_after_in_parent_fork_hooks,
)
