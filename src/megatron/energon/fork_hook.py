# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import os
import weakref
from dataclasses import dataclass
from typing import Callable

_after_in_child_fork_hooks = dict()
_after_in_parent_fork_hooks = dict()
_before_fork_hooks = dict()


def before_fork_hook(callable: Callable[[], None]):
    """
    Run function before the fork of a worker process. The function must be persistent.
    """
    if getattr(callable, "__self__", None):
        self = callable.__self__
        _before_fork_hooks[id(self)] = callable
        weakref.finalize(self, lambda: _before_fork_hooks.pop(id(self)))
    else:
        _before_fork_hooks[id(callable)] = callable
        weakref.finalize(callable, lambda: _before_fork_hooks.pop(id(callable)))


def after_in_parent_fork_hook(callable: Callable[[], None]):
    """
    Run function after the fork of a worker process. The function must be persistent.
    """
    # print(f"Adding after_in_child_fork_hook for {callable.__name__}\n", end="")
    if getattr(callable, "__self__", None):
        self = callable.__self__
        _after_in_parent_fork_hooks[id(self)] = callable
        weakref.finalize(self, lambda: _after_in_parent_fork_hooks.pop(id(self)))
    else:
        _after_in_parent_fork_hooks[id(callable)] = callable
        weakref.finalize(callable, lambda: _after_in_parent_fork_hooks.pop(id(callable)))


def after_in_child_fork_hook(callable: Callable[[], None]):
    """
    Run function after the fork of a worker process. The function must be persistent.
    """
    # print(f"Adding after_in_child_fork_hook for {callable.__name__}\n", end="")
    if getattr(callable, "__self__", None):
        self = callable.__self__
        _after_in_child_fork_hooks[id(self)] = callable
        weakref.finalize(self, lambda: _after_in_child_fork_hooks.pop(id(self)))
    else:
        _after_in_child_fork_hooks[id(callable)] = callable
        weakref.finalize(callable, lambda: _after_in_child_fork_hooks.pop(id(callable)))


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


def _run_before_fork_hooks():
    """
    Run all the functions that were registered with the before_fork_hook decorator.
    """
    # print(f"Running before_fork_hooks for pid={os.getpid()}")
    for hook in _before_fork_hooks.values():
        # print(f"Running before_fork_hook for {hook}\n", end="")
        hook()


def _run_after_in_child_fork_hooks():
    """
    Run all the functions that were registered with the after_in_child_fork_hook decorator.
    """
    # print(f"Running after_in_child_fork_hooks for pid={os.getpid()}")
    for hook in _after_in_child_fork_hooks.values():
        # print(f"Running after_in_child_fork_hook for {hook}\n", end="")
        hook()


def _run_after_in_parent_fork_hooks():
    """
    Run all the functions that were registered with the after_in_parent_fork_hook decorator.
    """
    # print(f"Running after_in_parent_fork_hooks for pid={os.getpid()}")
    for hook in _after_in_parent_fork_hooks.values():
        # print(f"Running after_in_parent_fork_hook for {hook}\n", end="")
        hook()


os.register_at_fork(
    before=_run_before_fork_hooks,
    after_in_child=_run_after_in_child_fork_hooks,
    after_in_parent=_run_after_in_parent_fork_hooks,
)
