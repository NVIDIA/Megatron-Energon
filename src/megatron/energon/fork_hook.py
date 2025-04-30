# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import os
import weakref
from typing import Any, Callable, TypeVar

_after_in_child_fork_hooks = weakref.WeakKeyDictionary()
_after_in_parent_fork_hooks = weakref.WeakKeyDictionary()
_before_fork_hooks = weakref.WeakKeyDictionary()


T = TypeVar("T", bound=Callable[[], None])


def before_fork_hook(obj: Any, callable: T):
    """
    Run function before the fork of a worker process.
    """
    # print(f"Adding before_fork_hook for {callable.__name__}\n", end="")
    _before_fork_hooks[obj] = callable


def after_in_child_fork_hook(obj: Any, callable: T):
    """
    Run function after the fork of a worker process.
    """
    # print(f"Adding after_in_child_fork_hook for {callable.__name__}\n", end="")
    _after_in_child_fork_hooks[obj] = callable


def after_in_parent_fork_hook(obj: Any, callable: T):
    """
    Run function after the fork of a worker process.
    """
    # print(f"Adding after_in_parent_fork_hook for {callable.__name__}\n", end="")
    _after_in_parent_fork_hooks[obj] = callable


def _run_before_fork_hooks():
    """
    Run all the functions that were registered with the before_fork_hook decorator.
    """
    # print(f"Running before_fork_hooks for pid={os.getpid()}")
    for hook in _before_fork_hooks.values():
        # print(f"Running before_fork_hook for {hook.__name__}\n", end="")
        hook()


def _run_after_in_child_fork_hooks():
    """
    Run all the functions that were registered with the after_in_child_fork_hook decorator.
    """
    # print(f"Running after_in_child_fork_hooks for pid={os.getpid()}")
    for hook in _after_in_child_fork_hooks.values():
        # print(f"Running after_in_child_fork_hook for {hook.__name__}\n", end="")
        hook()


def _run_after_in_parent_fork_hooks():
    """
    Run all the functions that were registered with the after_in_parent_fork_hook decorator.
    """
    # print(f"Running after_in_parent_fork_hooks for pid={os.getpid()}")
    for hook in _after_in_parent_fork_hooks.values():
        # print(f"Running after_in_parent_fork_hook for {hook.__name__}\n", end="")
        hook()


os.register_at_fork(
    before=_run_before_fork_hooks,
    after_in_child=_run_after_in_child_fork_hooks,
    after_in_parent=_run_after_in_parent_fork_hooks,
)
