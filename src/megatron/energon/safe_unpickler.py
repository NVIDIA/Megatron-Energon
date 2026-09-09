# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import pickle
import threading


class SafeUnpickler(pickle.Unpickler):
    """A `pickle.Unpickler` that only constructs classes/functions explicitly allowlisted
    in `_SAFE_CLASSES`, to protect against arbitrary code execution when unpickling data
    from disk. Use `add_safe_classes` to extend the allowlist for custom payload types.

    Usage: `SafeUnpickler(file).load()` instead of `pickle.load(file)`.
    """

    _SAFE_CLASSES: set = {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("PIL.Image", "Image"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self._SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Refusing to unpickle disallowed class '{module}.{name}' "
            )
        return super().find_class(module, name)
