# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import contextmanager
from typing import Any, Callable, Generator, Type, TypeVar

from megatron.energon.sample_utils import format_sample_compact, format_sample_detailed
from megatron.energon.source_info import SourceInfo, get_source_info

T = TypeVar("T")


class SkipSample(Exception):
    """Raise this exception in any processing function to skip the current sample."""

    pass


class SampleException(ValueError):
    @classmethod
    def from_sample_key(cls: Type[T], sample_key: str) -> T:
        return cls(f"Sample {sample_key} failed")

    @classmethod
    def from_sample(cls: Type[T], sample: Any, message: str = "") -> T:
        if message:
            message = f": {message}"
        return cls(f"Sample {format_sample_compact(sample)} failed{message}")


class FatalSampleError(SampleException):
    # This will not be handled by the error handler
    pass


SYSTEM_EXCEPTIONS = (
    SystemError,
    SyntaxError,
    ImportError,
    StopIteration,
    StopAsyncIteration,
    MemoryError,
    RecursionError,
    ReferenceError,
    NameError,
    UnboundLocalError,
    FatalSampleError,
)


class ErrorContext:
    """Tracks consecutive errors and enforces error tolerance limits.

    This class helps prevent infinite error loops by tracking consecutive failures
    and raising a FatalSampleError when a tolerance threshold is exceeded.

    Example:
        error_ctx = ErrorContext(
            name="MapDataset.map_fn",
            handler=self.worker_config.global_error_handler
            tolerance=100,
        )

        with error_ctx.handle_errors(sample):
            result = process_sample(sample)
    """

    name: str
    tolerance: int
    handler: Callable[[Exception, Any, list["SourceInfo"] | None], None]

    _consecutive_failures: int = 0

    def __init__(
        self,
        name: str,
        handler: Callable[[Exception, Any, list["SourceInfo"] | None], None],
        tolerance: int = 100,
    ):
        """Initialize error context.

        Args:
            name: Name of the operation being tracked (for error messages).
            handler: Error handler function to call on exceptions. Takes (exception, sample, sources).
                If None, exceptions will be raised after incrementing the counter.
            tolerance: Maximum number of consecutive failures before raising FatalSampleError.
                Set to 0 to disable tolerance checking.
        """
        self.name = name
        self.tolerance = tolerance
        self.handler = handler

    def reset(self) -> None:
        """Reset the consecutive failures counter."""
        self._consecutive_failures = 0

    @contextmanager
    def handle_errors(
        self,
        sample: Any,
    ) -> Generator[None, None, None]:
        """Context manager for handling exceptions during sample processing.

        Automatically tracks consecutive failures and resets on success.

        Args:
            sample: The sample being processed (used in error reporting).
        """
        try:
            yield
            # Success - reset counter
            self._consecutive_failures = 0
        except GeneratorExit:
            raise
        except SkipSample:
            pass
        except SYSTEM_EXCEPTIONS as e:
            raise FatalSampleError.from_sample(
                sample, f"{self.name} failed due to system exception: {e}."
            )
        except Exception as e:
            print(f"Except {e} in {self.name}")
            # Call the error handler if provided
            if self.handler is not None:
                # Call the error handler
                self.handler(e, sample, get_source_info(sample))

            # Increment counter (may raise FatalSampleError if tolerance exceeded)
            self._consecutive_failures += 1

            if self._consecutive_failures > 1:
                print(
                    f"ErrorContext {self.name} failed {self._consecutive_failures}/{self.tolerance} times in a row."
                )
            if self.tolerance > 0 and self._consecutive_failures >= self.tolerance:
                raise FatalSampleError.from_sample(
                    sample,
                    (
                        f"{self.name} failed {self._consecutive_failures} times in a row. "
                        f"Likely your code or dataset are broken."
                    ),
                )

    def __repr__(self) -> str:
        return f"ErrorContext(name={self.name!r}, tolerance={self.tolerance}, count={self._consecutive_failures})"


@contextmanager
def handle_restore_errors(
    error_handler: Callable[[Exception, Any, list["SourceInfo"] | None], None],
    sample: Any,
) -> Generator[None, None, None]:
    """Context manager for handling exceptions during sample restoration.

    Args:
        error_handler: Function to call when an exception occurs. Takes (exception, sample, sources).
        sample: The sample being restored.
    """
    try:
        yield
    except SkipSample as e:
        # Unexpected skip sample
        try:
            raise ValueError(f"Unexpected skip sample {sample} during restoration.") from e
        except Exception as e:
            error_handler(e, sample, get_source_info(sample))
    except GeneratorExit as e:
        # Unexpected skip sample
        try:
            raise ValueError(
                f"Unexpected generator early stopping for sample {sample} during restoration."
            ) from e
        except Exception as e:
            error_handler(e, sample, get_source_info(sample))
    except SYSTEM_EXCEPTIONS as e:
        raise FatalSampleError.from_sample(sample) from e
    except Exception as e:
        error_handler(e, sample, get_source_info(sample))


def log_exception(e: Exception, sample: Any, sources: list["SourceInfo"] | None = None) -> None:
    """Error handler that logs exceptions with sample information.

    This function prints the exception traceback, source information if available,
    and a smart representation of the failed sample to help with debugging.

    Args:
        e: The exception that was raised.
        sample: The sample that caused the exception.
        sources: Optional list of SourceInfo objects with sample provenance.
    """
    import traceback

    traceback.print_exc()
    print("-" * 10)

    if sources:
        print("Sources:")
        for source in sources:
            if hasattr(source, "dataset_path"):
                print(
                    f" - {source.dataset_path}[{source.index}] {source.shard_name}{source.file_names!r}"
                )
        print("-" * 10)

    sample_str = format_sample_detailed(sample)
    print(sample_str)

    print("-" * 10)


def reraise_exception(
    e: Exception, _sample: Any, _sources: list["SourceInfo"] | None = None
) -> None:
    """Error handler that simply reraises the exception.

    This is useful when you want failures to propagate immediately without
    any tolerance or logging.

    Args:
        e: The exception to reraise.
        _sample: The sample (unused).
        _sources: Source info (unused).

    Raises:
        The original exception.
    """
    raise e
