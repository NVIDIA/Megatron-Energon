<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Error Handling

Energon provides configurable error handling to gracefully manage exceptions during data loading and processing.
By default, errors in task encoder code are logged and the sample is skipped, allowing training to continue.
Some system exceptions (clearly programming errors) always cause immediate failure.

By default, if more than 100 samples in a row raise an error, processing will also fail. This is configurable for the
{py:class}`~megatron.energon.task_encoder.base.TaskEncoder` user-functions by setting the `@stateless(failure_tolerance=100)` parameter.

## Error Handler Configuration

Error handlers are configured via {py:class}`WorkerConfig <megatron.energon.WorkerConfig>`:

```python
from megatron.energon import WorkerConfig, log_exception, reraise_exception

worker_config = WorkerConfig(
    rank=0,
    world_size=1,
    num_workers=4,
    global_error_handler=log_exception,  # Default: logs and skips failed samples
    restore_error_handler=reraise_exception,  # Default: fails fast on restore
)
```

**`global_error_handler`:** Handles exceptions during normal iteration (training/validation).
Default is {py:func}`log_exception <megatron.energon.log_exception>` which logs details and skips the sample.

**`restore_error_handler`:** Handles exceptions during checkpoint restoration.
Default is {py:func}`reraise_exception <megatron.energon.reraise_exception>` which fails fast to ensure checkpoint integrity.

## System Exceptions

Programming errors and critical system issues always cause immediate failure and are never handled by error handlers:
`SystemError`, `SyntaxError`, `ImportError`, `StopIteration`, `StopAsyncIteration`, `MemoryError`, `RecursionError`, `ReferenceError`, `NameError`, `UnboundLocalError`, and {py:exc}`FatalSampleError <megatron.energon.FatalSampleError>`.
{py:exc}`FatalSampleError <megatron.energon.FatalSampleError>` is raised automatically when consecutive failure tolerance is exceeded or when a system exception occurs during sample processing.

## Built-in Error Handlers

### `log_exception`

Logs detailed error information and continues:
- Exception traceback
- Source information (dataset path, shard, index)
- Sample details in readable format

```python
from megatron.energon import log_exception

worker_config = WorkerConfig(
    rank=0,
    world_size=1,
    num_workers=4,
    global_error_handler=log_exception,
)
```

### `reraise_exception`

Immediately reraises the exception to halt iteration:

```python
from megatron.energon import reraise_exception

worker_config = WorkerConfig(
    rank=0,
    world_size=1,
    num_workers=4,
    global_error_handler=reraise_exception,  # Fail on any error
)
```

### Custom Error Handlers

Implement custom error handlers with this signature:

```python
def my_error_handler(
    exception: Exception,
    sample: Any,
    sources: list[SourceInfo] | None
) -> None:
    # Log to your monitoring system
    log_to_monitoring(exception, sample)
    
    # Optionally reraise for critical errors
    if isinstance(exception, CriticalError):
        raise exception
```

```python
worker_config = WorkerConfig(
    rank=0,
    world_size=1,
    num_workers=4,
    global_error_handler=my_error_handler,
)
```

## Failure Tolerance for Task Encoder Functions

By default, if more than 100 samples in a row raise an error, processing will fail with a {py:exc}`FatalSampleError <megatron.energon.FatalSampleError>`.

For {py:class}`TaskEncoder <megatron.energon.TaskEncoder>` methods, configure this via the `@stateless` decorator:

```python
from megatron.energon import DefaultTaskEncoder, stateless

class MyTaskEncoder(DefaultTaskEncoder):
    @stateless(failure_tolerance=50)
    def encode_sample(self, sample):
        # Process sample - tolerates up to 50 consecutive failures
        if sample.is_corrupted():
            raise ValueError("Corrupted sample")
        return sample
    
    @stateless(restore_seeds=True, failure_tolerance=200)
    def pack_selected_samples(self, samples):
        # Packing with higher tolerance and deterministic randomness
        return pack_samples(samples)
```

Set `failure_tolerance=0` to disable tolerance checking for a specific function.

```{admonition} Note
:class: important
Tolerance limits count *consecutive* failures. A single successful sample resets the counter.
```

## Skip or Fail Explicitly

Raise {py:exc}`SkipSample <megatron.energon.SkipSample>` to explicitly skip a sample without logging an error:

```python
from megatron.energon import SkipSample

def process_sample(sample):
    try:
        ...
    except MySpecificError:
        raise SkipSample()
    return sample
```

Raise {py:exc}`FatalSampleError <megatron.energon.FatalSampleError>` to cause immediate failure, bypassing the error handler:

```python
from megatron.energon import FatalSampleError

def process_sample(sample):
    try:
        ...
    except MyFatalError as e:
        raise FatalSampleError.from_sample(sample, "Critical corruption detected") from e
    return sample
```
