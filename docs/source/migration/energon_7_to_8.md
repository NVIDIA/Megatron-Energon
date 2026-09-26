<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Migrating from Energon 7.x to 8.x

Energon 8.x replaces the 7.x PyTorch-based loader with the custom
{py:class}`megatron.energon.DataLoader`. This guide describes the application and extension
changes required for that upgrade.

The dataset configuration format does not need a loader-specific migration. Most applications
that only call `get_loader()` or `get_savable_loader()` need lifecycle changes, while applications
with checkpoints, custom `TaskEncoder.batch()` implementations, custom datasets, or custom cache
types need the additional changes below.

## Migration Checklist

1. Use every loader as a context manager, or call `shutdown()` in a `finally` block.
2. Restore loader state before entering the context or otherwise starting the loader.
3. Make sure the object returned by `TaskEncoder.batch()` exposes writable `__restore_key__`
   metadata and that later batch transformations preserve it.
4. Audit randomness in task encoder methods and select global or task-encoder-local RNG handling.
5. Replace the deprecated `save_state()` and `restore_state()` methods with the rank or global
   variants.
6. Do not restore loader checkpoints written by Energon 7.x with the Energon 8.x loader.
7. Remove tuning that depends on the old periodic internal checkpoint arguments.
8. Update custom datasets, wrappers, file stores, and cache pools for the new lifecycle hooks.
9. Update direct `JsonParser` calls if application code uses its import or override options.

## Loader Type and Construction

Both loader factories now return {py:class}`megatron.energon.DataLoader`:

```python
from megatron.energon import DataLoader, get_savable_loader

loader: DataLoader = get_savable_loader(dataset)
```

`SavableDataLoader` remains an alias of `DataLoader` at the top-level `megatron.energon` package
for source compatibility. Direct imports from the removed `megatron.energon.savable_loader`
module must be changed.

```python
# develop
from megatron.energon.savable_loader import SavableDataLoader

# custom DataLoader
from megatron.energon import DataLoader
```

`get_loader()` and `get_savable_loader()` both use the custom loader. Continue using
`get_savable_loader()` when save/restore or its additional worker, garbage collection, and memory
pinning options are required.

## Own the Loader Lifetime

The custom loader owns its dataset and worker resources. It must be shut down explicitly. The
recommended form is a context manager:

```python
with get_loader(dataset) as loader:
    for batch in loader:
        train_step(batch)
```

This also handles exceptions and early loop exits. For a loader whose lifetime cannot be expressed
with `with`, use `try`/`finally`:

```python
loader = get_loader(dataset)
try:
    for batch in loader:
        train_step(batch)
finally:
    loader.shutdown()
```

Framework cleanup registration is equally valid when it matches the loader's lifetime. For
example, a `unittest.TestCase` can use `self.addCleanup(loader.shutdown)`, and code that owns
several loaders dynamically can register `loader.shutdown` with `contextlib.ExitStack.callback()`.

Important lifecycle rules:

- Entering the context calls `start()`.
- Do not enter a loader that has already started; this raises `DataLoader workers are already
  running`.
- Do not rely on garbage collection. Destruction of a running loader emits a resource warning and
  may have to terminate worker processes.
- A loader takes ownership of its dataset and closes it during shutdown. Construct a separate
  dataset instance for each simultaneously active loader.

## Preserve Restore Keys Through `batch()`

This is a subtle but important difference from `develop`. The old loader checkpointed worker
positions independently of the final yielded batch. The custom loader also checkpoints prefetched
outputs, so it reads the restore key from the final batch object.

Consequently, the value returned by a custom {py:meth}`megatron.energon.TaskEncoder.batch` must
provide a writable `__restore_key__` field. `BatchDataset` fills the concrete batched restore key
after `batch()` returns, but it can only do so if that field exists. Returning a plain object or a
dictionary without the field drops the key and makes later state restoration fail.

Prefer returning a subclass of {py:class}`megatron.energon.Batch`:

```python
import torch

from megatron.energon import Batch, TaskEncoder, edataclass


@edataclass
class TokenBatch(Batch):
    tokens: torch.Tensor


class TokenTaskEncoder(TaskEncoder):
    def batch(self, samples):
        return TokenBatch.from_samples(
            samples,
            tokens=torch.stack([sample.tokens for sample in samples]),
        )
```

For dictionary batches, include the key so `BatchDataset` can replace it:

```python
def batch(self, samples):
    return {
        "__key__": [sample.__key__ for sample in samples],
        "__restore_key__": None,
        "tokens": torch.stack([sample.tokens for sample in samples]),
    }
```

Any transformation after `batch()`, especially `encode_batch()`, must preserve the field as well.
When changing from one `Batch` subclass to another, use `derive_from()`:

```python
def encode_batch(self, batch):
    return ModelBatch.derive_from(batch, tokens=normalize(batch.tokens))
```

As a migration test, inspect a yielded batch and assert that `batch.__restore_key__` is not `None`,
then save and restore state with prefetched batches pending.

## Randomness in Task Encoder Functions

Energon 7.x provided `@stateless(restore_seeds=True)` for deterministic task encoder functions. It
temporarily seeds and later restores the process-global RNGs used by:

- `torch` default random functions
- `numpy.random`
- Python's `random` module

This path remains available in Energon 8.x for fork workers and the single main-process worker, so
those configurations can migrate without immediately rewriting random calls. It is not supported
with thread workers because all worker threads share the same process-global RNG state.

```python
import random

import numpy
import torch

from megatron.energon import stateless


@stateless(restore_seeds=True)
def encode_sample(self, sample):
    sample.torch_value = torch.rand(())
    sample.numpy_value = numpy.random.random()
    sample.python_value = random.random()
    return sample
```

```{warning}
Do not use `restore_seeds=True` or process-global random APIs with `worker_type="thread"`. Concurrent
workers can interleave saving, seeding, consuming, and restoring the shared state, making results
scheduling-dependent and breaking deterministic replay. A passing concurrency test does not make
this combination safe.
```

Energon 8.x additionally provides a worker-local RNG through `TaskEncoder.rng`. It is required for
randomized task encoder functions with thread workers and is recommended for new code in every
worker mode. Select this behavior with `restore_task_encoder_seeds=True` and use the generators
exposed by `self.rng`:

```python
import torch

from megatron.energon import stateless


@stateless(restore_task_encoder_seeds=True)
def encode_sample(self, sample):
    sample.torch_value = torch.rand((), generator=self.rng.torch)
    sample.numpy_value = self.rng.numpy.random()
    sample.python_value = self.rng.random.random()
    return sample
```

The task encoder RNG exposes:

| Attribute | Generator |
| --- | --- |
| `self.rng.torch` | CPU `torch.Generator` |
| `self.rng.torch_cuda` | CUDA `torch.Generator`, when CUDA is available |
| `self.rng.numpy` | `numpy.random.Generator` |
| `self.rng.random` | Python `random.Random` |

Worker-mode compatibility is:

| Worker mode | Supported deterministic RNG handling |
| --- | --- |
| `worker_type="fork"` | `TaskEncoder.rng` or process-global RNGs |
| `worker_type="main"` | `TaskEncoder.rng` or process-global RNGs |
| `worker_type="thread"` | `TaskEncoder.rng` only |

Use `restore_seeds=True` only with fork or main-process workers and only for calls that use global
RNG APIs. Use `restore_task_encoder_seeds=True` for calls that use `self.rng`; this path is safe for
all worker modes. Both flags can be enabled temporarily while migrating a fork or main-process
pipeline, but do not mix the two RNG sources in new code.

Plain `@stateless` does not manage randomness. A function that consumes randomness but uses neither
restore option may produce different output after `restore_sample()` or loader checkpoint restore.

For generator task encoder functions, Energon 8.x preserves the selected inner RNG state between
yields and restores the caller's outer RNG state whenever control leaves the generator. Do not
manually reseed between yields. Add a save/restore test that compares every yielded value, not only
the first value.

Switching existing code from global RNGs to `self.rng` is deterministic under the new scheme, but
it may change the exact random sequence. Treat that switch as a data-stream change rather than
expecting bitwise continuity with an Energon 7.x run.

## Save and Restore State

Save state while the loader is active:

```python
with get_savable_loader(dataset) as loader:
    for _, batch in zip(range(100), loader):
        train_step(batch)
    state = loader.save_state_rank()
```

Restore state before entering the context. Entering starts the workers and state cannot be restored
after that point.

```python
loader = get_savable_loader(new_dataset)
loader.restore_state_rank(state)

with loader:
    for batch in loader:
        train_step(batch)
```

The chaining helper is equivalent and keeps the required ordering:

```python
with get_savable_loader(new_dataset).with_restored_state_rank(state) as loader:
    for batch in loader:
        train_step(batch)
```

Use the following method replacements:

| `develop` method | Custom DataLoader method |
| --- | --- |
| `save_state(dst_rank)` | `save_state_global(global_dst_rank)` |
| `restore_state(state)` | `restore_state_global(state)` |
| `save_state_rank()` | Unchanged |
| `restore_state_rank(state)` | Unchanged |
| `save_state_global(global_dst_rank)` | Unchanged |
| `restore_state_global(state, src_rank=...)` | Unchanged |

### Checkpoint Compatibility

Loader checkpoints are not compatible across this migration. Energon 7.x writes
`SavableDataLoaderState` and periodic `SavableDatasetCheckpoint` state, while the custom loader
writes `RankState` containing worker states and restore keys for prefetched outputs. The new
`restore_state_rank()` requires a `RankState`.

Plan the upgrade at a training restart boundary. Either finish the old run with the Energon 7.x
environment or start the custom loader from an initial state. Keep the old code environment if old
checkpoints must remain resumable.

Normal determinism requirements still apply within one implementation: keep the dataset
configuration, seed, split, task encoder, and other stream-defining inputs consistent when
restoring.

## Factory Option Changes

`get_savable_loader()` adds these options:

| Option | Meaning |
| --- | --- |
| `worker_type="fork"` | Default process workers. |
| `worker_type="thread"` | Thread workers, intended for free-threaded Python and thread-safe extensions. |
| `worker_type="main"` | Execute worker logic in the main process. |
| `gc_freeze_at_start` | Controls garbage collector freezing when workers start. |
| `pin_memory` | Enables automatic CUDA-oriented memory pinning. |

The following compatibility arguments are still accepted by `get_savable_loader()` but are
ignored by the custom loader:

- `checkpoint_every_sec`
- `checkpoint_every_min_n_samples`
- `n_checkpoints`

Remove operational assumptions and tuning based on those periodic internal checkpoints. The custom
loader requests current worker state directly when `save_state_rank()` or `save_state_global()` is
called.

Passing `worker_config` to a loader factory remains deprecated. Set it on the dataset instead.
When `num_workers == 0`, keep the factory's default `prefetch_factor`; the factory maps it to the
single in-process worker required by the custom loader.

## Restore-Key Types for Custom Datasets

Restore keys are typed dataclasses instead of nested tuples. `Sample.__restore_key__` is now a
`RestoreKey | None`, and `Batch.__restore_key__` contains restore-key objects.

Custom wrappers that previously prepended tuple elements with `add_sample_restore_key()` should
define a frozen restore-key dataclass and use `wrap_sample_restore_key()`:

```python
from dataclasses import dataclass

from megatron.energon.flavors.base_dataset import RestoreKey
from megatron.energon.wrappers.base import WrappedRestoreKey, wrap_sample_restore_key


@dataclass(kw_only=True, slots=True, frozen=True)
class MyRestoreKey(WrappedRestoreKey):
    dataset_idx: int


def add_wrapper_key(sample, dataset_idx):
    return wrap_sample_restore_key(
        sample,
        MyRestoreKey,
        dataset_idx=dataset_idx,
    )


def restore_sample(self, restore_key: RestoreKey):
    assert isinstance(restore_key, MyRestoreKey)
    return self.datasets[restore_key.dataset_idx].restore_sample(restore_key.inner)
```

Do not depend on tuple slicing or string class tags in application code. `RestoreKey.as_tuple()` is
available for logging and diagnostic serialization, but `restore_sample()` consumes the typed key.

## Custom Dataset and Resource Hooks

The custom loader can run datasets in processes, threads, or the main process. Custom extensions
must distinguish shared lifetime from worker-local lifetime.

| Extension | Required migration |
| --- | --- |
| Direct `SavableDataset` subclass | Call `super().__init__(worker_config)`. Implement `reset_state()` for worker iteration state. |
| `BaseWrapperDataset` subclass | Continue implementing `reset_state_own()`; the base recursively resets wrapped datasets. |
| Mutable worker-only attributes | Add field names to `_worker_local_fields`. `_savable_fields` are also worker-local and checkpointed. |
| Worker-local handles | Open or initialize them for the worker and release them in `worker_close()`. |
| Shared dataset handles | Release them in `close()`, which loader shutdown invokes. |
| Custom `FileStore` | Add `worker_init()`, `worker_close()`, and `close()` as needed. |
| Custom `CachePool` | Implement the new `worker_init()` and `worker_close()` abstract methods. |

Thread workers share the same Python objects. Do not select `worker_type="thread"` until mutable
state and external libraries used by the dataset are thread-safe. Fork workers remain the default.

## Typed Configuration Parsing

Applications that call {py:class}`megatron.energon.typed_converter.JsonParser` directly need to
move `allow_imports` from individual conversion calls to the parser constructor:

```python
from megatron.energon.typed_converter import JsonParser

# Energon 7.x
value = JsonParser().raw_to_typed(raw, Config, allow_imports=False)

# Energon 8.x
value = JsonParser(allow_imports=False).raw_to_typed(raw, Config)
```

The same change applies to `safe_call_function()`. The parser-level default is now
`allow_imports=True`; set it to `False` explicitly when parsing untrusted input or when dynamic
`__module__` and `__class__`/`__function__` resolution is not required.

`JsonParser.override()` has been removed. Code that called it directly must apply overrides to the
raw configuration before calling `raw_to_typed()`, or retain equivalent application-level override
logic. Also account for these corrected conversion results when comparing serialized or cached
configuration objects:

- A fixed-size `Tuple[...]` is returned as a tuple rather than a list.
- Dataclass default values and default-factory results are used directly instead of being passed
  through conversion again.

## Validation Before Deployment

Run at least these checks for each custom pipeline:

1. Iterate and exit early with `num_workers=0` and with process workers; no loader cleanup warnings
   should be emitted.
2. Save state in the middle of an epoch, shut down, rebuild the dataset and loader, restore before
   startup, and compare the following batches.
3. Repeat the save/restore test with prefetching enabled and assert final batches carry non-`None`
   restore keys.
4. Exercise every custom `batch()` and `encode_batch()` return type.
5. Compare random outputs before and after restore for every randomized task encoder function,
   including all yields from generator functions.
6. If using custom stores or caches, verify both normal shutdown and exception cleanup.
7. If opting into thread workers, audit out all `restore_seeds=True` decorators and process-global
   random calls, then run concurrency tests under the intended free-threaded Python runtime.

For implementation details, see [](../internals/dataloader). For distributed checkpoint examples,
see [](../basic/save_restore) and [](../advanced/parallelism).
