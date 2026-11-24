<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(epath-storage)=

# EPath, Mapped Arrays, and Storage

Energon uses `EPath` so dataset code can work with local files and supported remote protocols through a
common path interface. Readers should preserve that abstraction until an operation specifically requires a
local file.

## Path handling

Accept `EPath` or values convertible to it at API boundaries. Use path composition and file operations
provided by `EPath`; converting to `pathlib.Path` or calling built-in `open` too early silently removes
remote support.

Protocol handlers may also compose an auxiliary path with a primary path, as used by mounted or crude
datasets. Keep the original dataset path in configuration and provenance even when bytes are served from a
cache or auxiliary location.

## Mapped numeric arrays

`map_epath` exposes one-dimensional typed arrays used by indexes. Its backend depends on the path and
options:

- a local path with memory mapping enabled uses `EPathNumpyMappedArray`;
- a remote path, or a request without memory mapping, uses seek/read access;
- `copy_to_local` first creates or reuses a local copy and then memory maps it.

The shape is inferred from file size, byte offset, and dtype. Validate that the remaining byte count is
divisible by the dtype size. The abstraction is one-dimensional; multidimensional interpretation belongs
to the caller.

A stepped slice may read the full byte span between its first and last element and apply the step afterward.
Avoid sparse, extremely wide slices on remote storage when a narrower access pattern is possible.

Mapped arrays and readers are context managers and must be closed when the owner is finished. Do not put an
open file, stream, or memory map into dataset configuration or checkpoint state.

## Frozen construction metadata

Use a snapshot to avoid repeating metadata reads while constructing loaders for
the same frozen dataset revision. Both modes use the same local or shared
filesystem path:

- `create=True` performs the normal metadata I/O and records its results,
  including repeated queries. It creates or replaces the snapshot after the
  context exits successfully.
- `create=False` (the default) reads the existing snapshot and serves recorded
  values without repeating their original storage operations. A missing file is
  an error; reading mode never creates or updates the snapshot.
- `sha256` is optional and only checks the package in reading mode. It is ignored
  when `create=True`, even if the supplied checksum does not match.

### Create the snapshot

This example uses a prepared text WebDataset. For another dataset, use its normal
task encoder and constructor arguments. Create the destination directory first
and enable INFO logging to see the snapshot path and checksum. The example below
configures logging for a standalone preparation script.

```python
import logging
from pathlib import Path

from megatron.energon import WorkerConfig, get_savable_loader, get_train_dataset
from megatron.energon.epathlib.metadata_snapshot import use_metadata_snapshot

logging.basicConfig(level=logging.INFO)
snapshot_path = Path("/local/loader-metadata.json")
snapshot_path.parent.mkdir(parents=True, exist_ok=True)
worker_config = WorkerConfig(rank=0, world_size=1, num_workers=2)


def build_dataset():
    return get_train_dataset(
        "/datasets/my-text-dataset",
        worker_config=worker_config,
        batch_size=8,
        shuffle_buffer_size=1000,
        max_samples_per_sequence=None,
    )


with use_metadata_snapshot(snapshot_path, create=True):
    dataset = build_dataset()
# The snapshot now exists. No iteration was needed to create it.
```

Creation records YAML/JSON contents, file checks, file sizes and JSONL terminal
offsets encountered by the constructor. It does not reuse earlier query results
during creation; repeated queries access storage and replace the recorded value.
It does not store sample/media payloads or dataloader checkpoint state. Captured
file contents are limited to 64 MiB, counting each recorded path once.
If construction or publication fails, an existing snapshot is left unchanged.
Choose one process to create the snapshot, then let other processes read it.

### Read the snapshot

On subsequent constructions, use the same path with `create=False`. The shared
setup and `build_dataset()` function above remain the same.

```python
with use_metadata_snapshot(snapshot_path, create=False):
    dataset = build_dataset()

# Start iteration and workers only after leaving the snapshot context.
with get_savable_loader(dataset) as loader:
    for batch in loader:
        train_step(batch)  # Your training loop.
```

Recorded reads use independent streams. Unrecorded paths use storage normally
and do not extend the snapshot. Each rank still constructs its own dataset with
its normal sharding settings. Leaving either context restores the previous
context, including after an exception.

### Optional checksum

Checksum validation applies only to reading mode; creation ignores `sha256`.
After successfully writing the snapshot, the creating process logs its path and
the full 64-character SHA-256 at INFO level:

```text
Created metadata snapshot /local/loader-metadata.json; sha256=<64-character digest>
```

Copy the value after `sha256=` into your job configuration and pass it when
reading. No separate checksum calculation or file read is needed:

```python
expected_digest = "paste the 64-character digest from the creation log"
with use_metadata_snapshot(snapshot_path, sha256=expected_digest):
    dataset = build_dataset()
```

Omitting `sha256` skips only the checksum comparison; the package must still be
valid and have a supported format. A checksum identifies a package, not the
current contents of its source dataset. Recreate the snapshot after changing
the recipe or dataset metadata. Keep those inputs immutable during capture and
replay. Exit the context before starting workers, including forked workers.

### Package format and in-memory use

Version 1 packages retain `format_version`, `recipe_identity`, base64-encoded
`contents`, and maps named `file_checks`, `sizes`, and `jsonl_sizes`. Keys are
logical EPath strings; `jsonl_sizes` contains terminal byte offsets. Path-based
creation writes an empty recipe identity and does not require a separate label.

Applications that already distribute snapshots in memory can still pass a
`MetadataSnapshot` object to `use_metadata_snapshot(snapshot)`. File-mode options
are omitted for this form. `MetadataSnapshot.load(path, sha256=..., recipe_identity=...)`
remains available; both expected values are optional. Existing labeled packages
remain readable, and an explicitly requested recipe identity is still checked.

## Worker and cache ownership

Open storage resources inside the worker that consumes them. Forked or spawned workers must not rely on a
parent's live file position or remote client object.

When an index is safe to cache locally:

- key the cache by source identity and every option that affects the bytes;
- define freshness checks for source or metadata changes;
- write atomically so another worker cannot observe a partial index;
- treat the cached location as an optimization, not as the dataset's logical identity.

Prepared `.nv-meta` files are part of the dataset configuration contract, while ephemeral cache files are
not.

## Provenance

Every loaded part should retain `SourceInfo` describing where it came from. A `SampleRecord` aggregates
this in `__sources__`, and transformations should propagate it. Provenance is needed for useful error
messages, debugging filters, and tracing samples whose payload came from auxiliary storage.

For a mounted sample, distinguish:

- the logical sample key;
- the primary dataset and shard;
- the physical path used for each part;
- any auxiliary source that supplied the bytes.

Do not replace these with the cache path alone.

## Adding a storage backend

Before adding a new protocol or reader:

1. define seek, range-read, listing, stat, and local-copy capabilities;
2. make unsupported operations fail explicitly;
3. verify path normalization and equality, since paths participate in resume identity;
4. verify cleanup after normal iteration, exceptions, and worker shutdown;
5. test local and remote behavior through the same dataset reader;
6. test empty files, offsets, dtype alignment, slicing, and cache freshness;
7. preserve logical paths and source provenance in configuration and samples.

User-facing remote and auxiliary dataset configuration is documented in
[Remote Datasets](../advanced/remote_dataset.md) and [Crude Datasets](../advanced/crude_datasets.md).
