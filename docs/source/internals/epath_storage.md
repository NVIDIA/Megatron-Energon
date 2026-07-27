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

For a mounted or joined sample, distinguish:

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
