<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(custom-dataset-factories)=

# Custom Dataset Factories

Dataset preparation defines the persistent representation. Dataset factory
resolution decides which runtime implementation opens a path. Keeping these
concerns separate makes it possible to add a specialized loader without
placing application-specific registration in the data-preparation workflow.

Use a custom factory when:

- a single-file format should be detected from its suffix or file contents;
- an application needs a specialized variant of an existing factory;
- a prepared manifest names an application-owned factory; or
- a new reader should participate in the standard Energon worker, filtering,
  shuffle, and restore pipeline.

A custom auxiliary key-value container is a different extension point. See
{ref}`custom-aux-protocols` when the data is looked up by key from a cooker
rather than iterated as the
primary dataset.

## Registering a Path Provider

{py:func}`register_dataset_factory_provider
<megatron.energon.flavors.dataset_factory_resolver.register_dataset_factory_provider>`
registers a provider with two class methods:

- `detect_path(path)` returns an `EnergonDatasetType` when the provider
  recognizes the path, otherwise `None`;
- `from_path(path, **kwargs)` constructs a
  {py:class}`BaseCoreDatasetFactory
  <megatron.energon.BaseCoreDatasetFactory>`.

A dataset factory class may be its own provider. This example gives a
specialized JSONL subclass precedence for `.special.jsonl` files and inherits
the built-in `from_path` implementation:

```python
from megatron.energon import DefaultCrudeJsonlDatasetFactory
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.dataset_factory_resolver import (
    PRIORITY_FIRST,
    register_dataset_factory_provider,
)
from megatron.energon.flavors.dataset_type import EnergonDatasetType


@register_dataset_factory_provider(priority=PRIORITY_FIRST)
class SpecialJsonlDatasetFactory(DefaultCrudeJsonlDatasetFactory):
    @classmethod
    def detect_path(cls, path: EPath) -> EnergonDatasetType | None:
        if path.name.endswith(".special.jsonl") and path.is_file():
            return EnergonDatasetType.JSONL
        return None
```

Import the registration module before calling
{py:func}`get_train_dataset <megatron.energon.get_train_dataset>`,
{py:func}`get_val_dataset <megatron.energon.get_val_dataset>`, or
`get_dataset_type`. The registry is process-local, so every rank that resolves
paths must import it.

## Resolution Priority

Providers are checked in ascending numeric priority:

| Priority | Intended use |
| --- | --- |
| `PRIORITY_FIRST` (`0`) | Narrow custom overrides checked before built-ins |
| `PRIORITY_SINGLE_FILE` (`100`) | Built-in single-file formats |
| `PRIORITY_MANIFEST` (`200`) | General prepared `.nv-meta` manifests |

The decorator defaults to `PRIORITY_FIRST`. Providers at the same priority are
checked in registration order, and detection stops at the first non-`None`
result.

Keep `detect_path` cheap and narrow. Check the distinguishing suffix, marker,
or small header rather than opening and parsing an entire dataset. A broad
high-priority provider can silently claim paths intended for another format.

`EnergonDatasetType` is a coarse public classification, while the selected
provider controls construction. A specialized variant should return the
existing category it extends. Introducing a new public category requires an
enum and compatibility change in addition to registering the provider.

## Implementing a New Primary Format

Registration alone is sufficient only when an existing factory can open the
data. A new primary format normally needs:

1. an integer-indexed reader implementing length, `reader[index]`, and resource
   cleanup;
2. `SampleRecord` values with stable keys, restore keys, shard identity, and
   source provenance;
3. a factory derived from the narrowest suitable base, usually
   `BaseSingleFileDatasetFactory` or `BaseManifestShardListDatasetFactory`;
4. format-specific `_build_reader` and `load_sample` implementations;
5. `as_file_store()` only if key lookup, joins, mounts, or primary auxiliary
   access are supported;
6. a stable, serializable `config()` used for diagnostics and checkpoint
   migration identity;
7. preparation or index-generation code when ordinal access cannot be derived
   cheaply from the source file.

Let `BaseIndexedDatasetFactory.build()` retain responsibility for worker
partitioning, filtering, shuffle, and `DatasetSampler`. Reimplementing those
layers in a format reader risks changing iteration and checkpoint semantics.

The primary reader is ordinal: it receives integer sample positions. A
`FileStore` is key-based random access. Supporting one does not automatically
provide the other.

## Prepared Manifests

For a directory of shards, prefer a prepared manifest over inventing another
directory detector. Use
`megatron.energon.flavors.common.manifest.write.write_manifest_dataset_metadata`
with `ShardInfo` entries and a `dataset_definition` naming the factory.
The standard manifest provider will select that factory from `.nv-meta`.

This keeps split metadata, exclusions, format versioning, and preparation
checks in the established layout. See
[Programmatic Data Preparation](data_prep_api.md) for the writer entry point.

## Testing

Cover at least:

- recognized and competing path shapes;
- provider priority and registration order;
- direct ordinal reads and stable sample keys;
- worker partitioning, filtering, and empty shards;
- exact mid-stream save and restore;
- stale, missing, or corrupt indexes;
- local and remote paths where the format promises both;
- `FileStore` lookup and provenance when supported.

The contributor-level contracts and full checklist are in
{ref}`dataset-formats`. Changes that alter existing ordering or metadata must
also follow {ref}`compatibility`.
