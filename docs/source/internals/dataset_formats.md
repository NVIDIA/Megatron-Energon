<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(dataset-formats)=

# Dataset Formats and Factories

A dataset flavor connects persistent files to Energon's common runtime pipeline. The flavor is responsible
for recognizing a path, loading format-specific metadata, and producing records. The base indexed factory
then supplies worker partitioning, filtering, shuffling, restore keys, and mapping into user samples.

## Core factory contracts

`BaseCoreDatasetFactory` is the common leaf contract. Implementations provide:

- `build(worker_rotation_offset, part_filter)`, returning a `SavableDataset`;
- `as_file_store()`, when the format supports key-based access;
- `__len__`, `__sample_type__`, `paths`, and `tags`.

`BaseIndexedDatasetFactory` is the usual base for a finite, indexable format. Its `build` implementation:

1. merges the caller's part filter with the factory filter;
2. applies a prepared filter index, if configured;
3. assigns virtual shards to workers;
4. asks the flavor to build an `IndexedSampleReader`;
5. wraps the reader in `DatasetSampler`;
6. maps each `SampleRecord` through the configured sample loader.

Subclasses normally implement `_build_reader`, `load_sample`, and `as_file_store` rather than recreating
this pipeline.

Two narrower bases cover common storage layouts:

- `BaseSingleFileDatasetFactory` represents a single data file as one virtual shard.
- `BaseManifestDatasetFactory` recognizes a prepared `.nv-meta` dataset and loads its dataset metadata.
- `BaseManifestShardListDatasetFactory` additionally loads a manifest shard list for the selected split.

Use the narrowest base that already models the format. Reimplementing worker assignment or filtering in a
leaf factory creates a second compatibility surface.

## Readers and records

An `IndexedSampleReader` has a deliberately small interface:

- `len(reader)` returns its ordinal sample count;
- `reader[index]` accepts an integer and returns `SampleRecord | None`;
- `close()` releases owned resources.

`SampleRecord` carries the format payload plus Energon metadata. In particular:

- `__key__` identifies the sample within its source;
- `__shard__` identifies its shard;
- `__restore_key__` is the structural key used for sample restoration;
- `__sources__` records provenance.

A reader may return `None` for a filtered or invalid ordinal only where its caller supports that behavior.
Do not use string keys on `IndexedSampleReader`; string lookup belongs to `FileStore` and part readers.

## Key and part access

`FileStore` retrieves `(data, SourceInfo)` for a string key and reports the underlying path. Part-aware
readers extend that model to enumerate or retrieve named parts. These interfaces support auxiliary media,
joins, mounts, and crude-dataset access.

When adding file-store support:

- use the same canonical sample keys as indexed iteration;
- populate `SourceInfo` so failures and output samples retain provenance;
- keep file handles worker-local and implement `close` where the reader owns resources;
- define missing-key and missing-part behavior consistently with existing stores.

## Path detection

`DatasetFactoryProvider` separates path detection from factory construction. Providers are registered with
an integer priority:

| Priority | Meaning |
| --- | --- |
| `0` | Explicit, first-choice formats. |
| `100` | Single-file formats. |
| `200` | Prepared manifest formats. |

Lower numbers are tried first. Providers with the same priority retain registration order. A detector
should be cheap and specific: a broad detector at a high precedence can silently claim paths belonging to
another format.

Importing a flavor module must register its provider before detection is used. If a new format is public,
also expose its factory through the intended API module.

## Adding a dataset flavor

Use this checklist:

1. **Persistent contract:** define files, metadata, versioning, keys, and sample ordering.
2. **Preparation:** implement or extend the preparator that writes indexes and metadata. Define when a
   cached index is stale and must be rebuilt.
3. **Reader:** implement integer ordinal access, length, resource cleanup, and `SampleRecord` provenance.
4. **Factory:** subclass the appropriate indexed base and implement only the format-specific hooks.
5. **Filtering:** verify `part_filter`, exclusions, and prepared `FilterIndex` translation.
6. **Key access:** implement `FileStore` or part readers if joins, mounts, or auxiliary lookup require it.
7. **Detection:** register a provider at the least surprising priority and test competing path shapes.
8. **Configuration:** ensure `config()` contains stable leaf identity fields used by diagnostics and
   checkpoint migration.
9. **Exports and docs:** expose the supported API and document preparation and recipe syntax.
10. **Tests:** cover preparation, direct reading, worker splitting, filtering, save/restore, detection, and
    stale or corrupt metadata.

Relevant tests include `test_dataset_factory_resolver.py`, the format-specific dataset tests, and
`test_filter_index.py`. A format that changes existing sample order also needs the iteration-order tests
listed in {ref}`compatibility`.

## Configuration is descriptive

A factory's `config()` result is a recursively serializable description used for logging and checkpoint
identity matching. Stable fields such as `_path`, `split_part`, `subset`, and `filter` are significant.

Configuration should not contain open readers or runtime progress. Runtime progress belongs to
`SavableDataset` state. If a field is required to reconstruct a factory but should not distinguish two
logical dataset leaves, document that choice and add a checkpoint-migration test.
