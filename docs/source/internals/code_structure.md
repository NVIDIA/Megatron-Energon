<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(code-structure)=

# Code Structure

This section describes the boundaries inside Energon that contributors need to preserve. It is not a
complete inventory of classes. Instead, it follows a dataset from configuration to worker iteration and
points to the extension contracts at each layer.

## Package map

The implementation lives under `src/megatron/energon`. Its main layers are:

| Package | Responsibility |
| --- | --- |
| `recipe` | Parse and resolve recipe graphs, references, blends, joins, subsets, and split overrides. |
| `flavors` | Detect prepared dataset formats and construct their readers and runtime datasets. |
| `wrappers` | Compose iteration behavior such as mapping, batching, blending, packing, epochizing, and striding. |
| `task_encoder` | Build the user-facing loading pipeline and invoke application-specific encoding hooks. |
| `checkpoint` | Save exact loader state and migrate saved progress to a changed recipe graph. |
| `epathlib` | Provide local and remote path access plus mapped-array helpers. |
| `cache` | Store generated indexes and cache metadata. |
| `media`, `decoders`, `transforms` | Decode and transform sample payloads. |
| `cli`, `tools` | Prepare datasets and expose operational utilities. |

## From a recipe to samples

The high-level construction flow is:

```text
YAML / Python recipe
        |
        v
Recipe graph resolution
        |
        v
Dataset factory detection and construction
        |
        v
Indexed reader + DatasetSampler
        |
        v
cook -> blend -> shuffle -> encode -> pack -> batch -> stride -> epochize
        |
        v
SavableDataLoader workers
```

`load_dataset` parses a recipe or constructs a dataset loader for a prepared dataset path.
`Recipe.post_initialize` resolves references relative to the recipe root. `Recipe.get_datasets` then builds
the runtime leaf factories, while `Recipe.traverse` can inspect the same graph without building an
iterable dataset.

`TaskEncoder` owns the standard training pipeline. Some stages are omitted when the corresponding hook is
not configured, and grouped packing creates a branch per packing group before blending the packed
branches. The ordering of these stages is part of reproducibility: moving a random or stateful operation
can change both checkpoint state and sample order.

## Two forms of dataset access

Energon has two distinct random-access interfaces. They should not be conflated:

1. **Ordinal iteration** uses an `IndexedSampleReader`. `reader[index]` accepts an integer sample ordinal
   and returns a `SampleRecord`. A `DatasetSampler` maps the worker's iteration order to those ordinals.
2. **Key and part lookup** uses `FileStore`, `SamplePartReader`, and related readers. These interfaces
   locate a sample or auxiliary part by a string key and are used by crude datasets, joins, mounts, and
   restore-by-key paths.

A format can support ordinary indexed training without exposing all file-store capabilities. Conversely,
adding a key lookup path does not define the dataset's iteration order.

## Construction and runtime objects

Dataset factories are configuration-time objects. Their `build` method creates `SavableDataset` objects
that execute inside workers. A factory's `config()` output describes how it was constructed; it is also
used to identify leaves during checkpoint migration. It is not a replacement for the mutable state saved
by `SavableDataset.save_state()`.

Runtime behavior is assembled from wrappers. `BaseWrapperDataset` records its child datasets and
propagates common operations such as reset and skip mode. A wrapper that owns mutable progress must also
declare and restore that progress. See {ref}`savability` before adding a wrapper.

## Worker ownership

The loader distinguishes physical workers from logical workers. Dataset partitioning, random number
generation, and saved worker state use logical worker identities. When multiple physical workers share a
logical worker, a `StrideDataset` assigns different outputs to each physical worker while advancing the
same logical stream. See {ref}`logical-workers` for the mapping and skip-mode contract.

Readers, open files, and mapped arrays belong to worker processes. Keep resource handles out of serialized
checkpoint state and close readers that own handles. Path and storage rules are described in
{ref}`epath-storage`.

## Stability boundaries

Changes can affect different compatibility surfaces:

- the public Python API;
- recipe YAML and prepared-dataset metadata;
- the exact checkpoint structure;
- the iteration order for a fixed seed and configuration.

A refactor can preserve one surface while breaking another. For example, replacing a wrapper without
changing its public constructor may still change saved state or iteration order. Use the review and test
matrix in {ref}`compatibility` when modifying the pipeline.

## Where to continue

- {ref}`dataset-formats` explains factory detection and the checklist for a new format.
- {ref}`savability` defines the state and restore contract for runtime datasets.
- {ref}`recipe-checkpoint` covers recipe graphs and topology-aware checkpoint migration.
- {ref}`packing-blending` covers the most stateful composition layers.
- {ref}`contribution-guidelines` contains the local development workflow.
