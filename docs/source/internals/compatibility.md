<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(compatibility)=

# Compatibility and Review Matrix

Energon has several independent compatibility surfaces. Review a change against each surface instead of
using "API compatible" as a synonym for "non-breaking."

## Compatibility surfaces

| Surface | Examples of a breaking change |
| --- | --- |
| Public Python API | Removing or renaming an exported class, parameter, return field, or import path. |
| Prepared dataset config | Changing required `.nv-meta` files, index encoding, metadata fields, or key interpretation. |
| Recipe config | Changing YAML names, defaults, nesting behavior, subset meaning, or split resolution. |
| Checkpoint | Changing saved classes, fields, wrapper topology, restore keys, or required buffer state. |
| Iteration order | Changing sharding, shuffle, RNG consumption, blend selection, skip advancement, or permutation. |

A change can belong to more than one row. For example, moving a shuffle stage may be both checkpoint and
iteration-order breaking without changing a public signature.

## Public versus internal API

Names exported by the top-level `megatron.energon` package form the primary public Python API. Documented
extension modules, such as the dataset factory provider interface, are also supported extension points even
if every implementation class is not exported at the top level. Other package internals may change, but
their effect on configuration, checkpoints, and ordering still requires review.

The deprecation module supports compatibility aliases for moved modules and symbols through import hooks
and module attribute lookup. When moving a supported name:

1. add and document the new location;
2. keep an old import mapping when compatibility is intended;
3. emit the standard deprecation warning;
4. test both the new and deprecated imports;
5. state the removal window in release notes;
6. mark the pull request if the old path is removed immediately.

`MetadatasetV2` remains a compatibility alias for the recipe system. This does not imply support for
obsolete v1 configuration.

## Reproducible iteration

For a fixed supported configuration and seed, iteration order is an explicit compatibility surface.
Important building blocks include:

- `WorkerRng`, whose state is saved and whose discrete choice avoids depending on changing framework
  implementations;
- `FeistelPermutation`, which creates a memory-constant bijection and uses cycle walking for non-power-of-two
  ranges;
- worker-to-shard assignment and logical-worker stride mapping;
- the number and position of random calls in wrappers and user hooks.

Changing a Feistel round function, number of rounds, seeding, or cycle-walking behavior changes the
permutation even if every sample still appears once. Treat such a change as iteration-order breaking.

Avoid incidental RNG calls in logging, validation, or skipped paths. A random decision that affects the
stream must use saved worker state.

## Checkpoint compatibility

Adding a field to `_savable_fields`, changing wrapper child structure, or changing restore-key shape can
break exact restoration. Recipe migration may still be able to preserve matched leaf progress, but that
does not make the exact checkpoint format compatible.

If an intentional migration is possible, add it explicitly and test both old-state input and new-state
round trips. Never silently assign state after an ambiguous leaf match.

## Review and test matrix

Use the rows relevant to the change:

| Area changed | Required focused tests | Also inspect |
| --- | --- | --- |
| Wrapper state or topology | Save/restore tests in `test_recipe.py`; affected wrapper tests | `test_checkpoint_resume.py`, restore keys, reset, skip mode |
| Recipe nodes or identities | `test_recipe.py`, `test_checkpoint_resume.py` | nested splits, typed subsets, ambiguous matches |
| Worker assignment or RNG | current-batch-index tests in `test_dataset.py`; `test_logical_workers.py` | fanout, distributed rank, exact ordering |
| Dataset factory or format | resolver and format-specific tests; `test_filter_index.py` | preparation, stale indexes, `FileStore`, provenance |
| Prepared metadata | preparation and format reader tests | old metadata fixtures and explicit versioning |
| Packing or blending | packing, grouping, and blending tests | mid-buffer restore, metrics, skip mode, RNG |
| Public imports | API and deprecation import tests | `__all__`, docs, typing |
| Documentation | `just docs` and a clean Sphinx build | warnings, links, generated API pages |

After focused tests, run the full suite with `just test`. Run `just check` before review and use `just fix`
to apply the repository's standard formatting.

## Pull request annotations

Call out each applicable class of breakage in the pull request description:

- `CHECKPOINT BREAKING CHANGE`
- `ITERATION ORDER BREAKING CHANGE`
- `API BREAKING CHANGE`
- `DATASET CONFIG BREAKING CHANGE`
- `RECIPE CONFIG BREAKING CHANGE`

Explain the affected versions or configurations, whether a migration exists, and which tests demonstrate
the new behavior. Checkpoint compatibility has its own versioning implications; the other breaking
surfaces normally require a major release.

## Reviewer questions

A reviewer should be able to answer:

1. Which compatibility rows can this change affect?
2. Does construction config remain stable and serializable?
3. Can state be saved with input consumed but output still pending?
4. Does reset produce the same state as a new construction?
5. Are random calls, worker mapping, and sample indexes unchanged?
6. Are old imports and configuration accepted, migrated, or rejected clearly?
7. Does remote storage behave like local storage where promised?
8. Do tests assert order and state, rather than only comparing sample sets?
