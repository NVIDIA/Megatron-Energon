<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(savability)=

# Savability and Restore Invariants

Energon checkpoints the data pipeline, not just a numeric sample counter. Every stateful runtime dataset
must therefore define enough state to continue the same worker stream and enough structure to reject an
incompatible exact restore.

## State, configuration, and restore keys

These three concepts serve different purposes:

| Concept | Purpose |
| --- | --- |
| `save_state()` | Mutable per-worker progress needed to continue iteration exactly. |
| `config()` | Descriptive construction information used for inspection and migration identity. |
| restore key | Structural address used to reconstruct a previously emitted sample. |

Do not derive mutable iteration state from `config()`, and do not assume a restore key is an ordinal index.
A wrapper may add its own component to a child's restore key so that the key remains meaningful through a
composed graph.

## `SavableDataset`

A `SavableDataset` declares simple mutable fields in `_savable_fields`. The base implementation saves a
class marker and recursively saves values that implement the savable protocol; other declared values are
deep-copied. Exact restore verifies the class and the expected fields before restoring them.

A complete implementation also defines:

- `reset_state_own()` for a clean restart of its own state;
- `len_worker()` and `worker_has_samples()` for worker-local availability;
- `set_skip_mode()` when work can be elided for discarded samples;
- `config()` for a stable, serializable description;
- `can_restore_sample()` and sample restoration where the dataset supports it.

State is per logical worker. The loader merges or distributes worker state at its process boundary; an
individual dataset should not invent a second global-worker aggregation scheme.

## Wrapper datasets

`BaseWrapperDataset` stores child datasets in a tuple and implements the common recursive mechanics:

- all children must use the same `WorkerConfig`;
- child state is saved under `datasets`;
- restore verifies the number of children;
- reset and skip mode propagate to children;
- a multi-child restore key records which child produced the sample.

A new wrapper should use those mechanics rather than manually walking a private child attribute. If it owns
additional progress, declare that progress in `_savable_fields` and reset it in `reset_state_own()`.

Review a new wrapper against this checklist:

1. Are all iterable children present in `datasets` in a stable order?
2. Does saved state include every cursor, buffer, random generator, and pending item?
3. Does reset clear both the wrapper's own state and nested savable helpers?
4. Are `len_worker` and `worker_has_samples` correct for empty and exhausted children?
5. Is skip mode propagated, handled locally, or deliberately blocked?
6. Do emitted restore keys contain enough structure to route sample restoration?
7. Does `config()` describe construction without including open handles or mutable progress?
8. Do tests save and restore in the middle of any buffer, group, or batch?

## Randomness and sample indexes

Use `WorkerRng` for stateful worker randomness. Its state is part of the checkpoint, and its choice
implementation is designed to remain stable across supported PyTorch versions.

`SampleIndex` and related helpers track nested sample-number scopes. Calling `skip()` must advance the same
logical index that ordinary execution would have consumed. This is required when `StrideDataset` discards
outputs while multiple physical workers fan out one logical stream.

Do not replace these helpers with an unsaved local counter or a module-global random generator.

## Skip mode

Skip mode means "advance the stream while omitting explicitly safe computation." It does not mean "ignore
state." A skipped transform must consume the same sample index and leave downstream state aligned with a
fully evaluated stream.

A callable marked `skip_safe` must not:

- mutate state that later samples observe;
- perform required validation or error detection;
- create a restore-key component that later stages need;
- consume untracked randomness;
- change how many samples the stage emits.

Selectors are not skip-safe because their decisions define which inputs form an output. For details on
physical/logical worker fanout, see {ref}`logical-workers`.

## Buffered stages

Buffers make restore bugs easy to hide. Packing, grouping, shuffling, and batching may have consumed more
input than they have emitted. Their state must preserve both the input position and the pending content.

Test at boundary cases:

- immediately before and after an output is emitted;
- with a partially filled buffer or batch;
- after one child of a multi-child wrapper is exhausted;
- while a sample has been pushed back for the next pack;
- after reset and after restore into a newly constructed pipeline.

A test that only checkpoints at an epoch boundary does not exercise this contract.

## Exact restore versus migration

The ordinary loader restore path expects the same runtime topology and validates class and field structure.
Recipe-aware checkpoint migration is a separate operation that constructs a fresh current topology and
copies compatible saved leaf progress into it. It must not weaken the exact-restore assertions inside
individual datasets.

See {ref}`recipe-checkpoint` for the identity and matching rules, and
[Saving and Restoring](../basic/save_restore.md) for the user-facing checkpoint API.
