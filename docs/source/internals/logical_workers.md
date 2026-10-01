<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(logical-workers)=

# Logical Workers and Skip Mode

Logical workers decouple dataset partitioning from the number of physical loader processes. This allows a
run to use more physical workers while preserving the logical data streams used for sharding, random
number generation, and checkpointing.

## Mapping physical to logical workers

Let:

- `P` be the global number of physical workers;
- `L` be the configured number of logical workers;
- `F = P / L` be the fanout.

`WorkerConfig` requires `L > 0`, `L <= P`, and `P` to be divisible by `L`. If no logical count is supplied,
`L = P`.

For a global physical worker id `p`:

```text
logical_worker_id = p // F
stride_offset     = p % F
```

With `L=2` and `P=6`, physical workers `0,1,2` consume different strides of logical stream `0`, and workers
`3,4,5` consume different strides of logical stream `1`.

The physical worker count is the distributed world size when loader workers are disabled, otherwise it is
`world_size * num_workers`.

## `StrideDataset`

When `P > L`, the pipeline is wrapped in `StrideDataset`. Each physical worker advances its full assigned
logical stream but emits only positions matching its stride offset. Positions assigned to sibling physical
workers are evaluated under skip mode where possible.

`StrideDataset` intentionally has no independent iteration cursor to save: the nested logical stream owns
progress. During iteration it activates the logical worker identity so nested datasets use the correct
partition, random stream, sample indexes, and restore keys.

This design means fanout is not ordinary modulo sharding at the file-reader boundary. All physical workers
for one logical worker represent one coordinated logical stream.

## Skip-safe work

The standard pipeline may elide a callable for a discarded stride position only when that callable is
marked `skip_safe`. Supported hook sites include cooking, encoding, pre-encoding, post-encoding, selected
sample packing, batching, and batch encoding.

A callable is skip-safe only if omitting it:

- has no observable side effect;
- skips no required validation;
- consumes no state or untracked random numbers;
- does not change sample count or routing;
- leaves sample-index and restore-key accounting correct.

Selection is never skipped because its decisions define which input samples belong to an output. Packing
can skip only the stages whose safety has been established independently; see {ref}`packing-blending`.

Marking a stateful transform skip-safe can produce a failure that appears only when logical fanout is
greater than one. Default to not skip-safe when uncertain.

## State and reproducibility

Dataset splitting and state use the logical worker id, not the physical id. Consequently:

- changing physical fanout need not redefine the logical dataset partitions;
- changing the logical worker count does redefine those partitions;
- every discarded position must still advance logical state exactly once;
- randomness used by a skipped callable must not leak into the shared stream.

A change to the mapping, stride order, or skip advancement is an iteration-order change and can also be a
checkpoint change.

## Testing changes

`test_logical_workers.py` covers mapping validation, skip-safe execution, and non-skip-safe execution.
Changes to this area should additionally test:

- fanout one, where `StrideDataset` is absent;
- at least two fanout values greater than one;
- exact save/restore after different stride offsets;
- empty and exhausted logical streams;
- stateful hooks that must still run;
- restore keys from retained samples;
- equality with the un-fanned logical stream after interleaving physical outputs.

Run the general iteration-order tests as well. A locally plausible output set is insufficient: ordering and
saved progress are part of the contract.
