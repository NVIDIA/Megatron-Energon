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
```

Physical membership remains contiguous by id. Within each group, sort its members by
`(local_worker_id, rank)` to assign stride offsets. This is the order in which synchronized
dataloader calls deliver outputs, and preserves order within each logical stream.

With `L=2` and `P=6`, physical workers `0,1,2` consume different strides of logical stream `0`, and workers
`3,4,5` consume different strides of logical stream `1`.

The physical worker count is the distributed world size when loader workers are disabled, otherwise it is
`world_size * num_workers`.

## `StrideDataset`

With explicit `logical_workers`, the loader wraps the pipeline in `StrideDataset`, including fanout one.
Each physical worker advances its full assigned
logical stream but emits only positions matching its stride offset. Positions assigned to sibling physical
workers are evaluated under skip mode where possible.

Physical snapshots of `StrideDataset` save only the next output index in the full logical stream,
including outputs discarded by striding. Its `export_logical_state()` returns this logical cursor
and the inner pipeline state, without serializing the stride wrapper itself. Outer execution
wrappers forward this operation. The loader retains a direct reference to the stride wrapper and
restores the inner pipeline and cursor under the current assignment.
During iteration the wrapper
activates the logical worker identity so nested datasets use the correct
partition, random stream, sample indexes, and restore keys.

Logical checkpoints retain the consumed checkpoint of each group's last contributor at the
current synchronized iteration, keyed by global logical group id. This member depends on the
local worker phase; it need not have the largest fixed stride offset. Each returned
`LogicalWorkerCheckpoint` contains a logical pipeline snapshot with its cursor and RNG state,
and an offset measured in full logical outputs. Source worker ids, worker counts, stride and
stride offsets are not serialized. Ordinary physical checkpoints require an unchanged assignment.

`WorkerConfig.logical_worker_last_physical_worker(group, iteration_index)` computes the last
contributor after the given number of completed dataloader calls. Only the phase modulo the
local worker count matters, so the loader uses its iteration count. At zero calls,
the end-of-round member supplies the initial checkpoint. Both logical save and restore use
this method. `logical_worker_next_offset(worker, previous_worker)` computes how many outputs
to skip from just after the previous contributor to this worker's next output. With no previous
worker, it gives the initial offset; with the same worker, it gives `stride - 1`.

Historical worker checkpoints can lag the consumed boundary. Saving converts the physical
output offset into a logical replay distance and exports the logical snapshot. An initial
snapshot also accounts for the selected member's first output offset. The loader passes the
previous contributor and replay distance directly to its stride wrapper when installing the
checkpoint. There is no shared restore context in `WorkerConfig`.

Before striding begins, the wrapper advances the full logical stream by the replay distance
under skip mode, allowing skip-safe pure work to be omitted. It does not need the old stride.
It then calculates the first retained offset using the current assignment and iteration count.
Subsequent outputs use the current stride. Replay counts and pending skips are transient
iteration bookkeeping; only the logical source cursor is a savable stride field.

Checkpoints are taken before iteration or at output yields. After replay, the replacement worker
checkpoint is taken after the first new output, rather than at the intermediate replay boundary.
Ordinary physical continuation skips `current_stride - 1` outputs; logical replay skips none.
These skips are derived when iteration resumes. Replay offsets are
loader progress, not permission to seek backward from a later sibling's state.

`save_state_logical_rank()` returns only groups whose last member is local, or `None` if there are
none. It requires synchronized ranks and makes no collectives. Partial local worker rounds are valid.
`get_logical_worker_group_ids()` identifies the groups to load; `for_save=True` identifies those to
write. `restore_state_logical_rank()` merges the supplied states by group id and maps each local
physical member to its group's checkpoint. The global variants gather and route all groups.
The overall `LogicalDataLoaderState` stores group checkpoints, logical worker count, micro batch
size, one iteration count and rank-local epoch/limit progress. The next worker is derived modulo the current local worker count,
without saving source topology or next-worker ids. A rank owning several groups returns several
entries in this one opaque object; callers can restore it directly or pass a sequence of such
objects loaded from group checkpoint files.

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

- fanout one, with the same wrapper topology as higher fanout when logical_workers is explicit;
- at least two fanout values greater than one;
- exact save/restore after different stride offsets;
- empty and exhausted logical streams;
- stateful hooks that must still run;
- restore keys from retained samples;
- equality with the un-fanned logical stream after interleaving physical outputs.

Run the general iteration-order tests as well. A locally plausible output set is insufficient: ordering and
saved progress are part of the contract.


## Loader iteration boundaries

Dataset builders express `virtual_epoch_length` and validation limits with outer
`EpochizeDataset` / `LimitDataset` wrappers. Both loaders extract these outer policies and
apply them after merging worker outputs. The execution order is:

```text
logical pipeline -> checkpoint replay -> striding -> worker queues -> epoch/limit -> caller
```

The loader retains the merged worker iterator across virtual epochs and non-resetting limits.
Its scheduling count continues across these boundaries, even when an epoch length is not
divisible by the number of workers. Logical and physical loader checkpoints save epoch/limit
progress separately, so replay does not consume the remaining public epoch budget.

A resetting limit discards prefetched outputs, shuts down the old workers, resets the source,
and starts a new stream on the next iteration. A checkpoint taken at the final yield preserves
the pending boundary; its restored iterator finishes that epoch before starting the next one.
