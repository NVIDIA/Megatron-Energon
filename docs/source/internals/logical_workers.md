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

### Delivery order

Ranks must advance their dataloaders synchronously. Within each logical group, stride
offsets follow delivery order: local worker index first, then data parallel rank. The
logical partition and its physical members are unchanged. Ordering is guaranteed within
each logical stream; streams may be interleaved in global batches.

For three ranks, two subprocess workers per rank, and two logical streams A and B:

| Synchronized call | Rank 0 | Rank 1 | Rank 2 |
| --- | --- | --- | --- |
| 0 | A0 | A1 | B0 |
| 1 | A2 | B1 | B2 |
| 2 | A3 | A4 | B3 |
| 3 | A5 | B4 | B5 |

## Saving and restoring logical workers

Ordinary `save_state_rank()` and `save_state_global()` save every physical worker,
including its stream position and a shared iteration count. The next worker is derived
from that count. Use the corresponding
`restore_state_rank()` or `restore_state_global()` with the same physical assignment.

For a checkpoint that can be restored with another physical fanout, set `logical_workers`
explicitly and use the separate logical save/restore methods. **Save the last contributor
to each logical group at the current iteration, and restore it into the same logical group.**
Group ids identify logical data streams and remain stable when the physical topology changes.
Each entry contains a logical pipeline snapshot, RNG state, and replay distance in logical
outputs. It contains no source rank, physical worker id, fanout, or stride assignment.
Save and restore treat the returned state as an opaque object; callers need not construct
or modify checkpoint records.

### Save and restore without gathering

Each rank can save its own contribution without distributed collectives:

```python
state = loader.save_state_logical_rank()
if state is not None:
    # state.worker_states maps global logical group ids to their checkpoints.
    torch.save(state, checkpoint_file)
    # Record each group id in state.worker_states -> checkpoint_file in your manifest.
```

`save_state_logical_rank()` returns `None` on ranks that own no group's last contributor.
A rank can own the last contributors of several groups; its returned state contains
all of them. Owning ranks return a checkpoint even before consuming any batches.
`loader.get_logical_worker_group_ids(for_save=True)` lists the groups this rank writes.

Save after the same number of dataloader calls on every rank; zero calls is also valid.
There is no requirement to complete a local worker round. Checkpoint ownership can change
after every call: in the example above, after one call rank 1 writes A and rank 2 writes B;
after two calls rank 0 writes A and rank 2 writes B. Query the save group ids at each save.
Prefetched batches do not count as consumed batches.

On a fresh loader under the new topology, discover which groups this rank needs, use your
manifest to load their checkpoint files, and restore them:

```python
group_ids = loader.get_logical_worker_group_ids()
files = {group_to_file[group_id] for group_id in group_ids}
states = [torch.load(path, weights_only=False) for path in files]
loader.restore_state_logical_rank(states)
```

Restore accepts one logical state or a sequence. Each required group must be present;
extra groups in the loaded files are allowed. Deduplicate file paths before loading a
sequence so each group occurs once. Missing or duplicate group checkpoints raise `ValueError`.
The checkpoint's original physical rank is not its restore destination: use logical group ids.

For example, with two logical groups and six ranks without subprocess workers, workers `2` and `5` save
groups `0` and `1`. After changing to four physical workers, workers `0,1` restore group
`0` and workers `2,3` restore group `1`. With no subprocess workers, these worker ids are
also the data parallel rank ids.

### Gather and restore globally

Alternatively, gather all logical checkpoints to one rank:

```python
state = loader.save_state_logical_global()  # Destination defaults to global rank 0.
# All ranks participate; only the destination rank receives the state to serialize.

# On fresh loaders, load state only on data parallel rank 0:
loader.restore_state_logical_global(state, src_rank=0)
```

Only the selected logical entries are gathered; other physical-worker states are not
included in the global checkpoint. All ranks must have made the same number of dataloader calls. Global saving checks their
iteration counts agree and selects the last contributor in each group. Without `src_rank`,
load the global state on every rank before restoring.

### Restore guarantees

The new configuration may use another number of ranks or subprocess workers, provided the global physical
worker count remains a multiple of the unchanged `logical_workers`. Micro batch size
and the dataset pipeline must remain unchanged. Logical stream order is preserved;
physical output ownership and rank interleaving can change. The overall checkpoint stores the
number of completed `next()` calls in the current dataloader iteration once. Restore computes the next local worker as that count modulo
the current number of workers per rank and assigns each logical stream's next outputs in
the current delivery order. No scheduling or physical-topology arguments are required.

With explicit `logical_workers`, sample indices used for stateless transform seeds
also follow logical stream identities, so changing physical fanout preserves seeded
sample transformations. Without explicit logical workers, indexing is unchanged.
Hooks that depend on physical worker identity or `current_batch_index` are not
topology independent; use sample-local stateless randomness for reproducible transforms.

Sparse checkpoints first replay the saved number of full logical outputs, then distribute
new outputs using the current fanout. The logical methods select the last contributors and route their checkpoints
to the correct groups; do not manually substitute another physical member's checkpoint.

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

The loader factories and constructors accept `epochize` and `limit`.
Epochization continues the worker stream between segments; a limit resets it when an
iteration completes. Standalone `EpochizeDataset` and `LimitDataset` provide the
corresponding behavior when composing processing datasets directly.

`IterationController` owns the loader's boundaries and their progress. It counts delivered
outputs and reports whether the source must reset; it has no dataset, worker, or queue references.
The loader creates one stride wrapper when logical workers are configured, followed by
sample logging when enabled by `worker_log_level`. Task encoders build the processing
pipeline. Output logging observes retained outputs after striding.
The loader owns worker startup, prefetch shutdown, and source resets. The execution order is:

```text
logical pipeline -> checkpoint replay -> striding -> worker queues -> epoch/limit -> caller
```

The loader retains the merged worker iterator across epochized segments.
Its scheduling count continues across these boundaries, even when an epoch length is not
divisible by the number of workers. Logical and physical loader checkpoints save epoch/limit
progress separately, so replay does not consume the remaining public epoch budget.

A limit discards prefetched outputs, shuts down the old workers, resets the source,
and starts a new stream on the next iteration. A checkpoint taken at the final yield preserves
the pending boundary; its restored iterator finishes that epoch before starting the next one.


## Checkpoint and worker responsibilities

The checkpoint schemas and conversion/routing helpers in `savable_loader.py` handle logical
snapshots, group merging, and validation independently of loader execution.

The worker wrappers in `savable_loader.py` own per-worker iteration, checkpoint history,
and snapshot installation.
The main-process and subprocess paths use the same physical checkpoint representation:
a dataset snapshot, its RNG state and output counter, plus a physical replay offset.
Pending logical restores remain explicitly distinguishable because their offsets count full
logical outputs instead.

Snapshot installation resets the pipeline once and copies mutable dataset state once before
restoring it. This keeps the caller's checkpoint and retained history unchanged during replay
or later iteration. RNG restoration belongs to this same worker operation. The stride wrapper
receives its logical cursor and transient replay/assignment information directly.


## Finite-source exhaustion

A worker can reach EOF through prefetch before the caller consumes its last output.
The worker retains its post-EOF checkpoint separately from ordinary yield checkpoints,
even though both can have the same output counter. Save uses the post-EOF state only
after the main loader has observed source EOF. Saving at the final yield therefore
preserves the pending `StopIteration`; saving after it resumes the next pass.

Historical replay stops at source EOF, including when every remaining output was replayed.
A new worker iterator records a fresh origin checkpoint so replay does not cross into
a previous pass.

After source exhaustion, a loader restored with rotated worker order rebuilds the workers
in ordinary order using their end-of-pass snapshots. Each physical identity retains its
advanced dataset and RNG state. Ordinary unrotated iteration keeps its persistent workers.
Limits reset the source explicitly when an iteration completes.
