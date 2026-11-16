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

Choose the checkpoint API according to the restore requirements:

- `save_state_rank()` / `restore_state_rank()` save every local physical worker. Each
  rank can reload its own file without communication, including with fanout, provided
  rank assignments, world size, workers per rank, and logical worker count stay fixed.
  `save_state_global()` / `restore_state_global()` gather and restore a rank-ordered
  list of these physical states.
- `save_state_logical_rank()` / `restore_state_logical_rank()` save one checkpoint per
  logical group and allow changing the physical topology. The global variants,
  `save_state_logical_global()` / `restore_state_logical_global()`, gather and broadcast
  these logical states.

Without an explicit logical count, each physical worker is a separate logical worker
(`L = P`). With fanout greater than one, only one physical member contributes per group
to a logical save. The remaining instructions describe logical checkpoints.

**Save the last contributor
to each logical group at the current iteration, and restore it into the same logical group.**
Group ids identify logical data streams and remain stable when the physical topology changes.
Each entry contains a logical pipeline snapshot, RNG state, and replay distance in logical
outputs. It contains no source rank, physical worker id, fanout, or stride assignment.
Save and restore treat the returned state as an opaque object; callers need not construct
or modify checkpoint records.

### Save and restore without gathering

For one file per logical worker, see
{ref}`save-restore-per-logical-rank`. The example below instead stores each rank's
contribution in one file and uses a manifest to locate its logical groups.

Each rank can save its own contribution without distributed collectives:

```python
states = dict(loader.save_state_logical_rank())
if states:
    torch.save(states, checkpoint_file)
    # Record each logical group id in states -> checkpoint_file in your manifest.
```

`save_state_logical_rank()` returns `(logical_id, state)` pairs, or an empty tuple on
ranks that own no group's last contributor. Each state includes one group's checkpoint,
the total logical worker count, micro batch size, and iteration index. A rank can own
the last contributors of several groups. Owning ranks return checkpoints even before consuming any batches.
`loader.get_logical_worker_group_ids(for_save=True)` lists the groups this rank writes.

Save after the same number of dataloader calls on every rank; zero calls is also valid.
There is no requirement to complete a local worker round. Checkpoint ownership can change
after every call: in the example above, after one call rank 1 writes A and rank 2 writes B;
after two calls rank 0 writes A and rank 2 writes B. Query the save group ids at each save.
Prefetched batches do not count as consumed batches.

Synchronization requires matching delivered output counts and boundary phases, not just
calling `next()` at the same time. The loader does not coordinate EOF across ranks. A finite
partition can end while another rank still has outputs; `epochize` and `limit` only cap
iteration and do not pad or balance those partitions. For synchronized training, use
repeated nonempty streams with a common output budget, or ensure the finite pipelines
produce equal output counts. Supporting uneven finite streams requires an explicit
application policy for padding, dropping outputs, or coordinating completion.

On a fresh loader under the new topology, discover which groups this rank needs, use your
manifest to load their checkpoint files, and restore them:

```python
group_ids = loader.get_logical_worker_group_ids()
files = {group_to_file[group_id] for group_id in group_ids}
states = {}
for path in files:
    states.update(torch.load(path, weights_only=False))
loader.restore_state_logical_rank(states)
```

Restore accepts a mapping of logical ids to states or a sequence of `(logical_id, state)`
pairs. Each required group must be present; extra groups in the loaded files are allowed.
Deduplicate file paths before loading so each group is loaded once. Missing or duplicate
group checkpoints in a sequence raise `ValueError`.
The checkpoint's original physical rank is not its restore destination: use logical group ids.

For example, with two logical groups and six ranks without subprocess workers, workers `2` and `5` save
groups `0` and `1`. After changing to four physical workers, workers `0,1` restore group
`0` and workers `2,3` restore group `1`. With no subprocess workers, these worker ids are
also the data parallel rank ids.

### Gather and restore globally

Alternatively, gather all logical checkpoints to one rank:

```python
states = loader.save_state_logical_global()  # Destination defaults to global rank 0.
# All ranks participate; only the destination rank receives the state to serialize.

# On fresh loaders, load state only on data parallel rank 0:
loader.restore_state_logical_global(states, src_rank=0)
```

Only the selected logical entries are gathered; other physical-worker states are not
included in the global checkpoint. All ranks must have matching delivered output counts and
boundary phases. Merging checks those values agree among the contributed states; it does
not synchronize iteration or coordinate EOF. Without `src_rank`,
load the global state on every rank before restoring.

### Restore guarantees

The new configuration may use another number of ranks or subprocess workers, provided the global physical
worker count remains a multiple of the saved logical worker count. Set `logical_workers`
to that saved count before building the dataset under the new topology. Leaving it unset
would instead default to the new physical count and redefine the logical partitions.
The dataset pipeline must remain unchanged. With subprocess workers, the micro batch size
may be reduced to a divisor of the saved size; replay distances are scaled accordingly. Logical stream order is preserved;
physical output ownership and rank interleaving can change. The checkpoint's `iteration_index`
counts outputs delivered per rank within the current worker pass, excluding prefetch and
checkpoint replay. It resets when the loader observes that pass's `StopIteration`, including
for an epochized pass. A checkpoint taken immediately after the final yield still contains
the completed pass's count; the count becomes zero after the caller advances again and
observes `StopIteration`. The underlying dataset can continue across virtual epochs while
this scheduling counter resets. Restore computes the next local worker as this count modulo
the current number of workers per rank and assigns each logical stream's next outputs in
the current delivery order. No scheduling or physical-topology arguments are required.

Sample indices used for stateless transform seeds follow logical stream identities,
including when the logical count defaults to the physical count. Changing physical
fanout with a fixed logical count preserves seeded sample transformations.
Hooks that depend on physical worker identity or `current_batch_index` are not
topology independent; use sample-local stateless randomness for reproducible transforms.

Sparse checkpoints first replay the saved number of full logical outputs, then distribute
new outputs using the current fanout. The save/restore methods select the last contributors and route their checkpoints
to the correct groups; do not manually substitute another physical member's checkpoint.

## `StrideDataset`

The loader wraps the pipeline in `StrideDataset`, including fanout one when the logical count is implicit.
Each physical worker advances its full assigned
logical stream but emits only positions matching its stride offset. Positions assigned to sibling physical
workers are evaluated under skip mode where possible.

The loader calls `StrideDataset.save_state()` directly. Ordinary recursive state contains
the stride's logical cursor, the epochize/limit progress, and the inner pipeline state.
The physical assignment and transient replay fields are excluded from `_savable_fields`.
Restore uses `stride.restore_state()` before supplying transient replay instructions.
During iteration the wrapper
activates the logical worker identity so nested datasets use the correct
partition, random stream, sample indexes, and restore keys.

Logical checkpoints retain the consumed checkpoint of each group's last contributor at the
current synchronized iteration, keyed by global logical group id. This member depends on the
local worker phase; it need not have the largest fixed stride offset. Each returned
`SavableDatasetCheckpoint` contains a logical pipeline snapshot with its cursor and RNG state,
and an offset measured in full logical outputs. Source worker ids, worker counts, stride and
stride offsets are not serialized.

`WorkerConfig.logical_worker_last_physical_worker(group, iteration_index)` computes the last
contributor after the given number of completed dataloader calls. Only the phase modulo the
local worker count matters, so the loader uses its iteration count. At zero calls,
the end-of-round member supplies the initial checkpoint. Both save and restore use
this method. `logical_worker_next_offset(worker, previous_worker)` computes how many outputs
to skip from just after the previous contributor to this worker's next output. With no previous
worker, it gives the initial offset; with the same worker, it gives `stride - 1`.

Historical worker checkpoints can lag the consumed boundary. Each snapshot already contains
only the logical pipeline and its cursor. A worker converts the physical output offset into a
logical replay distance when selecting a historical checkpoint. An initial
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
Uninterrupted continuation skips `current_stride - 1` outputs; full-stream replay skips none.
These skips are derived when iteration resumes. Replay offsets are
loader progress, not permission to seek backward from a later sibling's state.

The overall `SavableDataLoaderState` stores group checkpoints, logical worker count, micro batch
size and one iteration count within the current worker pass. Boundary progress, including a
pending StopIteration, belongs to the logical dataset tree; no separate loader boundary
field is saved. The next worker is derived modulo the current local worker count,
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

- fanout one with both implicit and explicit logical worker counts;
- at least two fanout values greater than one;
- exact save/restore after different stride offsets;
- empty and exhausted logical streams;
- stateful hooks that must still run;
- restore keys from retained samples;
- equality with the un-fanned logical stream after interleaving physical outputs.

Run the general iteration-order tests as well. A locally plausible output set is insufficient: ordering and
saved progress are part of the contract.


(loader-iteration)=

## Loader iteration boundaries

The loader factories and constructors accept `epochize` and `limit`.
Epochization continues the worker stream between segments; a limit resets it when an
iteration completes. Standalone `EpochizeDataset` and `LimitDataset` provide the
corresponding behavior when composing processing datasets directly.

The loader constructs the boundary wrapper inside its stride wrapper, around the
TaskEncoder-built processing pipeline:

```text
StrideDataset -> EpochizeDataset / LimitDataset -> processing pipeline
```

Public lengths remain output budgets per rank after striding. For a budget divisible by
the local physical worker count, each logical stream receives the per-physical-worker
budget multiplied by fanout. For remainders, `WorkerConfig.logical_worker_length()` counts
the contributions assigned to each logical group in synchronized delivery order. Every
replica of that group receives the same logical boundary, even when a group spans ranks.
With both options enabled, the smaller length is used and completion resets the source.

All count conversion belongs to the loader integration. Before each worker iterator
starts, the loader supplies the group's endpoint as an ordinary wrapper length, multiplied
by the local physical worker count to account for the wrapper's normal division among
workers. `EpochizeDataset` and `LimitDataset` only count outputs and save their offset;
they have no logical-worker option or replay-specific state. The stride does not inspect
boundary wrappers.

Boundary wrappers retain their full offset at the final yield and reset it only when the
iterator completes. Ordinary recursive save/restore therefore preserves both partial
segments and pending boundary completion. Epochization retains its inner iterator between
segments. A limit closes that iterator and resets the inner pipeline after completion.
The loader's scheduling count restarts for each worker pass.

Configured lengths are not saved. A restored loader may use new epochize/limit lengths
and another physical fanout while retaining its logical pipeline position. Before
iteration starts, the loader sets the endpoint to the snapshot's progress plus its
logical replay distance and the newly allocated remaining budget. The boundary wrapper
then uses its ordinary length check for the whole pass. If the new budget has already
been consumed, replay reaches the saved position and the iterator completes without new
outputs. Replay stays within the checkpoint's iterator pass and never reopens an
exhausted iterator.

After a partial save, the remaining rank budget is allocated under the new worker mapping.
Each group's endpoint is its recovered logical position plus its share of those remaining
outputs. This handles groups whose contributions before the save differ from their shares
under the new fanout. The endpoint is plain length configuration supplied by the loader,
recomputed during restore and excluded from the saved dataset state.

Output logging remains outside the stride, so it observes only retained outputs.
Worker startup, queue shutdown, checkpoint selection, and group routing remain owned by
the loader.


## Checkpoint and worker responsibilities

The checkpoint schemas and routing helpers in `savable_loader.py` handle logical
snapshots, group merging, and validation independently of loader execution.

The worker wrappers in `savable_loader.py` own per-worker iteration, checkpoint history,
and snapshot installation.
The main-process and subprocess paths use the same checkpoint representation:
a recursive dataset snapshot containing its logical cursor and boundary state, RNG state, plus a replay offset
in full logical outputs. Physical output counters are only used internally to locate the
historical snapshot matching consumed outputs despite worker prefetching.

Snapshot installation resets the pipeline once and copies mutable dataset state once before
restoring it. This keeps the caller's checkpoint and retained history unchanged during replay
or later iteration. RNG restoration belongs to this same worker operation. The stride wrapper
restores its logical cursor through the ordinary dataset state and receives transient replay information directly.


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
