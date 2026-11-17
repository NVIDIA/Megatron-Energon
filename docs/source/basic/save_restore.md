<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Save and Restore

For long-running training jobs, you will usually need to stop and resume the training including the data loader.
One of energon's unqiue features is the deterministic save and restore capability.

At any iteration, you'll be able to store the overall state of the data loader across all ranks and accurately resume it later on, to continue where it left off.
Below, we list a few different ways to achieve that.

## Scenarios

All these scenarios work in distributed and non-distributed settings.

For simplicity, we are assuming the same loader and data for all scenarios from the [](../basic/quickstart):
```python
from megatron.energon import get_train_dataset, get_savable_loader, WorkerConfig

worker_config = WorkerConfig.default_worker_config()

def get_my_loader():
    return get_savable_loader(get_train_dataset(
        'coyo-coco-dataset.yaml',
        batch_size=4,
        shuffle_buffer_size=100,
        max_samples_per_sequence=100,
        worker_config=worker_config,
    ))

```

### 1. Save/Restore the State per Rank Separately

In this scenario, each rank saves and restores its own state in an independent file.
`save_state_rank()` saves every local physical worker, including when a logical group
spans several ranks. Each rank reloads its own file without transferring state across
ranks. Keep the rank assignment, world size, workers per rank, and logical worker
count unchanged when restoring.

To change the physical topology on restart, use
{ref}`save-restore-per-logical-rank` instead. The logical APIs select one checkpoint
per logical worker and route it to the ranks that need it.

```python
# Saving the state
loader = get_my_loader()

# Iterate for some steps
for i, batch in zip(range(10), loader):
    print(batch)
    break

# Save the state
state = loader.save_state_rank()
# Save the state on each rank
# In this example, save the state using `torch.save`, this can of course be custom
torch.save(state, f'dataloader_state_rank{worker_config.rank}.pth')
```

```python
# Restoring the state
loader = get_my_loader()

# Now, when restoring the state:
state = torch.load(f'dataloader_state_rank{worker_config.rank}.pth', weights_only=False)

# Restore the state for the loader on each rank separately
loader.restore_state_rank(state)
```

(save-restore-per-logical-rank)=

### 2. Save/Restore the State per Logical Rank Separately

In this scenario, save one file per global logical worker id, independently of the
physical rank that writes it. On restart, each rank loads the files for its assigned
logical workers. This allows the number of ranks or loader workers to change without
requiring a manifest mapping logical workers to rank files.

Set `worker_config.logical_workers` explicitly before building the dataset, and keep
that count fixed when restoring. The global physical worker count must be a positive
multiple of the logical worker count. See {ref}`logical-workers` for the mapping and
restore guarantees.

All ranks must save after the same number of delivered batches, including zero.
`save_state_logical_rank()` selects each logical group's last contributing physical worker.
A rank may contribute several groups or return an empty tuple; the rank responsible for a
group can change at each save. The save and restore methods below use no distributed
collectives.

```python
from pathlib import Path

import torch

# Use a separate shared directory for each training checkpoint.
checkpoint_dir = Path('checkpoint_step_1000/dataloader')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# On each rank, after synchronized training iterations:
for logical_id, state in loader.save_state_logical_rank():
    torch.save(state, checkpoint_dir / f'logical_{logical_id}.pth')
```

Wait until all ranks have finished writing before treating the training checkpoint
as complete. Every returned state preserves the logical worker count,
micro batch size, and iteration index. File storage remains
under your control.

On restart, construct a fresh loader with the saved logical worker count and the
desired physical topology. Before iterating, load the files requested by that rank:

```python
# worker_config must already describe the new topology and fixed logical count.
loader = get_my_loader()

logical_ids = loader.get_logical_worker_group_ids()
states = {
    logical_id: torch.load(checkpoint_dir / f'logical_{logical_id}.pth', weights_only=False)
    for logical_id in logical_ids
}
loader.restore_state_logical_rank(states)
```

Use files from the same training checkpoint for every rank. Multiple ranks may load
the same logical worker file when they share a logical group. The loader assigns the
restored state to their physical workers and resumes the logical stream.

To gather the logical states onto one rank, all ranks call
`loader.save_state_logical_global(global_dst_rank=0)`. The destination receives the
same `(logical_id, state)` pairs; other ranks receive `None`. Restore the collection
with `loader.restore_state_logical_global(states, src_rank=0)`, where `src_rank` is
within the data parallel group. All ranks participate in the broadcast. See
{ref}`logical-workers` for details.



### 3. Save/Restore the State on the Primary Rank Only

In this scenario, the primary rank (usually rank 0) is responsible for saving the state.
All ranks' states are collected (gathered) by one rank and can be stored in one file.
When restoring, the primary rank scatters the entries of the rank-ordered state list to their ranks. The physical topology and logical worker count must remain unchanged.
This approach centralizes the state management, which can simplify the process and reduces the number of files stored.

```python
# Saving the state
loader = get_my_loader()

# Iterate for some steps
for i, batch in zip(range(10), loader):
    print(batch)
    break

# Save the state to primary rank 0
state = loader.save_state_global(global_dst_rank=0)
if worker_config.rank == 0:
    # Only rank 0 has the state now, for the others, the state is None
    # In this example, save the state using `torch.save`, this can of course be custom
    torch.save(state, 'dataloader_state.pth')
```

```python
# Restoring the state
loader = get_my_loader()

# Load the state only on the primary rank
if worker_config.rank == 0:
    state = torch.load('dataloader_state.pth', weights_only=False)
else:
    state = None

# Restore the state for the loader, broadcasting from rank 0
loader.restore_state_global(state, src_rank=0)
```


```{admonition} Note
:class: important
Even though only one rank collects the states, all ranks need to execute the `loader.save_state_global()` and `loader.restore_state_global()` lines of code
```

### 4. Save the State on the Primary Rank, Restore on Ranks Separately

In this scenario, the primary rank saves the state, but each rank restores the state separately. Each rank loads the global list and selects its own rank's state. The physical topology and logical worker count must remain unchanged. This approach combines centralized saving with distributed restoring and is rather uncommon.

Depending on the framework used for training, that framework may already handle the scattering/gathering of the states. In that case, refer to the first scenario using `save_state_rank`/`restore_state_rank`.

```python
# Saving the state
loader = get_my_loader()

# Iterate for some steps
for i, batch in zip(range(10), loader):
    print(batch)
    break

# Save the state
state = loader.save_state_global(global_dst_rank=0)
if worker_config.rank == 0:
    # In this example, save the state using `torch.save`, this can of course be custom
    torch.save(state, 'dataloader_state.pth')
```

```python
# Restoring the state
loader = get_my_loader()

# Load on all ranks
state = torch.load('dataloader_state.pth', weights_only=False)

# Restore the state for the loader on current rank, using all ranks checkpoint
loader.restore_state_global(state, src_rank=None)
```

## Inspecting and Redistributing Checkpoint Files

Use `energon checkpoint info FILE...` to inspect physical rank states, global physical
state lists, individual logical states, or logical collections saved as mappings or
`(logical_id, state)` pairs. Files wrapped in a `dataloader_state_dict` entry are also
supported. Inspecting one logical contribution does not require every group.

To redistribute a complete checkpoint into two ranks:

```bash
energon checkpoint redist --new-world-size 2 checkpoint_rank*.pt redistributed/
```

Supply every physical rank file in natural filename order, or one global file. For
logical checkpoints, supply files covering every logical group exactly once; filenames
may have gaps or arbitrary names. All contributions must come from the same synchronized
save. Redistribution selects each logical group's last contributor from physical saves.

The destination uses one physical worker per logical worker. The new rank count must
divide the logical worker count; configure workers per rank to their quotient (zero
subprocess workers is also valid when the quotient is one), and retain the saved logical
worker count. Rank-file inputs produce `state_rank0.pt`, `state_rank1.pt`, and so on.
Use `--output-filename 'loader_{rank}.pth'` to choose another filename template. A global
collection stays in one file with its original filename and checkpoint envelope.
Physical inputs produce physical states for `restore_state_rank` or `restore_state_global`.
Logical rank outputs are mappings of logical ids to independently savable states; global
logical outputs are `(logical_id, state)` pairs. Pass either form to the corresponding
logical restore method.

## Error Handling During Restore

By default, energon uses a strict error handler during checkpoint restoration to ensure data integrity.
If sample restoration fails, the exception is immediately raised.

You can customize this behavior via the `restore_error_handler` parameter in {py:class}`WorkerConfig <megatron.energon.WorkerConfig>`.
See [](../advanced/error_handling) for details.

## Summary

In each of these scenarios, ensure that the logic for saving and restoring the state is appropriately synchronized across ranks to maintain consistency.
If you encounter torch distributed errors, likely torch distributed calls are out of sync, or not all ranks are called correctly. If unsure, debug using the first scenario, saving each rank separately.
