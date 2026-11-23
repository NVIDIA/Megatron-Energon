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
This is our recommended way, since it avoids transferring the data across ranks.

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
torch.save(dataloader_state, f'dataloader_state_rank{worker_config.rank}.pth')
```

```python
# Restoring the state
loader = get_my_loader()

# Now, when restoring the state:
state = torch.load(f'dataloader_state_rank{worker_config.rank}.pth')

# Restore the state for the loader on each rank separately
loader.restore_state_rank(state)
```


### 2. Save/Restore the State on the Primary Rank Only

In this scenario, the primary rank (usually rank 0) is responsible for saving the state.
All ranks' states are collected (gathered) by one rank and can be stored in one file.
When restoring, the state is scatterd from the primary rank to all other ranks.
This approach centralizes the state management, which can simplify the process and reduces the number of files stored.

```python
# Saving the state
loader = get_my_loader()

# Iterate for some steps
for i, batch in zip(range(10), loader):
    print(batch)
    break

# Save the state to primary rank 0
state = loader.save_state_global(dst_rank=0)
if worker_config.rank == 0:
    # Only rank 0 has the state now, for the others, the state is None
    # In this example, save the state using `torch.save`, this can of course be custom
    torch.save(dataloader_state, 'dataloader_state.pth')
```

```python
# Restoring the state
loader = get_my_loader()

# Load the state only on the primary rank
if worker_config.rank == 0:
    state = torch.load('dataloader_state.pth')
else:
    state = None

# Restore the state for the loader, broadcasting from rank 0
loader.restore_state_global(state, src_rank=0)
```


```{admonition} Note
:class: important
Even though only one rank collects the states, all ranks need to execute the `loader.save_state_global()` and `loader.restore_state_global()` lines of code
```

### 3. Save the State on the Primary Rank, Restore on Ranks Separately

In this scenario, the primary rank saves the state, but each rank restores the state separately. Each rank loads all saved states and selects the correct one. This approach combines centralized saving with distributed restoring and is rather uncommon.

Depending on the framework used for training, that framework may already handle the scattering/gathering of the states. In that case, refer to the first scenario using `save_state_rank`/`restore_state_rank`.

```python
# Saving the state
loader = get_my_loader()

# Iterate for some steps
for i, batch in zip(range(10), loader):
    print(batch)
    break

# Save the state
state = loader.save_state_global(dst_rank=0)
if worker_config.rank == 0:
    # In this example, save the state using `torch.save`, this can of course be custom
    torch.save(dataloader_state, 'dataloader_state.pth')
```

```python
# Restoring the state
loader = get_my_loader()

# Load on all ranks
state = torch.load('dataloader_state.pth')

# Restore the state for the loader on current rank, using all ranks checkpoint
loader.restore_state_global(state, src_rank=None)
```

(checkpoint-recipe-migration)=
## Migrating a Checkpoint After Recipe Changes

The restore methods above are exact restores: they expect the new loader to have
the same dataset topology and configuration as the saved loader. To resume after
editing a recipe, save resume metadata alongside the normal loader state. Two
kinds of edits are supported:

* adding datasets,
* setting the `weight` of datasets in a `blend` to `0`.

To stop using a dataset, set its weight to `0` instead of deleting its entry. The
dataset is not read anymore, but its entry keeps the samples of all other
datasets assigned to the same workers. Its progress is dropped: if the weight is
raised again later, the dataset starts from a fresh state.

The following per-rank example saves both objects:

```python
import torch

from megatron.energon.checkpoint import collect_resume_metadata

loader = get_my_loader()

# Train for some number of steps.
for batch in loader:
    train_step(batch)
    if should_checkpoint():
        break

state = loader.save_state_rank()
resume_metadata = collect_resume_metadata(loader)

torch.save(
    {
        "dataloader_state": state,
        # Store the serializable form rather than depending on pickling the class.
        "dataloader_resume_metadata": resume_metadata.to_dict(),
    },
    f"dataloader_state_rank{worker_config.rank}.pth",
)
```

After editing the recipe, construct the new loader first, migrate the saved state
to its topology, and then restore:

```python
import torch

from megatron.energon.checkpoint import migrate_dataloader_state

checkpoint = torch.load(
    f"dataloader_state_rank{worker_config.rank}.pth",
    weights_only=False,
)

# get_my_loader() now reads the edited recipe.
loader = get_my_loader()
migrated_state = migrate_dataloader_state(
    loader,
    checkpoint["dataloader_state"],
    checkpoint["dataloader_resume_metadata"],
)
loader.restore_state_rank(migrated_state)
```

Migration reuses the saved progress of every dataset whose configuration is
unchanged. The configuration of a dataset includes:

* dataset path,
* `split_part`,
* relative or absolute subset,
* filter name,
* the assignment of its samples to workers.

Added datasets and datasets with a changed configuration start from a fresh
state. The state of everything that combines the datasets, such as blending,
shuffling, packing, and batching, is rebuilt for the new recipe. This also covers
a recipe that changes between a single used dataset and a blend of several
datasets.

If a dataset is still part of the recipe but its saved progress cannot be
reused, `migrate_dataloader_state` emits a `UserWarning` that names the dataset.
Such a dataset starts from a fresh state.

```{admonition} Migration is not an exact continuation
:class: important

Adding or reweighting datasets changes the global blended stream. Migration
preserves the progress of the datasets, but it cannot preserve the exact next
global sample sequence of the old recipe.

A path move or a change to `split_part`, subset, or filter starts that dataset
fresh. The same applies to a dataset that is listed more than once with the same
configuration, because its entries cannot be told apart. Changes to auxiliary
references, tokenizer, or task-encoder behavior are not validated
automatically; only migrate when the retained dataset progress is semantically
compatible with the new training configuration.

With more than one worker (workers of all ranks combined), the samples of a
dataset are assigned to the workers depending on the entries listed before it.
Add new datasets at the end of the recipe, and do not delete or reorder entries:
this can change the assignment of the datasets behind them. Their progress then
cannot be reused, and they are named in the warning described above. Changing
weights, including to `0`, keeps the assignment of all datasets.

Samples that were read but not yet returned when the state was saved are
dropped if they are buffered above the changed part of the recipe, for example
in the shuffle buffer, in a packing buffer, or in an incomplete batch. With
worker processes, a few samples around the checkpoint may additionally be
repeated or skipped.
```

The example uses `save_state_rank()` and must run independently on every data
parallel rank. If a training framework gathers loader states centrally, migrate
each rank's state against the corresponding newly constructed loader before
restoring it.

## Error Handling During Restore

By default, energon uses a strict error handler during checkpoint restoration to ensure data integrity.
If sample restoration fails, the exception is immediately raised.

You can customize this behavior via the `restore_error_handler` parameter in {py:class}`WorkerConfig <megatron.energon.WorkerConfig>`.
See [](../advanced/error_handling) for details.

## Summary

In each of these scenarios, ensure that the logic for saving and restoring the state is appropriately synchronized across ranks to maintain consistency.
If you encounter torch distributed errors, likely torch distributed calls are out of sync, or not all ranks are called correctly. If unsure, debug using the first scenario, saving each rank separately.
