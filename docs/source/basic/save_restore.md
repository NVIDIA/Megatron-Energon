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

## Summary

In each of these scenarios, ensure that the logic for saving and restoring the state is appropriately synchronized across ranks to maintain consistency.
If you encounter torch distributed errors, likely torch distributed calls are out of sync, or not all ranks are called correctly. If unsure, debug using the first scenario, saving each rank separately.
