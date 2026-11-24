<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Save and Restore

Save each rank's data loader alongside its training checkpoint. When training resumes,
each rank continues with its next undelivered batch.

(checkpoint-version-compatibility)=
## Checkpoint version compatibility

Save and restore dataloader checkpoints with the same Energon version. For a source
checkout, use the same Git revision. Pin that version for training runs that may need
to resume.

Recipe migration and checkpoint redistribution also operate within the version that
wrote the checkpoint.

## Save and restore per rank

Each data-parallel rank saves its own loader state at the same training step.
Use the data-parallel rank in the filename so each rank can reload its own state.
If your training framework gathers checkpoints into a single file, store each
rank's state under its data-parallel rank and pass that entry back on restore.

```python
import torch
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

path = f'dataloader_state_rank{worker_config.rank}.pth'
with get_my_loader() as loader:
    for batch in loader:
        train_step(batch)
        if should_checkpoint():
            torch.save(loader.save_state_rank(), path)
            break

# Resume from your training checkpoint.
state = torch.load(path, weights_only=False)
with get_my_loader().with_restored_state_rank(state) as loader:
    for batch in loader:
        train_step(batch)
```

(checkpoint-recipe-migration)=
## Migrating a Checkpoint After Recipe Changes

Exact restore expects the new loader to have the same dataset topology and configuration
as the saved loader. To resume after
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

### Custom loader checkpoint timeouts and batch sizes

Asynchronous worker startup and checkpoint requests wait up to 60 seconds per request
by default. Pass `worker_request_timeout` to the loader factory to adjust that limit.
The timeout applies to asynchronous workers. Main-process loading runs synchronously.

A loader with zero workers uses the training process's random number generators.
Save and restore those generators with the training checkpoint.

Redistributing a checkpoint with the same world size preserves each rank's next worker.
To change the world size, set `prefetch_factor=0` on the source loader to avoid pending
batches, or pass `--discard-prefetched-batches` to `energon checkpoint redist` to skip
the undelivered batches and resume from each worker's saved position.
The CLI also rejects checkpoints that would resume a generator partway through an input.
When the CLI reduces the micro-batch size, the target loader applies its `drop_last`
setting to the resized prefetched batches on restore. Keep the grouped batch size and
tail policy the same when restoring grouped batches.

### Migrating a custom loader checkpoint with workers

By default, migration rejects a changed recipe with pending prefetched batches.
Set `prefetch_factor=0` on the source loader to avoid them, or pass
`discard_prefetched_batches=True` to `migrate_dataloader_state()` to skip the
undelivered batches. Both migration options report how many batches each affected
source rank discards. An unchanged recipe still supports exact replay
with prefetching. The custom loader checkpoint has a different format from the
previous PyTorch-backed loader checkpoint.
