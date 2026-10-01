<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Parallelism

Neural network parallelism can be categorized into several types:

1. **Data Parallelism** (DP): This involves splitting the data across multiple processors and performing the same operation on each subset of the data. It is commonly used to increase the global batch size.

2. **Model Parallelism**: In this approach, different parts of the model are distributed across multiple processors. This is useful when the model itself is too large to fit into the memory of a single processor.

3. **Pipeline Parallelism** (PP): This technique involves breaking down the model into different stages and processing different mini-batches of data through these stages in a pipeline fashion. It helps in improving the utilization of resources and reducing idle time.

4. **Tensor Parallelism** (TP): This method splits individual tensors (weights and activations) across multiple devices. It is particularly effective for very large models where even a single layer cannot fit into the memory of one device.

These parallelisms have different consequences for the dataloader:

- **Data Parallelism** (DP): The dataloader needs to ensure that each processor gets a different subset of the data. This is supported by Energon. The data parallel groups should be specified in the worker config.

- **Pipeline Parallelism** (PP): Data is typically only loaded on the first Pipeline Parallel rank, and propagates through the other ranks within the pipeline parallel group. This means, you only instantiate an Energon dataset and loader on the first ranks of those groups.

- **Tensor Parallelism** (TP): The dataloader will load the same input data on multiple devices. Typically, this can be ensured by either instantiating the dataloader exactly the same on the same data parallel ranks in different data parallel groups, or e.g. by loading the data only once and distributing it using torch distributed.


(logical-workers-fanout)=

## Logical Workers and Fanout

Energon normally creates one logical dataset partition for every physical worker.
A physical worker is a PyTorch DataLoader worker process; when `num_workers=0`,
the rank's main process counts as one physical worker. The total physical worker
count is therefore:

```text
world_size * num_workers    if num_workers > 0
world_size                  if num_workers == 0
```

For example, the following configuration is valid:

```python
from megatron.energon import WorkerConfig

worker_config = WorkerConfig(
    rank=data_parallel_rank,
    world_size=4,
    num_workers=3,
    logical_workers=2,
)
```

It creates `4 * 3 = 12` physical workers and two logical streams. Twelve is
divisible by two, so the fanout is six:

| Physical global worker IDs | Logical global worker ID | Stride offsets | Stride |
| --- | --- | --- | --- |
| 0, 1, 2, 3, 4, 5 | 0 | 0, 1, 2, 3, 4, 5 | 6 |
| 6, 7, 8, 9, 10, 11 | 1 | 0, 1, 2, 3, 4, 5 | 6 |

Set {py:attr}`WorkerConfig.logical_workers
<megatron.energon.WorkerConfig.logical_workers>` to use fewer logical dataset
partitions than physical execution slots. This separates dataset stream identity
from the number of processes executing those streams. `logical_workers` must:

- be greater than zero;
- not exceed the global physical worker count; and
- divide the global physical worker count exactly.

When it is omitted, the logical and physical worker counts are equal and there
is no fanout.

All six physical workers assigned to one logical stream advance that same
deterministic stream. Each one yields only the outputs at its stride offset.
Interleaving offsets `0` through `5` reconstructs the original logical stream
in order; the physical workers do not receive independently shuffled streams.

### What Fanout Skips

The stride is applied after packing and batching. The unit retained by a
physical worker is therefore a final pipeline output--normally a batch, or a
packed/encoded sample when batching is disabled.

Each physical worker must still advance all outputs of its logical stream to
preserve ordering and checkpoint state. For five of every six outputs in this
example, the worker enters **skip mode**. Skip mode can omit pure computation,
but it cannot omit the decisions that define the stream.

Consequently, fanout always still performs:

- primary dataset iteration and worker-local ordinal advancement;
- blend and shuffle decisions;
- batch and pack boundary selection;
- restore-key and sample-index advancement; and
- any function that is not explicitly skip-safe.

It may avoid work such as decoding, augmentation, tokenization, collation, or
tensor construction when that work is pure, its result is needed only for a
discarded output, and every outer stage allows skip mode to reach it.

```{admonition} Skip safety is a chain
:class: important

Skip mode propagates inward only through wrappers whose own callable is
skip-safe. An unsafe outer `encode_batch`, `batch`, or final packing function
blocks skip mode from reaching otherwise safe sample transforms below it.
Marking one inner function is therefore not enough when a later stage still
needs its real output.
```

### Example: Streaming Packing

Packing selection must always run because it determines which samples form the
next output. Keep the information required for that decision cheap--for example,
store token counts in primary metadata--and defer expensive materialization
until after selection:

```python
from megatron.energon import (
    DefaultTaskEncoder,
    PackedSamplesOutput,
    skip_safe,
    stateless,
)


class PackedTaskEncoder(DefaultTaskEncoder):
    max_pack_tokens = 8192

    @stateless
    def preencode_sample(self, sample):
        # This runs for retained and discarded packs: the selector needs it.
        sample.num_tokens = sample.metadata_token_count
        return sample

    @stateless
    def select_next_pack(self, samples):
        # Never skip-safe: these pulls and boundaries define the stream.
        pack = []
        token_count = 0
        for sample in samples:
            if pack and token_count + sample.num_tokens > self.max_pack_tokens:
                return PackedSamplesOutput(
                    packs=[pack],
                    pushback=(sample,),
                )
            pack.append(sample)
            token_count += sample.num_tokens
        return [pack] if pack else []

    @skip_safe
    @stateless
    def postencode_sample(self, sample):
        # Expensive work deferred until the selected pack is known.
        sample.image = decode_and_augment(sample.lazy_image)
        return sample

    @skip_safe
    @stateless
    def pack_selected_samples(self, samples):
        # Pure final construction; omitted for a discarded pack.
        return collate_pack(samples)
```

Use this with `packing_buffer_size="stream"`. For a discarded packed output:

1. `select_next_pack` still pulls samples and chooses the boundary;
2. the carryover/pushback and all sample indexes still advance;
3. `postencode_sample` can be omitted only because both it and
   `pack_selected_samples` are skip-safe; and
4. `pack_selected_samples` is omitted while its output index still advances.

If final packing is not skip-safe, it needs real inputs, so Energon must also
run `postencode_sample`. If tokenization is required to decide pack boundaries,
it belongs before or inside selection and fanout cannot avoid it. The
optimization is most useful when selection can use cheap lengths while media
decode or tensor construction happens afterward.

Packing selectors--`select_samples_to_pack` and `select_next_pack`--must never
be marked skip-safe.

### Example: Batching Without Packing

Without packing, a fully skip-safe outer chain can avoid per-sample processing,
collation, and final batch encoding for discarded batches:

```python
from megatron.energon import DefaultTaskEncoder, skip_safe, stateless


class BatchedTaskEncoder(DefaultTaskEncoder):
    @skip_safe
    @stateless
    def encode_sample(self, sample):
        return decode_and_tokenize(sample)

    @skip_safe
    @stateless
    def batch(self, samples):
        return collate(samples)

    @skip_safe
    @stateless
    def encode_batch(self, batch):
        return normalize_batch(batch)
```

The dataset still consumes enough input samples to establish each batch
boundary. For a discarded batch, however, the skip-safe chain can replace the
sample transforms and batch construction with state/index advancement.

If `encode_batch` is overridden but not skip-safe, it runs and prevents skip
mode from reaching `batch` or `encode_sample`. If only `encode_batch` is
skip-safe, Energon can omit that function but an unsafe `batch` still runs and
blocks skipping the per-sample encoder. Generator functions cannot be elided
because their output cardinality is part of the stream.

With `batch_size=None` and no packing, each encoded sample is a final output;
then a skip-safe `encode_sample` is sufficient because no batching wrapper sits
outside it.

### Where `skip_safe` Helps and Where It Does Not

`skip_safe` is useful for expensive, deterministic, side-effect-free work whose
result is consumed only by the current output. Typical candidates are media
decode, pure augmentation with sample-isolated RNG, token/tensor construction,
collation, and final normalization.

It does not reduce source reads, blend/shuffle work, packing selection, or the
number of logical outputs advanced. It must not be used for:

- state updates needed by later retained outputs;
- required validation, logging, counters, writes, or cache population;
- random calls whose state is shared with later outputs;
- filtering or generators that change output cardinality; or
- any computation needed by a non-skip-safe outer stage.

At fanout six, reachable skip-safe work can be avoided for roughly five of six
outputs processed by each physical worker. This is a compute optimization, not
a promise of a sixfold reduction in data I/O or total pipeline work.

See {ref}`skip-safe-functions` for the TaskEncoder annotation contract.

```{admonition} Stream and checkpoint semantics
:class: important

`logical_workers`, the dataset or recipe, blend weights, split, subset, filters,
shuffle settings, and seed all contribute to the data stream definition. Keep
them stable for an exact checkpoint restore.

A fixed logical worker count keeps dataset partition and RNG identities stable
when physical fanout changes, provided the new physical worker count remains a
valid multiple. It does not make arbitrary recipe or task-encoder changes
equivalent to the old stream. If the recipe changes, use
{ref}`checkpoint-recipe-migration`
and account for its weaker ordering guarantees.
```

The assignment can be inspected for diagnostics:

```python
assignment = worker_config.logical_assignment_for_physical(
    physical_global_worker_id=0
)
print(
    assignment.logical_global_worker_id,
    assignment.stride_offset,
    assignment.stride,
)
```


## Example

Example with the following ranks and worker configuration (Data Parallel = 2, Pipeline Parallel = 2, Tensor Parallel = 2):
* `Global Rank 0`: `DP Rank = 0` (DP group A), `PP Rank = 0`, `TP Rank = 0`
* `Global Rank 1`: `DP Rank = 0` (DP group B), `PP Rank = 0`, `TP Rank = 1`
* `Global Rank 2`: `DP Rank = X` (No DP group), `PP Rank = 1`, `TP Rank = 0`
* `Global Rank 3`: `DP Rank = X` (No DP group), `PP Rank = 1`, `TP Rank = 1`
* `Global Rank 4`: `DP Rank = 1` (DP group A), `PP Rank = 0`, `TP Rank = 0`
* `Global Rank 5`: `DP Rank = 1` (DP group B), `PP Rank = 0`, `TP Rank = 1`
* `Global Rank 6`: `DP Rank = X` (No DP group), `PP Rank = 1`, `TP Rank = 0`
* `Global Rank 7`: `DP Rank = X` (No DP group), `PP Rank = 1`, `TP Rank = 1`

When saving the state of the data loader, we only need to store the states
of global ranks 0 and 4, i.e. the fist DP group "A".
Ranks 1 and 5 will have the same state as they are duplicates.

When restoring the state, global ranks 0, 1, 4, 5 need to receive a state.

There are different ways to achieve this. The following example illustrates how the state
can be saved and restored in a distributed setting.


```py
import torch
from megatron.energon import get_train_dataset, get_savable_loader, WorkerConfig

# Initialize the process group
torch.distributed.init_process_group(backend='nccl')

# Get the DP, PP, TP ranks
global_rank = torch.distributed.get_rank()
data_parallel_rank = [0, 0, None, None, 1, 1, None, None][global_rank]
pipeline_parallel_rank = [0, 0, 1, 1, 0, 0, 1, 1][global_rank]
tensor_parallel_rank = [0, 1, 0, 1, 0, 1, 0, 1][global_rank]

if global_rank in (0, 4):
    # DP Group A
    # If on rank 0 or 4, the DP group consists of those ranks (each representing DP ranks 0 and 1).
    data_parallel_group = torch.distributed.new_group(ranks=[0, 4])
elif global_rank in (1, 5):
    # DP Group B
    # If on rank 1 or 5, the DP group consists of those ranks (each representing DP ranks 0 and 1).
    data_parallel_group = torch.distributed.new_group(ranks=[1, 5])
else:
    data_parallel_group = None

if data_parallel_rank is not None:
    assert pipeline_parallel_rank == 0, "Only Pipeline Parallel ranks 0 load data"
    
    # Set the worker config correspondingly
    worker_config = WorkerConfig(
        rank=data_parallel_rank,
        world_size=torch.distributed.get_world_size(data_parallel_group),
        num_workers=3,
        data_parallel_group=data_parallel_group,
    )

    # Create the loader with that config
    loader = get_savable_loader(get_train_dataset(
        'coyo-coco-dataset.yaml',
        batch_size=4,
        shuffle_buffer_size=100,
        max_samples_per_sequence=100,
        worker_config=worker_config,
    ))

    # Iterate the data
    for i, batch in zip(range(10), loader):
        # Do forward-backward pass
        print(batch)
        break

    if tensor_parallel_rank == 0:
        # Save the state only for the first TP rank (the other TP ranks have a copy of that state)
        # Save the state
        state = loader.save_state_rank()
        # E.g. save to disk with torch
        torch.save(state, f"dataloader_rank{data_parallel_rank}.pt")

        # Alternatively, save once for the whole dp group:
        # state = loader.save_state_global(global_dst_rank=0)
        # if state is not None:
        #     torch.save(state, "dataloader.pt")


# ... when loading:
if data_parallel_rank is not None:
    assert pipeline_parallel_rank == 0, "Only Pipeline Parallel ranks 0 load data"

    # Restore the state for a new loader
    loader = get_savable_loader(get_train_dataset(
        'coyo-coco-dataset.yaml',
        batch_size=4,
        shuffle_buffer_size=100,
        max_samples_per_sequence=100,
        worker_config=worker_config,
    ))

    # E.g. load from disk as saved above
    state = torch.load(f"dataloader_rank{data_parallel_rank}.pt")
    # Restore the state
    loader.restore_state_rank(state)

    # Alternatively, when using a global checkpoint,
    # load the checkpoint from disk on every dp rank:
    # state = torch.load("dataloader.pt")
    # loader.restore_state_global(state)

    # Or load only once from disk for each dp group:
    # if data_parallel_rank == 0:
    #     state = torch.load("dataloader.pt")
    # else:
    #     state = None
    # loader.restore_state_global(state, src_rank=0)

```
