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
    state = torch.load(f"dataloader_rank{data_parallel_rank}.pt", weights_only=False)
    # Restore the state
    loader.restore_state_rank(state)

```


## Logical workers

Set `WorkerConfig(logical_workers=L, ...)` to keep `L` global dataset streams while using
more physical workers. The global physical count is `world_size * max(1, num_workers)`;
it must be a positive multiple of `L`. Each physical member advances its group's complete
pipeline and retains a different stride of final outputs. Sharding and worker seeds use
the logical identity. Without an explicit count, existing physical partitioning is unchanged.

For example, `world_size=2, num_workers=4, logical_workers=2` gives four physical members
per logical stream. Ranks advance synchronously. Members retain outputs in local-worker,
then rank order within each stream; different streams may interleave differently after a
topology change. Finite streams are not padded or balanced across ranks.

Use `@skip_safe` or `@stateless(skip_safe=True)` for pure transforms whose work may be
omitted for discarded outputs. An unsafe outer stage blocks skipping its inputs. Packing
selection, filtering decisions, generators, and size metrics still execute. Shuffle buffers
retain real inputs. A skip-safe callable must preserve output cardinality, have no required
side effects or validation, and use isolated randomness. Exceptions, including `SkipSample`,
from a skip-safe callable are fatal when logical workers are enabled. Cooking is skipped only when both the cooking hook
and the selected cooker are marked safe. Fanout still advances all source positions;
it does not promise reduced source I/O.

Keep the recipe, logical count, seed, and TaskEncoder fixed for exact logical replay.
Use `@stateless(restore_task_encoder_seeds=True)` with `self.rng` for reproducible transforms
in thread workers. `current_sample_index` follows logical identities. Physical worker IDs and
`current_batch_index` are execution details and
may change with topology.
