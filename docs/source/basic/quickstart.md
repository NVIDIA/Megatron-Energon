<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Quickstart

You can use Megatron Energon to load datasets in the energon dataset format. This format is essentially [WebDataset](https://github.com/webdataset/webdataset) with some meta data added.

For a moment let's assume you already have prepared a dataset in the needed format, and it's stored on
disk at `/my/dataset/path`. If you want to create a dataset now, check out [](data_prep).

If you simply want to use some dummy data for trying this out, checkout the unit test method `create_captioning_test_dataset` inside `tests/test_dataset.py`.

```{admonition} Good to know
:class: tip
You can also store your dataset inside an S3-compatible object store and load it from there! See [](../advanced/remote_dataset)
```

You can then load the dataset like this:
```python
from megatron.energon import get_train_dataset, get_loader, WorkerConfig

ds = get_train_dataset(
    '/my/dataset/path',
    batch_size=1,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
    worker_config=WorkerConfig.default_worker_config(),
)

loader = get_loader(ds)

for batch in loader:
    # Do something with batch
    # Infer, gradient step, ...
    pass
```

At first, we call {py:meth}`get_train_dataset <megatron.energon.get_train_dataset>` (click to see signature).
The method will check what kind of dataset is on disk and instantiate the correct class for it.

A worker configuration is always needed to specify how the work is distributed across multiple ranks and workers.
In this simple example, we use a helper method {py:meth}`default_worker_config <megatron.energon.WorkerConfig.default_worker_config>` to get reasonable default values.

The dataset should not be iterated directly, but used with a loader which handles the worker processes.
The batches will contain samples of the sample type specified in the [task encoder](task_encoder).

```{admonition} Good to know
:class: tip
Since we did not specify a task encoder above, the {py:class}`DefaultTaskEncoder <megatron.energon.DefaultTaskEncoder>` will be used.
It will not transform the data. For batching it will use common sense magic to pad and stack tensors or build lists if the type is unknown.
```

_Wait. Why does the dataset create batches? Shouldn't the dataloader do that?_

Energon will create batches at dataset level.
Internally, most of the cool things that energon can do (such as blending datasets together, [sequence packing](../advanced/packing), etc.)
are dataset wrappers. Even the process of batching is such a wrapper and the default {py:meth}`get_train_dataset <megatron.energon.get_train_dataset>`
function will construct a suitable combination of all these based on the arguments you pass to that function.
Check out the [](basics_flow) section to see the steps in which the data is processed.

_Why must `shuffle_buffer_size` and `max_samples_per_sequence` be set explicitly?_

As the library is designed to work on (sequential) webdatasets but still wants to provide proper shuffling, these parameters are required. To make sure, the user does not forget to set these, we enforce them to be set explicitly.
A value of 100 for both settings for image datasets seems to work well (i.e. balanced shuffling randomness vs seeking performance impact), but datasets where the samples are lots larger or smaller might require different settings.
Setting the sequence length to a very small size compared to the number of samples in the dataset will result in more random access, thus slowing down dataloading, so the recommendation is to set it to a high enough value.
At the same time, a high value reduces the shuffling randomness, which requires a larger shuffle buffer size to compensate for that (i.e. higher memory footprint and longer state restore times).

## Tutorial 1: Preparing the Dataset

For running any tutorials, you'll need your data structured as webdataset. For more details and more options check out [](data_prep).

For example, in your folder containing the tar files, run
```sh
$ energon prepare ./

# Example answers to interactive questions:
Ratio: 8,1,1
Dataset class: CaptioningWebdataset
Field map: Yes
  image: jpg
  caption: txt  # if txt contains the caption
# or
  caption: json[caption]  # if .json contains {"caption": "My nice image"}
```


## Tutorial 2: Loading a Dataset

Let's be a bit more concrete and try out the above data loading code with a real dataset.
We are going to print the first batch and stop.

```python
from megatron.energon import get_train_dataset, get_loader, WorkerConfig

ds = get_train_dataset(
    '/path/to/your/dataset',
    batch_size=None,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
    worker_config=WorkerConfig.default_worker_config(),
)

loader = get_loader(ds)

for batch in loader:
    print(batch)
    break
```

This prints

```text
CaptioningSample(__key__='part_00123/00403.tar/004030195', image=tensor([[[0.4549, 0.4784, 0.5059,  ..., 0.8392, 0.8275, 0.7961],
          [0.4549, 0.4784, 0.5020,  ..., 0.6431, 0.6275, 0.5882],
          [0.4510, 0.4706, 0.4941,  ..., 0.6235, 0.6353, 0.6078],
          ...,
          [0.4471, 0.4196, 0.4510,  ..., 0.8471, 0.8039, 0.8275],
          [0.4667, 0.4353, 0.4667,  ..., 0.8196, 0.7804, 0.8078],
          [0.4824, 0.4549, 0.4824,  ..., 0.8196, 0.7843, 0.8118]],

         [[0.3608, 0.3843, 0.4118,  ..., 0.7373, 0.7255, 0.6941],
          [0.3608, 0.3843, 0.4078,  ..., 0.5412, 0.5255, 0.4863],
          [0.3569, 0.3765, 0.4000,  ..., 0.5098, 0.5216, 0.4941],
          ...,
          [0.3608, 0.3333, 0.3647,  ..., 0.7529, 0.7098, 0.7333],
          [0.3804, 0.3490, 0.3804,  ..., 0.7255, 0.6863, 0.7137],
          [0.3961, 0.3686, 0.3961,  ..., 0.7255, 0.6902, 0.7176]],

         [[0.2510, 0.2745, 0.3020,  ..., 0.6000, 0.5882, 0.5569],
          [0.2510, 0.2745, 0.2980,  ..., 0.4039, 0.3882, 0.3490],
          [0.2471, 0.2667, 0.2902,  ..., 0.3765, 0.3882, 0.3608],
          ...,
          [0.2667, 0.2392, 0.2706,  ..., 0.6510, 0.6000, 0.6235],
          [0.2863, 0.2549, 0.2863,  ..., 0.6235, 0.5765, 0.6039],
          [0.3020, 0.2745, 0.3020,  ..., 0.6235, 0.5882, 0.6157]]]), caption='Cello Renting vs. Buying: Which is Right for You?')
```

Awesome, it returns a {py:class}`CaptioningSample <megatron.energon.CaptioningSample>` with the attributes

- `__key__`: `part_00123/00403.tar/004030195`, the identifier of the sample like `TAR_FILE/INDEX`
  - All sample types will have a key. It's in the base class {py:class}`Sample <megatron.energon.Sample>` 
- `image`: The image as a tensor of shape `(1, 3, 267, 400)` (RGB image in a batch of size 1)
- `caption`: A list of strings (here just one since batch size is one) 

Let's also talk about the {py:class}`WorkerConfig <megatron.energon.WorkerConfig>`. As energon is made for distributed training,
you always need to provide a worker config to the dataset so specify how many ranks and workers there are and which rank you're currently on.
For this simple tutorial, we don't really distribute the work, so we use only a single rank with 4 workers. Check out the helper method {py:meth}`default_worker_config <megatron.energon.WorkerConfig.default_worker_config>` to see how the worker config is constructed. Also don't be afraid to click the *`[source]`* link and look at the very short source code of it.

## Tutorial 3: Batch Size

Actually, we would like to use a `batch_size` of more than one, let's go with 2 for now.

```python
from megatron.energon import get_train_dataset, get_loader, WorkerConfig

loader = get_loader(get_train_dataset(
    '/path/to/your/dataset',
    batch_size=2,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
    worker_config=WorkerConfig.default_worker_config(),
))

for batch in loader:
    print(batch)
    break
```

The output will be similar to above but with different shapes and lengths:

- `batch.__key__`: A list of two keys
- `batch.image`: Tensor of shape `(2, 3, 267, 400)`
- `batch.caption`: A list of two caption strings

The default [task encoder](task_encoder) automagically padded and stacked the items to a batch.
This may be ok for some cases, but usually you will want to process and batch your data differently.

Hence, we can

- either use an existing task encoder
- or define a custom one (see [](task_encoder))

## Tutorial 3: Blending using Metadataset

A typical use case is to blend multiple datasets of the same (or similar type) together.
For example, you may want to blend the COCO dataset with the COYO dataset.
The easiest way to do this, is to use the metadataset pattern. 
For this you need to create a new `yaml` file that defines the meta dataset:

`coyo-coco-dataset.yaml`:
```yaml
__module__: megatron.energon
__class__: MetadatasetV2
splits:
  # Train dataset, the datasets will be blended according to their weights 
  train:
    blend:
      - weight: 5
        path: ./coco
      - weight: 2
        path: ./coyo
  # For val and test, datasets will be concatenated
  val:
    path: ./coco
  test:
    path: ./coyo
```

This assumes, that the datasets `coyo` and `coco` exist in subfolders next to the `coyo-coco-dataset.yaml` file. You could also use absolute paths, but that will not work well when using object storage such as S3.

Usage in your loader, simply use {py:func}`get_train_dataset <megatron.energon.get_train_dataset>`:
```python
from megatron.energon import get_train_dataset, get_loader, WorkerConfig

loader = get_loader(get_train_dataset(
    'coyo-coco-dataset.yaml',
    batch_size=4,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
    worker_config=WorkerConfig.default_worker_config(),
))

for batch in loader:
    print(batch)
    break

```

If you need to handle samples from different datasets differently in your pipeline, you will want to use `subflavors`.
For these and other details, check out the [](metadataset) section. Energon also supports blending by specifying the number of repetitions for each dataset using [](../advanced/epochized_blending).

## Tutorial 4: Distributed Loading

For multi-GPU support, you may need to adapt the worker config.
So far we have only used the default worker config, which you can get by calling {py:func}`WorkerConfig.default_worker_config() <megatron.energon.WorkerConfig.default_worker_config>`.
This default config tries to infer your multi-GPU setup by using `torch.distributed`, which is fine in most cases.
If you are not using any distributed setup, the default config will work, too. In that case, it assumes a single local rank.

However, if you have a more complex multi-node setup and other non-data-parallel stragies, you may need to set it up yourself. 
The following example shows how it could be set.

```python
from megatron.energon import get_train_dataset, get_loader, WorkerConfig
import torch.distributed as dist

worker_config = WorkerConfig(
    rank=SET_YOUR_GLOBAL_DATA_RANK_HERE,
    world_size=SET_YOUR_GLOBAL_WORLD_SIZE_HERE,
    num_workers=2,
)

loader = get_loader(get_train_dataset(
    'coyo-coco-dataset.yaml',
    batch_size=4,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
    worker_config=worker_config,
))

for batch in loader:
    print(batch)
    break
```

For a detailed discussion on parallelism, check out [](../advanced/parallelism).

## Tutorial 5: Saving and Restoring

For saving and restoring the state (e.g. when pausing and resuming training), the loader must be instantiated with the savable loader.

```python
from megatron.energon import get_train_dataset, get_savable_loader, WorkerConfig

ds = get_train_dataset(
    'coyo-coco-dataset.yaml',
    batch_size=4,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
    worker_config=WorkerConfig.default_worker_config(),
)

# Must use the savable loader here. This provides methods to save
# and load the state of the data loader
loader = get_savable_loader(ds)

for i, batch in zip(range(10), loader):
    print(batch)
    break

# Save the state
state = loader.save_state_rank()
# Could save the state now using torch.save()

# ... when loading:
# Could load the state with torch.load()

# Restore the state for a new loader
ds = get_train_dataset(
    'coyo-coco-dataset.yaml',
    batch_size=4,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
    worker_config=WorkerConfig.default_worker_config(),
)
loader = get_savable_loader(ds)
loader.restore_state_rank(state)
```

We provide code for different scenarios for saving and loading in distributed settings especially in the section [](save_restore).

## More Features

Check out the topics in Advanced Usage for details on specific features.
