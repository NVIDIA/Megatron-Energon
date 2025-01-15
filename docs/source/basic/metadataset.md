<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Metadataset

Metadatasets allow combining datasets together in a variety of ways by using a yaml config file.
This is useful for example if you want to mix multiple datasets together, and especially if you want to reuse that combination.

To create a metadataset, you simply create a yaml file of the following format.
Example `demo-metadataset.yaml` file:

```yaml
__module__: megatron.energon
__class__: MetadatasetV2
splits:
  # Split defines the train/val/test subsets composition 
  train:
    # Mix the following datasets
    blend:
      - weight: 5
        path: ./coco
      - weight: 2
        path: ./coyo
      - weight: 1
        path: ./other
  val:
    # Only use coco-val for val
    path: ./coco
  test:
    # Only use coyo-test for test
    path: ./coyo
```


In the above example, we create a blend of three datasets. Out of the yielded training samples, 62.5% ({math}`=\frac{5}{8}`) will come from `./coco`, 25% from `./coyo` and 12.5% from `./other`.
Note that the relative paths in the metadataset are relative to the location of the metadataset file. Absolute paths are allowed but won't work for object storage.

To use the metadataset in your loader, simply load it with {py:func}`get_train_dataset <megatron.energon.get_train_dataset>` instead of a normal energon dataset:
```python
from megatron.energon import get_train_dataset

ds = get_train_dataset(
    'demo-metadataset.yaml',
    batch_size=4,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
)

for batch in ds:
    print(batch)
    break

```

Here is another example that takes both the training and the validation set of coyo into the blended training data (with different weights though):

```yaml
__module__: megatron.energon
__class__: MetadatasetV2
splits:
  # Split defines the train/val/test subsets composition 
  train:
    # Mix the following datasets
    blend:
      - weight: 5
        path: ./coco
      - weight: 2
        path: ./coyo
        split_part: train
      - weight: 1
        path: ./coyo
        split_part: val  # <-- Takes the val set of coyo into the train split
  val:
    # Only use coco-val for val
    path: ./coco
  test:
    # Only use coyo-test for test
    path: ./coyo
```

Actually `split_part: train` is the default, so there's no need to explicitely specify that.
When referring to datasets under `val:` obviously `split_part: val` is the default.

Energon also supports blending by specifying the number of repetitions for each dataset using [Epochized Blending](../advanced/epochized_blending).

(sect-subflavors)=
## Subflavors

Subflavors are a way to *tag* samples that come from different origins so that they can still be differentiated after blending.
Even when blending many datasets together, you might want to handle some of them differently in your [Task Encoder](task_encoder).
For example when doing OCR, you might have one dataset with full pages of text and one with only paragraphs. In your task encoder you could decide to augment the images differently.

Here is a modified example of the above `metadataset.yaml` config file that adds some subflavors:
```yaml
__module__: megatron.energon
__class__: MetadatasetV2
splits:
  # Split defines the train/val/test subsets composition 
  train:
    # Blend the following datasets
    blend:
      - weight: 5
        path: ./coco
        # Set the __subflavors__ property of the samples
        subflavors:
          augmentation_type: small_images
          text_length: short
      # Combine coyo-train and coyo-val
      - weight: 2
        path: ./coyo
        split_part: train
        # Set the __subflavors__ property of the samples
        subflavors:
          augmentation_type: large_images
          text_length: short
      - weight: 1
        path: ./coyo
        split_part: val
        # Set the __subflavors__ property of the samples
        subflavors:
          augmentation_type: large_images
          text_length: short
  # For val and test, blending will actually concatenate the datasets
  val:
    # Only use coco val for val
    path: ./coco
    subflavor: small_images
    subflavors:
      augmentation_type: small_images
      text_length: short
  test:
    path: ./coyo
```

In the above example, the coco training samples will now have the subflavor `augmentation_type` set to `small_images` while the samples from coyo, will have that property set to `large_images`.

Note that subflavors are entirely custom and you can use any name and any value for them, for example `foo: bar`
In the code they will be passed around as a dictionary.


## Classes
* {py:class}`DatasetLoaderInterface <megatron.energon.DatasetLoaderInterface>`: Common interface for dataset loaders. Provides methods for constructing/loading the actual train- or val-mode dataset.
  * {py:class}`MetadatasetV2 <megatron.energon.MetadatasetV2>`: The metadataset loader using the yaml example above. Blends datasets for train-mode, and concatenates for val-mode.
  * {py:class}`DatasetLoader <megatron.energon.DatasetLoader>`: The dataset loader using a dataprepped folder (containing `.nv-meta` folder).

## Functions
* {py:func}`get_train_dataset <megatron.energon.get_train_dataset>`: Returns the train-mode (meta)dataset.
* {py:func}`get_val_dataset <megatron.energon.get_val_dataset>`: Returns the val-mode (meta)dataset.
