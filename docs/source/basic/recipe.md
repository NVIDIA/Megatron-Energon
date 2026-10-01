<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Recipe

Recipes allow combining datasets together in a variety of ways by using a yaml config file.
This is useful for example if you want to mix multiple datasets together, and especially if you want to reuse that combination.

To create a recipe, you simply create a yaml file of the following format.
Example `demo-recipe.yaml` file:

```yaml
__module__: megatron.energon
__class__: Recipe
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


## Migrating from Metadataset

`Recipe` replaces both legacy Metadataset formats. The migration depends on
which class name the YAML uses.

### Migrating `MetadatasetV2`

`MetadatasetV2` uses the same YAML structure as `Recipe`. Change only the class:

```diff
 __module__: megatron.energon
-__class__: MetadatasetV2
+__class__: Recipe
 splits:
   train:
     blend:
       - weight: 1
         path: ./dataset
```

The {py:class}`MetadatasetV2 <megatron.energon.MetadatasetV2>` Python symbol
remains as a deprecated compatibility alias, but new recipes should not depend
on it.

### Migrating legacy `Metadataset` v1

The v1 `Metadataset` class and the `megatron.energon.metadataset` Python package
have been removed. A v1 weighted `datasets` list becomes a Recipe `blend` list.
For example, convert:

```yaml
__module__: megatron.energon
__class__: Metadataset
splits:
  train:
    datasets:
      - weight: 2
        path: ./dataset-a
        subflavor: source-a
      - weight: 1
        path: ./dataset-b
        split_part: val
```

to:

```yaml
__module__: megatron.energon
__class__: Recipe
splits:
  train:
    blend:
      - weight: 2
        path: ./dataset-a
        tags:
          __subflavor__: source-a
      - weight: 1
        path: ./dataset-b
        split_part: val
```

Apply these conversions throughout nested files:

| Metadataset v1 | Recipe |
| --- | --- |
| `__class__: Metadataset` | `__class__: Recipe` |
| `splits.<name>.datasets` | `splits.<name>.blend` |
| `subflavor: value` | `tags: {__subflavor__: value}` |
| `megatron.energon.metadataset.*` imports | `megatron.energon.recipe.*` imports |

Existing `subflavors` entries remain supported as a compatibility alias for
`tags`; new recipes should use `tags`. `split_part`, `dataset_config`,
`split_config`, weights, and `shuffle_over_epochs_multiplier` also remain
supported. Relative paths are still resolved relative to the containing YAML
file.

After conversion, validate every top-level and nested recipe:

```shell
energon lint /path/to/recipe.yaml
```

Renaming a YAML file from `metadataset.yaml` is optional; detection uses its
contents rather than its filename. If training resumes after changing the recipe
composition, follow {ref}`checkpoint-recipe-migration` instead of
performing an exact restore.


In the above example, we create a blend of three datasets. Out of the yielded training samples, 62.5% ({math}`=\frac{5}{8}`) will come from `./coco`, 25% from `./coyo` and 12.5% from `./other`.
Note that the relative paths in the recipe are relative to the location of the recipe file. Absolute paths are allowed but won't work for object storage.

By default, blend weights target the number of samples yielded from each dataset. To make weights
target a task-defined unit, set `blend_weight_unit` on the blend and register a matching metric on
the task encoder with `@sample_size_metric`. For example, `blend_weight_unit: tokens` makes the
weights target token volume if the task encoder registers a `tokens` sample size metric:

```yaml
__module__: megatron.energon
__class__: Recipe
splits:
  train:
    blend_weight_unit: tokens
    blend:
      - weight: 5
        path: ./coco
      - weight: 2
        path: ./coyo
```

The built-in unit `samples` is reserved for sample-count blending and is used when
`blend_weight_unit` is omitted.

To use the recipe in your loader, simply load it with {py:func}`get_train_dataset <megatron.energon.get_train_dataset>` instead of a normal energon dataset:
```python
from megatron.energon import get_train_dataset

ds = get_train_dataset(
    'demo-recipe.yaml',
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
__class__: Recipe
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

(shuffle-over-epochs)=
## Shuffling over Epochs

`shuffle_over_epochs_multiplier` controls shard-slice shuffling for training
datasets. It may be passed to {py:func}`get_train_dataset
<megatron.energon.get_train_dataset>` and set on Recipe nodes or dataset
references.

| Value | Behavior |
| --- | --- |
| `None` | Do not shuffle shard slices. |
| `1` | Shuffle without replacement so each slice is visited once per epoch. |
| Integer greater than `1` | Shuffle the slices from that many epochs together, without replacement within that window. |
| `-1` | Draw shard slices with replacement, producing an effectively infinite sequence of epochs. |

For example, the following training reference continuously samples shard slices
with replacement:

```yaml
__module__: megatron.energon
__class__: Recipe
splits:
  train:
    path: ./large-dataset
    shuffle_over_epochs_multiplier: -1
```

The setting applies to shard slices, not to arbitrary individual samples.
Samples inside a selected slice are consumed by the dataset sampler, while
`parallel_shard_iters` controls how many selected slices may be active in
parallel.

In nested Recipes, multipliers are merged on the path to each leaf:

* positive integers multiply, so outer `2` and inner `3` produce `6`,
* `-1` takes precedence over positive values, and
* `None` disables shuffling and takes precedence over every other value.

The loader argument participates in the same merge. This lets a caller enlarge
the shuffle window for a whole Recipe while individual references refine it.

```{admonition} Reproducibility
:class: important

The effective shuffle setting is part of the stream definition. Keep it, the
seed, dataset paths, split assignments, subsets, and blend configuration stable
for an exact checkpoint restore. In particular, changing between finite
without-replacement shuffling and `-1` changes which slices may be repeated or
skipped.
```

(sect-subflavors)=
(sect-tags)=
## Tags

Tags are custom key/value attributes attached to samples from a dataset or recipe. They can
differentiate samples after blending and drive cooking, packing, routing, or application-specific
processing.
Even when blending many datasets together, you might want to handle some of them differently in your [Task Encoder](task_encoder).
For example when doing OCR, you might have one dataset with full pages of text and one with only paragraphs. In your task encoder you could decide to augment the images differently.

The effective mapping is available as {py:attr}`Sample.__tags__
<megatron.energon.Sample.__tags__>`. For backward compatibility, recipe and dataset
configuration also accepts `subflavors`, and samples expose `__subflavors__` as a
read/write alias for `__tags__`. Specify only one name at a time; providing both is
an error. Dataset factories and references similarly retain a `subflavors` alias,
and `Cooker.has_subflavors` aliases `Cooker.has_tags`.

Here is a modified example of the above `recipe.yaml` config file that adds some tags:
```yaml
__module__: megatron.energon
__class__: Recipe
splits:
  # Split defines the train/val/test subsets composition 
  train:
    # Blend the following datasets
    blend:
      - weight: 5
        path: ./coco
        # Set the __tags__ property of the samples
        tags:
          augmentation_type: small_images
          text_length: short
      # Combine coyo-train and coyo-val
      - weight: 2
        path: ./coyo
        split_part: train
        # Set the __tags__ property of the samples
        tags:
          augmentation_type: large_images
          text_length: short
      - weight: 1
        path: ./coyo
        split_part: val
        # Set the __tags__ property of the samples
        tags:
          augmentation_type: large_images
          text_length: short
  # For val and test, blending will actually concatenate the datasets
  val:
    # Only use coco val for val
    path: ./coco
    tags:
      augmentation_type: small_images
      text_length: short
  test:
    path: ./coyo
```

In the above example, the coco training samples have the tag `augmentation_type` set to `small_images`, while the samples from coyo have it set to `large_images`.

Note that tags are entirely custom and you can use any name and any value for them, for example `foo: bar`
In the code they will be passed around as a dictionary.

## Auxiliary Data

In the recipe, you can also specify auxiliary data for each blended dataset.
Please check out [Auxiliary Data](aux-data), for more information.

## Classes
* {py:class}`DatasetLoaderInterface <megatron.energon.DatasetLoaderInterface>`: Common interface for dataset loaders. Provides methods for constructing/loading the actual train- or val-mode dataset.
  * {py:class}`Recipe <megatron.energon.Recipe>`: The recipe loader using the yaml example above. Blends datasets for train-mode, and concatenates for val-mode.
  * {py:class}`DatasetLoader <megatron.energon.DatasetLoader>`: The dataset loader using a dataprepped folder (containing `.nv-meta` folder).

## Functions
* {py:func}`get_train_dataset <megatron.energon.get_train_dataset>`: Returns the train-mode dataset or recipe.
* {py:func}`get_val_dataset <megatron.energon.get_val_dataset>`: Returns the val-mode dataset or recipe.
