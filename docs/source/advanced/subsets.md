<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Dataset Subsets

Dataset subsets allow restricting a dataset (or parts of a metadataset hierarchy) to a specific
portion of the available samples. This is useful for rapid prototyping, ablation studies, or
constructing disjoint train/validation/test splits that differ from the original dataset
configuration.

A subset is defined by a two-element `range` list. Each element can be either

* a **percentage** string (e.g. `"0%"`, `"12.5%"`, `"100%"`) – interpreted relative to the parent
  dataset size, or
* an **absolute** integer – interpreted as a sample index. Absolute indices are only allowed for
  *leaf* datasets (`path` to a prepared dataset containing `.nv-meta`).

Percentages are inclusive of the lower bound and exclusive of the upper bound. If the upper bound
is omitted (`null`), the subset continues to the end of the dataset.

## Basic example

The snippet below keeps the first 80 % of *COYO* for training while evaluating on the remaining
20 %. Note how the `subset` key is placed directly next to the corresponding `path`.

```yaml
__module__: megatron.energon
__class__: MetadatasetV2
splits:
  train:
    path: ./coyo
    subset:
      range: ["0%", "80%"]
  val:
    path: ./coyo
    split: train
    subset:
      range: ["80%", "100%"]
```

## Nested subsets and merging rules

Subsets can appear at any level that ultimately yields samples
(direct `path` reference to a prepared dataset containing `.nv-meta`, `join`, `blend`, `blend_epochized`).
When multiple subsets are nested, the *inner* subset is applied first, then the portion selected by the *outer* subset is applied *within* the already selected range.
For percentages the ranges are composed multiplicatively.

Example: the outer subset `[0%, 50%]` followed by an inner subset `[25%, 75%]` results in the final
range `[25%, 50%]` of the original dataset.

Absolute indices short-circuit merging: they can **only** be specified at the leaf level and must
not be combined with another absolute range farther up the hierarchy.

## Absolute ranges

Absolute indices are handy when exact sample counts are required. The following configuration takes
the first **1000** samples from *COCO* and mixes them with the full *COYO* dataset using
weight-based blending:

```yaml
__module__: megatron.energon
__class__: MetadatasetV2
splits:
  train:
    # The subset is applied to each blended dataset separately.
    # I.e. for the first, the sample range is [0, 500], for the second the range is [0%, 50%]
    subset: {range: [0%, 50%]}
    blend:
      - weight: 1.0
        path: ./coco
        subset:
          # Take exactly 1000 samples (indices 0-999)
          range: [0, 1000]
      - weight: 1.0
        path: ./coyo
```

## Python usage

No API changes are required on the Python side – subsets are fully specified in the YAML. Simply
load the dataset with the regular helpers.
