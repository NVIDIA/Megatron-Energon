<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Filter Indexes

Filter indexes select a precomputed set of samples from a prepared dataset. The same
filter can be used with different splits, including splits that reorder shards.

## Creating a Filter

Pass the original global sample indexes to keep, in strictly increasing order:

```python
from megatron.energon import build_filter_index
from megatron.energon.epathlib import EPath

build_filter_index(EPath("/data/my_dataset"), "quality", [1, 5, 9, 10, 13])
```

Indexes refer to the full dataset before split selection. For manifest datasets, shard
order comes from `.nv-meta/.info.json` or `.info.yaml`. A direct JSONL file is one shard.
You can pass `shards=` to choose an explicit source order or subset.

If you already have indexes within each shard, use:

```python
from megatron.energon import build_filter_index_from_shard_indexes

build_filter_index_from_shard_indexes(
    EPath("/data/my_dataset"),
    "quality",
    {
        "parts/data-0.tar": [1, 5, 9],
        "parts/data-1.tar": [0, 3],
    },
)
```

Indexes must be strictly increasing within each shard and within its original sample
count. Omitted shards retain no samples. Use `overwrite=True` to replace an existing filter.

## Loading a Filter

Pass `filter_name` when loading the dataset:

```python
from megatron.energon.dataset_config import get_dataset_from_config

factory = get_dataset_from_config(
    "/data/my_dataset",
    split_part="train",
    training=True,
    worker_config=worker_config,
    filter_name="quality",
)
```

In a recipe, use `filter`:

```yaml
splits:
  train:
    path: /data/my_dataset
    filter: quality
```

## Saved Files and Stale Filters

For manifest datasets, filters are stored in `.nv-meta/`:

- `filter_<name>.json`: shard order, original and retained counts, and primary-file sizes.
- `filter_<name>.idx`: the kept original sample indexes, stored as `uint64` values.

For direct files, the names are `<file-name>.<name>.json` and `<file-name>.<name>.idx`,
next to the data file. Example JSON:

```json
{
  "version": 3,
  "shards": [
    {
      "name": "parts/data-0.tar",
      "sample_count": 100,
      "filtered_sample_count": 20,
      "byte_size": 102400
    }
  ]
}
```

Loading rejects filters if recorded shards are missing or their original sample counts
or primary-file sizes have changed. Rebuild the filter after updating the dataset.
Version-1 and version-2 filters also require rebuilding.

These checks do not validate companion-file contents or detect content changes that
preserve both primary-file sizes and sample counts.
