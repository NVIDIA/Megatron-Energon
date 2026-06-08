<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Filter Indexes

Filter indexes restrict a random-access dataset to a precomputed set of samples without
running a predicate while training. The filter lives in the dataset metadata directory
(`.nv-meta`).

For manifest/directory datasets, the sidecars are:

- `filter_<name>.json`: filter metadata with the sidecar format version and filtered sample counts per shard.
- `filter_<name>.idx`: the kept original sample indexes, stored as a raw `uint64` translation table.

For single-file datasets such as `.jsonl`, `.bin`, or `.parquet`, the sidecars are:

- `<file-name>.<name>.json`
- `<file-name>.<name>.idx`

The translation table stores original global sample indexes in increasing order. Energon uses
the filtered per-shard counts from the JSON sidecar for worker sharding. At read time, the
filtered index is translated back to the original sample index before delegating to the dataset
reader. The translation table is opened lazily, so loading filter metadata does not require
opening the `.idx` file until samples are actually read.

## Building a Filter

Energon exposes a low-level Python API for building filter indexes from explicit kept
indexes. It does not provide a data-format-specific `energon filter` command.

```python
from megatron.energon import build_filter_index
from megatron.energon.epathlib import EPath

build_filter_index(
    EPath("/data/my_dataset"),
    "quality_v1",
    [1, 5, 9, 10, 13],
)
```

Global indexes must be strictly increasing and within the total original sample count.
For prepared datasets, shard counts are read from `.nv-meta/.info.json`. Pass
`shards` explicitly only for datasets without that metadata, such as direct file-backed
datasets. If you already have shard-local indexes for a manifest dataset, use
`build_filter_index_from_shard_indexes` instead:

```python
from megatron.energon import build_filter_index_from_shard_indexes

build_filter_index_from_shard_indexes(
    EPath("/data/my_dataset"),
    "quality_v1",
    {
        "parts/data-0.tar": [1, 5, 9],
        "parts/data-1.tar": [0, 3],
    },
)
```

Shard-local indexes must be strictly increasing and within the original count for each
shard. The same APIs can be used for prepared WebDataset, prepared JSONL shard datasets,
prepared Parquet shard datasets, single-file Parquet datasets, and bin-idx datasets. For
single-file datasets such as `.bin`, `.jsonl`, and `.parquet`, the shard name is the file name.

## Loading a Filter

Pass the filter name when loading a dataset:

```python
from megatron.energon.dataset_config import get_dataset_from_config

dataset = get_dataset_from_config(
    "/data/my_dataset",
    split_part="train",
    training=True,
    worker_config=worker_config,
    filter_name="quality_v1",
)
```

In a metadataset v2 YAML file, use the `filter` field:

```yaml
splits:
  train:
    path: /data/my_dataset
    filter: quality_v1
```
