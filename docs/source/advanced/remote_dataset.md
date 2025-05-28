<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Remote Dataset

Since version 2.0.0, Megatron Energon supports the use of remote datasets. Since version >5.2.0, Energon file access is based on [Multi Storage Client (MSC)](https://github.com/NVIDIA/multi-storage-client).
This means you can train or validate with your data right from any storage by simply swapping the dataset path for a so-called _MSC URL_.

## Prerequisites

For using a remote dataset, install one or more of the extras:
* `s3`
* `aistore`
* `azure-blob-storage`
* `google-cloud-storage`
* `oci`

like this:
```sh
pip install megatron-energon[s3,oci]
```

Set up the msc config as described in [Multi Storage Client documentation](https://nvidia.github.io/multi-storage-client/).

You can also use the rclone config with msc, as was described prior to 5.2.0.

## The URL syntax

The syntax is a simple as 

```
msc://CONFIG_NAME/PATH
```

For example:

```
msc://coolstore/mainbucket/datasets/somedata
```

You can use this URL instead of paths to datasets in

* Functions like `get_train_dataset`, `get_val_dataset`
* Inside [metadataset](../basic/metadataset) specifications
* As arguments to `energon prepare` or `energon lint`. Note that those may be slow for remote locations.
* Or as a path to [`energon mount`](energon-mount) to locally inspect your remote dataset 😎

Example usage:

```python
ds = get_train_dataset(
    'msc://coolstore/mainbucket/datasets/somedata',
    batch_size=1,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
)
```
