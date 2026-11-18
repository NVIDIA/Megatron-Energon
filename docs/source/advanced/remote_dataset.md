<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Remote Dataset

Megatron Energon supports the use of remote datasets. Since version >5.2.0, Energon file access is based on [Multi Storage Client (MSC)](https://github.com/NVIDIA/multi-storage-client).
This means you can train or validate with your data right from any storage by simply swapping the dataset path for a so-called _MSC URL_.

When opening TAR and JSONL offset indexes on plain POSIX storage, Energon gets the index
size from the open file descriptor to avoid an extra metadata lookup. Object storage and
profiles with a metadata provider, including composite profiles, use the size reported by MSC.

## Prerequisites

For using a remote dataset, install energon with one or more of the extras:
* `s3`
* `aistore`
* `azure-storage-blob`
* `google-cloud-storage`
* `oci`

like this:
```sh
pip install megatron-energon[s3,oci]
```

Set up the msc config as described in [Multi Storage Client documentation](https://nvidia.github.io/multi-storage-client/).
You can also use the rclone config with msc, as was described prior to 5.2.0.

For fast data loading we recommend to activate MSC local caching:

```yaml
cache:
  size: 500G
  use_etag: true
  eviction_policy:
    policy: "fifo"
    refresh_interval: 3600
  cache_backend:
    cache_path: /tmp/msc_cache # prefer to use local NVMe, but Lustre path also works
```

And point MSC to the config with 

```sh
export MSC_CONFIG=/path/to/msc_config.yaml
```


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
* As arguments to `energon prepare`, `energon prepare-media`, or `energon lint`. Note that those may be slow for remote locations.
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

## DSS URLs

In environments that provide datasets through the NVIDIA Dataset cache layout, Energon can also resolve DSS paths:

```
dss://DATASET_NAME@VERSION/path/inside/dataset
```

The dataset name and version are required. The path is resolved under `NVDATASET_CACHE_DIR`, so that environment variable must point to the local dataset cache root before loading DSS paths.

## Frozen construction metadata

Applications may prepare a versioned metadata package for a frozen dataset
revision and serve it while constructing a loader:

```python
from megatron.energon.epathlib.metadata_snapshot import (
    MetadataSnapshot,
    use_metadata_snapshot,
)

snapshot = MetadataSnapshot.load(
    "/local/loader-metadata.json",
    sha256=expected_digest,
    recipe_identity=expected_recipe_identity,
)
with use_metadata_snapshot(snapshot):
    dataset = build_dataset()
# Start iteration and workers only after leaving the context.
```

Version 1 packages contain `format_version`, `recipe_identity`, base64-encoded
`contents`, and maps named `file_checks`, `sizes`, and `jsonl_sizes`. Keys are
logical EPath strings; `jsonl_sizes` contains terminal byte offsets. A producer
must capture these values from the intended immutable dataset revision. The
loader rejects a mismatched package digest or recipe identity.

File checks must be JSON booleans. Sizes and terminal byte offsets must be
nonnegative integers. The loaded mappings are read-only. A recorded index size
takes precedence over the POSIX descriptor size inside the snapshot context.

Recorded reads receive independent streams. Unrecorded paths retain their normal
storage backend. Leaving the context restores ordinary access, including after
an exception. Keep media payloads outside the package and rebuild it after any
metadata revision. Digest validation identifies the selected package; it does not
check whether source files have changed. Applications own capture, distribution,
and snapshot selection; this API does not provide an automatic cache or launcher.
