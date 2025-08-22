<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Megatron-Energon Documentation

This is the documentation of Megatron's multi-modal data loader "Energon".

We recommend getting started in the [Introduction](intro/introduction) section, which explains what Energon is and how to install it.

Once installed, check out the **Basic Usage** section starting with [Quickstart](basic/quickstart) for some basic examples and tutorials.
Some underlying concepts, will be explained in the rest of that section.

For specific use cases and advanced usage, please read **Advanced Usage**.

In the end you will also find some documentation on how to interface with energon programmatically and how to contribute to the code base.

```{toctree}
---
caption: Introduction
maxdepth: 2
---

intro/introduction
intro/installation
```


```{toctree}
---
caption: Basic Usage
maxdepth: 2
---
basic/quickstart
basic/data_prep
basic/data_decoding
basic/basics_flow
basic/task_encoder
basic/metadataset
basic/save_restore
basic/glossary
```


```{toctree}
---
caption: Advanced Usage
maxdepth: 2
---
advanced/remote_dataset
advanced/crude_datasets
advanced/custom_sample_loader
advanced/repro_scaling
advanced/packing
advanced/grouping
advanced/joining_datasets
advanced/subsets
advanced/epochized_blending
advanced/custom_blending
advanced/parallelism
```


```{toctree}
---
caption: API
maxdepth: 2
---
api/modules
api/cli
```


```{toctree}
---
caption: Internals
maxdepth: 2
---
internals/contrib_guidelines
internals/concepts
internals/code_structure
```

# Indices and tables

- [](genindex)
- [](modindex)
