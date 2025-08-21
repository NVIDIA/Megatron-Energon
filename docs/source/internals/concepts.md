<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# High-Level Concepts

As shown in [Quickstart](../basic/quickstart), a few simple lines of code can be used to fire up
Energon and iterate samples from a previously created dataset.

But behind the simple calls to `get_train_dataset` and `get_loader` and while iterating the data,
a lot of things are happening, and in this section, we will explain how this machinery works.

**Synchronous vs. Asynchronous.**
Energon is made for highly-parallelized data loading across multiple nodes, ranks and workers.
When using multiple workers, energon supports both multi-processing as well as free-threaded (GIL-less)
multi-threading.
For an easy-to-follow explanation, we will first focus on the simple synchronous case with a single rank
and no workers and later explain the asynchronous case.

## Datasets and Wrappers

We use the term *dataset* ambiguously. In the user-facing docs, it usually refers to a WebDataset or JSONL
dataset on disk somewhere.
Internally, we say *dataset* to refer to class derived from {py:class}`SavableDataset <megatron.energon.SavableDataset>`.

Energon builds up a data processing pipeline from the datasets, where one dataset serves as the source and transformations
are added on top.

The source dataset is typically the `WebdatasetSampleLoaderDataset`.

TODO:
- Explain three types of dataset: leaf, wrapper, root. And how they pass the samples like a pipeline
  - Mention merging (like batching, packing)
  - Mention splitting (one sample can yield multiple new samples)
  - Mention skipping (raise `SkipSample`)
- Give an example of what the tree might look like
- Look at a full samples and all the keys, restore keys etc. it contains
- Quick example on how a new dataset could be added
  - Use the example to highlight 3 types of class variables (shared state, local savable state, local volatile state) but no details yet, refer to sections below

## Building up the Dataset Pipeline

TODO:
- Explain how the dataset classes are initialized and loaded from `dataset.yaml` or a `metadataset.yaml`
- Explain how the `...Factory` classes work and why we need them (init vs. build)
- Refer to asynch below for explanation on how it works with multiple workers.

## Randomness and Determinism

TODO:
- Explain how user must deal with randomness in their task encoder
- Explain where seeds are set, saved, restored
- Explain `@stateless` with args

## Saving and Restoring

TODO:
- Refer to save_restore.md
- Explain the state variables of datasets in more detail
- Look at an example of a stored state on disk
- Explain how the redistribution for reproducible scaling works

## Asynchronous Data Loading

TODO:
- WorkerConfig
- Multi-threading vs. Multi-processing
  - A bit of history, and how we used torch dataloader with forking before
- ...
