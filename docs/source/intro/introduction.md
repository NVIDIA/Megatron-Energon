<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# General

Megatron-Energon is a data loader that works best with your [Megatron](https://github.com/NVIDIA/Megatron-LM) project.
However, you can use it in any of your PyTorch-based deep learning projects.

What can it offer compared to other data loaders?

The most important features are:

* Comes with a standardized WebDataset-based format on disk
* Optimized for high-speed multi-rank training
* Can handle very large datasets
* Can easily mix and blend multiple datasets
* Its state is savable and restorable (deterministic resumability)
* Handles various kinds of multi-modal data even in one training run

Energon also comes with a command line tool that you can use to prepare your datasets.
