<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Contribution Guidelines

If you want to contribute to this repository please adhere to the following guidelines

- Always use [black](https://pypi.org/project/black/) to format your code before committing
- Python `@dataclass` and `NamedTuple` are preferred over dictionaries, which don't allow for IDE
  auto-completion and type checking
- User-exposed classes and methods should be documented in Google-style docstrings that are parsed by sphinx
  and end up in this documentation
- Breaking changes should be marked in the message of pull requests:
  - `CHECKPOINT BREAKING CHANGE`: When the save/restore structure changed incompatibly (check test `test_metadataset:TestDataset.test_save_restore_state_train`)
  - `RANDOMNESS BREAKING CHANGE`: When the randomness changed (check tests `test_dataset:TestDataset.test_current_batch_index_generator`, `test_dataset:TestDataset.test_current_batch_index`, maybe more)
  - `API BREAKING CHANGE`: When the external programming api changed incompatibly
  - `DATASET CONFIG BREAKING CHANGE`: When the dataset config (`.nv-meta` folder) changed incompatibly
  - `METADATASET CONFIG BREAKING CHANGE`: When the metadataset config changed
- In a release, all breaking changes except checkpoint lead to a new major version.
