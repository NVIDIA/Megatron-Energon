<a name="top"></a>

<div align="center">
  <a href="#">
    <img src="docs/source/images/energon3_darkbg_border.png" alt="Logo" height="384">
  </a>

<h3 align="center">Megatron Energon</h3>
  <p align="center">
    Megatron's multi-modal data loader
    <br />
    <a href="#">Report Bug</a>
    ·
    <a href="#">Request Feature</a>
  </p>
</div>

# Quickstart
**Megatron Energon** is the multi-modal data loader of Megatron.

It is a pip-installable python package that offers
- dataset-related classes that you can import in your project
- a command line utility for data preprocessing and conversion

This document is just a quick start. Please also check out the [documentation](https://nvidia.github.io/documentation/megatron-energon/).

## Installation 

```shell
pip install megatron-energon
```
Or
```shell
pip install git+https://github.com/NVIDIA/Megatron-Energon.git
```

**NOTE**: We encourage you to install the package (and not just import a local copy). This will ensure you have all the needed dependencies and that you can use the command line tool.

For more details on installing this package, see [here](https://nvidia.github.io/documentation/megatron-energon/).

## Usage of command line tool

After installation, the command `energon` will be available.

Here are some examples for things you can do:

| Command | Description  |
|---|---|
| `energon prepare DATASET_ROOT` | Take an existing WebDataset and add the required yaml files to turn it into an energon-compatible dataset |
| `energon lint DATASET_ROOT` | Verify that the dataset complies with the energon dataset format and that all samples are loadable |


## Usage of the library

To get started, pick a [WebDataset](https://github.com/webdataset/webdataset)-compliant dataset and run `energon prepare DATASET_ROOT` on it, to run the interactive assistant and create the `.nv-meta` folder.

Once done, try to load it from your Python program:

```python
from megatron.energon import get_train_dataset, get_loader


train_loader = get_loader(get_train_dataset(
    '/my/dataset/path',
    batch_size=32,
    shuffle_buffer_size=None,
    max_samples_per_sequence=None,
))

for batch in train_loader:
    # Do something with batch
    # Infer, gradient step, ...
    pass
```

For more details, read the [documentation](https://nvidia.github.io/documentation/megatron-energon/).

Most likely, you'll need your own [task encoder](https://nvidia.github.io/documentation/megatron-energon//task_encoders.html).
