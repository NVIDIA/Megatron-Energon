<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Command-Line Interface

After you [installed](../intro/installation) energon, a script called `energon` will be added to your PATH.
It provides commands to prepare, preview, or lint datasets on disk.

Here's a simple example:

```shell
energon prepare /mnt/data/my_captioning_webdataset
```

The above command will scan your existing off-the-shelf [web dataset](https://webdataset.github.io/webdataset/)
and add the [needed metadata](data-on-disk) to make it compatible with Energon. 

Below, you can see the available sub-commands under `energon`.


```{eval-rst}
.. click:: megatron.energon.cli.main:main
   :prog: energon
   :nested: short
```

(energon_data_prepare)=
## energon prepare

An interactive tool to generate metadata for your existing webdataset.
This will help make the dataset compliant with our [format](data-on-disk).

The tool will ask you for a train/val/test split and how to assign the webdataset fields to the
fields of the corresponding sample type in Energon.

See [Data Preparation](../basic/data_prep) for more details on how to use this command.


## energon info

Prints information about the dataset such as overall number of samples and size.


## energon lint

You can execute this tool on the prepared dataset to check if the data is valid and loadable.
It will report any problems such as non-readable images.


## energon preview

This command will load a dataset and display samples one-by-one on the console.
Note that this will not work for datasets with non-standard flavors or crude datasets.
