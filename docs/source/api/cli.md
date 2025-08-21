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

(energon-prepare)=
## energon prepare

An interactive tool to generate metadata for your existing webdataset (or [jsonl dataset](create-jsonl-dataset)).
This will help make the dataset compliant with our [format](data-on-disk).

The tool will ask you for a train/val/test split and how to assign the webdataset fields to the
fields of the corresponding sample type in Energon.

See [Data Preparation](../basic/data_prep) for more details on how to use this command.


## energon info

Prints information about the dataset such as overall number of samples and size.
It also prints the energon version that was used to prepare the dataset, if a recent version was used.


## energon lint

You can execute this tool on the prepared dataset to check if the data is valid and loadable.
It will report any problems such as non-readable images.

(energon-mount)=
## energon mount

Use this to mount your [prepared dataset](../basic/data_prep) as a virtual read-only filesystem and inspect it using `ls` or other file browsing tools.
It is as simple as running

```shell
energon mount /PATH/TO/DATASET ./MY_MOUNT_POINT
```

This will leave the process in the foreground and the mount will exist as long as the program is running.
If you want to detach the process to the background, use the `-d` or `--detach` flag.


Two modes are supported by `energon mount`:

|     | Flat mode (default)  | Sample folder mode (flag `-s`)  |
| --- | --- | --- |
| Description  | All files from all shards listed at<br/>the root of the mount point.  | One folder per sample key,<br/>each folder containing files<br/>named by the sample part extension   |
| Example      | `001.jpg`<br/>`001.txt`<br/>`002.jpg`<br/>`002.txt`<br/>`...`  | `001/`<br/>`┣ jpg`<br/>`┗ txt`<br/>`002/`<br/>`┣ jpg`<br/>`┗ txt`<br/>`...`   |

```{warning}
You should not use the same sample keys in multiple shards of the same dataset.
If you do, `energon mount` will not work as intended and it will display WARNING files in the virtual mount.
```


## energon preview

This command will load a dataset and display samples one-by-one on the console.
Note that this will not work for datasets with non-standard flavors or crude datasets.
