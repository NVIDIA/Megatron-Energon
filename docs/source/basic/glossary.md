<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Glossary

* **Batch Grouping**
    * Allows you to programmatically decide which samples (out of a buffer) will be put into one batch. See [](../advanced/grouping.md).
* **Cooking**
    * Used to transform crude (raw) samples into a populated instance of a sample data class.
* **Crude Dataset**
    * An energon dataset, that does not yield a readily-populated sample (instance of dataclass), but a raw dict.
    * A cooker is used to handle this transformation in the user's custom task encoder. See [](crude-data).
* **Grouping**
    * See "Batch Grouping"
* **Monolithic Dataset**
    * The simple form of putting all your text and media data into the same WebDataset (see [](monolithic-dataset)).
    * The other option is to use a "Polylithic Dataset"
* **Packing**
    * For Energon, with "packing" we mean "sequence packing". See "Sequence Packing" below.
* **Polylithic Dataset**
    * Used to split the text-based data from the (usually larger) media-data.
    * Each modality will be put in its own dataset and one can refer to the other by file names.
    * For more information see [](polylithic-dataset)
* **Sequence Packing**
    * A method to better utilize the available context length / sequence length of a model and reduce padding.
    * Explained in [](../advanced/packing.md)
* **Task Encoder**
    * An Energon-specific concept: The TaskEncoder is a user-defined class to customize the steps of the data flow pipeline.
    * See [](../basic/basics_flow.md) and [](../basic/task_encoder.md)
* **WebDataset**
    * A file-format to store your dataset on disk, based on TAR files. See [https://github.com/webdataset/webdataset](https://github.com/webdataset/webdataset).
    * Energon's dataset format builds on WebDataset and extends it with additional files, see [](data-on-disk).
