<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(data-preparation)=
# Data Preparation

The aim of data preparation is to convert your data to a format that the energon loader can understand and iterate.
The outcome will be a [WebDataset](https://github.com/webdataset/webdataset) with some extra information stored in a folder called `.nv-meta`.
Below in [](data-on-disk) we explain the details about this format.

These are the typical steps to get your data ready:

1. Create a normal [WebDataset](https://github.com/webdataset/webdataset) from your data
2. Run our preparation tool `energon prepare` to convert to an energon-compatible format

(wds-format)=
## Step 1: Creating a WebDataset

Example for a WebDataset (e.g. image captioning dataset):

```
shards
├── shard_0000.tar
│   ├── sample_0000.jpg
│   ├── sample_0000.txt
│   ├── sample_0000.detail.json
│   ├── sample_0001.jpg
│   ├── sample_0001.txt
│   └── sample_0001.detail.json
├── shard_0001.tar
│   ├── sample_0002.jpg
│   ├── sample_0002.txt
│   ├── sample_0002.detail.json
│   ├── sample_0003.jpg
│   ├── sample_0003.txt
│   └── sample_0003.detail.json
└── ...
```

In the example you can see two shards (i.e. tar files) with multiple samples. Each group of files with the same basename makes one sample.
So `sample_0000.jpg`, `sample_0000.txt` and `sample_0000.detail.json` are three parts that belong to the first sample.

Note that each sample may have a different number of parts, for example some samples may have more images than others.
In this case, they should still have the same basename, for example `sample_0000.img1.jpg` and `sample_0000.img2.jpg`. For an advanced example for interleaved data, check out [this section](interleaved-sample-loader).

The order of samples in the tar file is important. Samples with the same base name (~before the first dot of the filename) must follow each other.
The base name is used to group the samples, i.e. in the example `sample_0000` is the first group name, with the part types `jpg`, `txt`, `detail.json`.

The default behavior of energon is to parse the contents by extensions (e.g. ending on `.json` will automatically use `json.loads`, `.png` will load the image).

### Building a WebDataset using Python
The easiest way to construct a WebDataset from existing data (e.g. from another torch dataset or a folder with files) is to use the ShardWriter from the webdataset library:

```py
import webdataset as wds


if __name__ == '__main__':
    # Wherever your dataset comes from
    my_dataset = ...
  
    with wds.ShardWriter("parts/data-%d.tar", maxcount=10000) as shard_writer:
        for key, data in my_dataset:
            sample = {
                "__key__": key,
                "png": data['image'],
            }
            shard_writer.write(sample)
```


## Step 2: Preparing the Dataset

Once you have a WebDataset ready, you will want to prepare it for use with Energon.
This means adding additional meta data files next to the data.
This step does *not* change or copy the contents of your tar files.

Just run the `energon prepare /path/to/dataset` command, which will interactively walk you through the process.

The command will

* Search for all `*.tar` files in the given folder
* Index them so samples can be accessed randomly
* Ask you how you want to split the data into train/val/test paritions
* Ask you how to decode the data (field map or sample_loader.py)
* store all this information in a subfolder `.nv-meta/`, see details [below](data-on-disk).

### Splitting the dataset into train/val/test

The first thing that the `energon prepare` assistant will ask you, is how you want to split the data by ratios.
However, if you have a pre-determined split, you can also pass that to energon. See the examples below.

#### Example 1: Let energon do the split
```text
shards
├── shard_0000.tar
├── shard_0001.tar
└── ...
```
Commandline:
```
> energon prepare ./
# Exemplary answers to interactive questions:
Ratio: 8,1,1
Dataset class: CaptioningWebdataset
Field map: Yes
  image: jpg
  caption: txt  # if txt contains the caption
# or
  caption: json[caption]  # if .json contains {"caption": "My nice image"}
```

#### Example 2: Presplit shards by prefix
```text
shards
├── train_shard_0000.tar
├── train_shard_0001.tar
├── ...
├── val_shard_0000.tar
├── val_shard_0001.tar
└── ...

```
Commandline:
```
> energon prepare --split-parts 'train:shards/train_.*' --split-parts 'val:shards/val_.*' ./
```

#### Example 3: Presplit shards by folder
```text
shards
├── train
│   ├── shard_00001.tar
│   ├── shard_00001.tar
│   └── ...
├── val
│   ├── shard_00001.tar
│   ├── shard_00001.tar
│   └── ...
└── ...
```
Commandline:
```
> energon prepare --split-parts 'train:shards/train/.*' --split-parts 'val:shards/val/.*' ./
```

### Sample Types

After the split is set up, the assistant will ask you which sample type you want to use.
We provide a set of common sample types such as for image captioning or visual question answering, they are listed below.

If none of these fits, you may need to set up your own new sample type.
Here are your options:

* You have a new type sample which is rather common but not in our list below
  * Please add your type to energon and create a pull request so we can add it
* Your sample type is experimental or used temporarily only
  * You can add the sample type class in your code repository and create the `dataset.yaml` manually, referring to your class with `__class__`

(sect-sample-types)=
#### Available Sample Types

These are the possible integrated types you can currently choose from:

* {py:class}`Sample <megatron.energon.Sample>`: Base dataclass for samples from source webdatasets.
  * Attributes:
    * {py:attr}`__key__: str <megatron.energon.Sample.__key__>`: Unique identifier of the sample within the dataset. Useful for backtracking the source of a single sample.
    * {py:attr}`__key__: str <megatron.energon.Sample.__restore_key__>`: Structured key of the sample, which can be used to regenerate the sample without storing the whole sample.
    * {py:attr}`__subflavor__: str <megatron.energon.Sample.__subflavor__>`: Deprecated.
    * {py:attr}`__subflavors__: dict[str, Any] | None <megatron.energon.Sample.__subflavors__>`: Represents the subflavors (i.e. custom dict data) set for the source dataset (typically in the metadataset).
  * {py:class}`CaptioningSample <megatron.energon.CaptioningSample>`: Represents a sample for captioning
    * Attributes:
      * {py:attr}`image: torch.Tensor <megatron.energon.CaptioningSample.image>`: The input image tensor
      * {py:attr}`caption: str <megatron.energon.CaptioningSample.caption>`: The target caption string
  * {py:class}`ImageSample <megatron.energon.ImageSample>`: Represents a sample which only contains an image (e.g. for reconstruction)
    * Attributes:
      * {py:attr}`image: torch.Tensor <megatron.energon.ImageSample.image>`: The image tensor
  * {py:class}`ImageClassificationSample <megatron.energon.ImageClassificationSample>`: Represents a sample which contains an image with a caption
    * Attributes:
      * {py:attr}`image: torch.Tensor <megatron.energon.ImageClassificationSample.image>`: The image tensor
      * {py:attr}`label: int | None <megatron.energon.ImageClassificationSample.label>`: The label of the sample, as integral representation
      * {py:attr}`label_name: str | None <megatron.energon.ImageClassificationSample.label_name>`: The label of the sample 
  * {py:class}`InterleavedSample <megatron.energon.InterleavedSample>`: Represents a sample which contains interleaved media, such as image and text.
    * Attributes:
      * {py:attr}`sequence: list[torch.Tensor | str] <megatron.energon.InterleavedSample.sequence>`: The interleaved media (either a torch.Tensor or string for text)
  * {py:class}`MultiChoiceVQASample <megatron.energon.MultiChoiceVQASample>`: Represents a sample for visual question answering, with a choice of answers and one correct answer.
    * Attributes:
      * {py:attr}`image: torch.Tensor <megatron.energon.MultiChoiceVQASample.image>`: The input image tensor
      * {py:attr}`context: str <megatron.energon.MultiChoiceVQASample.context>`: The context/question for the image
      * {py:attr}`choices: List[str] | None <megatron.energon.MultiChoiceVQASample.choices>`: The candidate answers
      * {py:attr}`correct_choice_idx: int | None <megatron.energon.MultiChoiceVQASample.correct_choice_idx>`: The index of the correct answer
  * {py:class}`OCRSample <megatron.energon.OCRSample>`: Sample type for optical character recognition.
    * Attributes:
      * {py:attr}`image: str <megatron.energon.OCRSample.image>`: The input image
      * {py:attr}`text: str <megatron.energon.OCRSample.text>`: The text string for the whole image
      * {py:attr}`block_boxes: torch.Tensor | None <megatron.energon.OCRSample.block_boxes>`: The bounding boxes of the block in the image float(N, 4|5<x,y,w,h,confidence>)
      * {py:attr}`block_classes: torch.Tensor | list[str] | None <megatron.energon.OCRSample.block_classes>`: The classes of th blocks
      * {py:attr}`block_text: torch.Tensor | None <megatron.energon.OCRSample.block_text>`: The text content of the blocks
      * {py:attr}`lines_boxes: torch.Tensor | None <megatron.energon.OCRSample.lines_boxes>`: The bounding boxes of the text lines
      * {py:attr}`lines_text: list[str] | None <megatron.energon.OCRSample.lines_text>`: The text content of the text lines
      * {py:attr}`words_boxes: torch.Tensor | None <megatron.energon.OCRSample.words_boxes>`: The bounding boxes of the text words
      * {py:attr}`words_text: list[str] | None <megatron.energon.OCRSample.words_text>`: The text content of the text words
      * {py:attr}`chars_boxes: torch.Tensor | None <megatron.energon.OCRSample.chars_boxes>`: The bounding boxes of the text characters
      * {py:attr}`chars_text: list[str] | None <megatron.energon.OCRSample.chars_text>`: The text content of the text characters
  * {py:class}`TextSample <megatron.energon.TextSample>`: Represents a sample which only contains a text string (e.g. for text generation)
    * Attributes:
      * {py:attr}`text: str <megatron.energon.TextSample.text>`: The text string
  * {py:class}`VidQASample <megatron.energon.VidQASample>`: Represents a sample which contains a video and a question with answer.
    * Attributes:
      * {py:attr}`video: VideoData <megatron.energon.VidQASample.image>`: The input image tensor
      * {py:attr}`context: str <megatron.energon.VQASample.context>`: The context/question
      * {py:attr}`answers: list[str] | None <megatron.energon.VQASample.answer>`: The answer string
      * {py:attr}`answer_weights: torch.Tensor | None <megatron.energon.VQASample.answer_weights>`: Weights for possibly multiple answers
  * {py:class}`VQASample <megatron.energon.VQASample>`: Represents a sample which contains an image, a question/context and an answer
    * Attributes:
      * {py:attr}`image: torch.Tensor <megatron.energon.VQASample.image>`: The input image tensor
      * {py:attr}`context: str <megatron.energon.VQASample.context>`: The context/question
      * {py:attr}`answers: list[str] | None <megatron.energon.VQASample.answer>`: The answer string
      * {py:attr}`answer_weights: torch.Tensor | None <megatron.energon.VQASample.answer_weights>`: Weights for possibly multiple answers
  * {py:class}`VQAOCRSample <megatron.energon.VQAOCRSample>`: Sample type for question answering related to optical character recognition.
    * Attributes:
      * {py:attr}`image: str <megatron.energon.VQAOCRSample.image>`: The input image
      * {py:attr}`context: str <megatron.energon.VQAOCRSample.text>`: The context/question
      * {py:attr}`text: str <megatron.energon.VQAOCRSample.text>`: The text contained in the image
      * {py:attr}`answers: list[str] | None <megatron.energon.VQAOCRSample.answer>`: The answer string
      * {py:attr}`answer_weights: torch.Tensor | None <megatron.energon.VQAOCRSample.answer_weights>`: Weights for possibly multiple answers
      * {py:attr}`words_boxes: torch.Tensor | None <megatron.energon.VQAOCRSample.words_boxes>`: The bounding boxes of the text words
      * {py:attr}`words_text: list[str] | None <megatron.energon.VQAOCRSample.words_text>`: The text content of the text words


(sample-loading)=
### Sample Loading

There are multiple options for how to convert the data stored in the tar files to an instance of one of the sample types above.

After choosing the sample type, `energon prepare` will ask if you want to use a "simple field map" or a "sample loader".
There is a also a third method called "CrudeWebdataset".

#### Field Map

If your data consists of simple text, json and images that can be decoded by the standard [webdataset auto decoder](https://rom1504.github.io/webdataset/api/webdataset/autodecode.html),
and they map directly to the attributes of your chosen sample type from the list above, use a "field map".
The field map stores which file extension in the webdataset shall be mapped to which attribute of the sample class.

#### Sample Loader

If your data needs some custom decoding code to compute the sample attributes from the data in the tar, you should use a custom sample loader.
The code shall only contain the dataset-specific decoding, no project-specific decoding.

Example for a special format (e.g. ocr dataset) for which we will use a custom `sample_loader.py`:

```text
parts
├── segs-000000.tar
│   ├── 636f6d706f6e656e747362656e6566693030616e6472(0025).jp2
│   ├── 636f6d706f6e656e747362656e6566693030616e6472(0025).lines.png
│   ├── 636f6d706f6e656e747362656e6566693030616e6472(0025).mp
│   ├── 636f6d706f6e656e747362656e6566693030616e6472(0025).words.png
│   ├── 636f6d706f6e656e747362656e6566693030616e6472(0075).jp2
│   ├── 636f6d706f6e656e747362656e6566693030616e6472(0075).lines.png
│   ├── 636f6d706f6e656e747362656e6566693030616e6472(0075).mp
│   └── ...
└── ...
```
`.mp` (`msgpack` content) files are automatically decoded, containing:
```json
{
  "identifier": "componentsbenefi00andr",
  "pageno": 25,
  "size": {"w": 2286, "h": 3179},
  "lines": [
    {"l": 341, "t": 569, "b": 609, "r": 1974, "text": "CHAPTER 4  ADVANCED TRAFFIC CONTROL SYSTEMS IN INDIANA"},
    {"l": 401, "t": 770, "b": 815, "r": 2065, "text": "A variety of traffic control systems currently exist"},
    //...
  ],
  "words": [
    {"l": 341, "t": 577, "b": 609, "r": 544, "text": "CHAPTER"},
    {"l": 583, "t": 578, "b": 607, "r": 604, "text": "4"},
    //...
  ],
  "chars": [
    {"t": 579, "b": 609, "l": 341, "r": 363, "text": "C"},
    {"t": 579, "b": 609, "l": 370, "r": 395, "text": "H"},
    //...
  ],
}
```

`sample_loader.py`:
```python
import torch


def sample_loader(raw: dict) -> dict:
    return dict(
        __key__=raw["__key__"],
        image=raw["jp2"],
        text="\n".join(line["text"] for line in raw["mp"]["lines"]),
        lines_boxes=torch.tensor(
            [
                (line["l"], line["t"], line["r"] - line["l"], line["b"] - line["t"])
                for line in raw["mp"]["lines"]
            ],
            dtype=torch.int64,
        ),
        lines_text=[line["text"] for line in raw["mp"]["lines"]],
        words_boxes=torch.tensor(
            [
                (line["l"], line["t"], line["r"] - line["l"], line["b"] - line["t"])
                for line in raw["mp"]["words"]
            ],
            dtype=torch.int64,
        ),
        words_text=[line["text"] for line in raw["mp"]["words"]],
        chars_boxes=torch.tensor(
            [
                (line["l"], line["t"], line["r"] - line["l"], line["b"] - line["t"])
                for line in raw["mp"]["chars"]
            ],
            dtype=torch.int64,
        ),
        chars_text=[line["text"] for line in raw["mp"]["chars"]],
    )


def part_filter(part: str) -> bool:
    return part in ("jp2", "mp")
```

For more information please also read [](custom-sample-loader).

(wds-format-special)=
### Special format
Sometimes, your data will not be easily represented as a `field_map` explained above. 
For example, your data may contain

* structured data like nested boxes for each sample
* custom binary formats
* xml / html / pickle etc.

In those cases you have two options:

1. Creating a custom `sample_loader.py` in the `.nv-meta` folder as explained above.
    * This will typically do the job and is preferred if you only have to do some small conversions.
2. Using a `CrudeWebdataset`
    * For more intricate conversions, you can use a CrudeWebdataset that will pass your samples in a raw form into your TaskEncoder where you can then convert them based on the subflavor for example. For more details see [](crude-data).

Even for these specific wds formats, you would start preparing your data using the [dataset preparation command](#energon_data_prepare), but you will need to define a custom sample loader or select `CrudeWebdataset` in the dataprep wizard. 

(data-on-disk)=
## Dataset Format on Disk

The energon library supports loading large multi-modal datasets from disk.
To load the dataset, it must comply with the format described in this section.

A valid energon dataset must contain an `.nv-meta` folder with certain files as shown below.

```
my_dataset
├── .nv-meta
│   ├── dataset.yaml
│   ├── split.yaml
│   └── .info.yaml
├── shards
│   ├── shard_000.tar
│   ├── shard_001.tar
│   ├── ...
```

Note that the `shards` folder is just an example. The shards and their folder can be named differently, but the `.nv-meta` structure is always the same.

### Files in `.nv-meta`
#### dataset.yaml 
The `dataset.yaml` contains the dataset definition, i.e. the dataset class to use as loader, optional decoders.
If you want to create such a file, you should consider using the [CLI preparation tool](energon_data_prepare).

Here's an example:
```yaml
sample_type:
  __module__: megatron.energon
  __class__: CaptioningSample
field_map:
  image: jpg
  caption: txt
```

The `__class__` and `__module__` values help the library construct the correct object.
The `field_map` specifies how the fields from each webdataset sample are mapped to the members of the sample dataclass.

In this example, the dataclass is
```python
@dataclass
class CaptioningSample(Sample):
    image: torch.Tensor
    caption: str
```

In some scenarios, you might need a more advanced way to map samples into the dataclass.
In that case, please check out [this page](../advanced/advanced_dataformat).

#### split.yaml
This file contains the splits (i.e. train, val, test), each a list of the shards for each split.
It can also contain an "exclude list" to exclude certain samples or shards from training.
Example:

```yaml
exclude: []
split_parts:
  train:
  - shards/shard_000.tar
  - shards/shard_001.tar
  val:
  - shards/shard_002.tar
  test:
  - shards/shard_003.tar
```

To exclude certain shards or samples, you need to add those to the `exclude` list as follows:

```yaml
exclude:
  - shards/shard_004.tar
  - shards/shard_001.tar/000032
  - shards/shard_001.tar/000032
split_parts:
...
```
The above code excludes the entire shard `004` and two samples from the shard `001`.

#### .info.yaml
The hidden info file is auto-generated and contains statistics about each shard.

Example:
```yaml
shard_counts:
  shards/000.tar: 1223
  shards/001.tar: 1420
  shards/002.tar: 1418
  shards/003.tar: 1358
```
