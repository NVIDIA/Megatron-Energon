<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(data-preparation)=
# Data Preparation

The aim of data preparation is to convert your data to a format that the energon loader can understand and iterate.
Energon's primary data format is [WebDataset](https://github.com/webdataset/webdataset) with some extra information stored in a folder called `.nv-meta`.
Below in [](data-on-disk) we explain the details about this format.
We also support a simpler JSONL format which will always be interpreted as [crude data](crude-data).

## Important Considerations

Depending on what your data looks like and how you are planning to use it, you will have to make a few choices,
**before you prepare your dataset**:

**Monolithic Dataset vs. Polylithic (primary and auxiliary) Datasets**

You can include the media (images/video/audio) inside the same webdataset along with the text-based data of each sample (such as labels, captions, etc.).
Or you can keep the media separate (either in another indexed webdataset or as individual files on disk).
When using JSONL, the media will always be separate, so JSONL datasets are always polylithic unless they are text-only.

The monolithic option is faster to load. However, there are a few reasons why the other option may be preferable:

* You need to keep the original media files and you don't want to duplicate them in the tar files
* Your media data is very large (e.g. long videos) and you need to keep your primary dataset small (containing just the text-based data and meta information)
* You want to re-use the same media with different labels or you want to train on different subsets
* You want to train with [online packing](../advanced/packing.md) and can't fit all the media of the packing buffer in memory. With polylithic datasets you can use caching to avoid that issue.

**How to shard the data**

When using a WebDataset, it will be split into a bunch of shards (i.e. tar files). You'll have to decide how many samples to put in one shard and how many shards to get overall.

To maximize the loading speed, use as few shards as possible. Even a single shard can work well!
However, if you cannot handle files above a certain size you may need to split the shards more.
A good rule of thumb is to keep your **number of shards below 10k**.

If you are using remote filesystems like S3, there may be an opposing constraint: S3 [limits the number of requests per second](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html)
that you can make for a single prefix (e.g. filename). By using more shards, you can increase the overall rate. Ideally, you would still want to stay below 10k shards.


**Raw vs. baked data**

When using images for example, you could put either the encoded JPG, the decoded pixel values or even the encoded features into the dataset.

Typically, we recommend to go with the "original form" (e.g. JPG) and do all the processing on the fly inside the [cooker](crude-data) and [task encoder](../basic/task_encoder).
This way, you can change the processing and keep your dataset.

However, if the processing becomes a bottleneck, you can move some of it into the dataset creation phase by baking the information in.

Keep in mind that others may also want to use your dataset for a different project.


(monolithic-dataset)=
## Steps to Create a Monolithic Dataset

These are the typical steps to get your data ready:

1. Create a normal [WebDataset](https://github.com/webdataset/webdataset) from your data (including all the media content)
2. Run our preparation tool [`energon prepare`](energon-prepare) to create the additional metadata needed by energon. See [](data-on-disk).

(polylithic-dataset)=
## Steps to Create a Polylithic Dataset

1. Create the primary [WebDataset](https://github.com/webdataset/webdataset) or JSONL file from your text-based part of the data (meta information, labels etc.)
    * Include the file names (don't use absolute paths) of the media that belongs to each sample (e.g. as strings inside a json entry)
2. Create the auxiliary dataset(s). Can be multiple datasets, e.g. one per modality.
    * Either as a folder on disk with all the media files inside
    * Or as another WebDataset that contains just the media files (with the exact same names)
3. Run our preparation tool `energon prepare` **on both datasets** (yes also on the JSONL) to convert to an energon-compatible format
    * Configure both datasets as `CrudeWebdataset` (JSONL always is by default)
    * For the auxiliary datasets, we recommend to enable the [media metadata feature](media-metadata) to store additional information about the media (like image size, resolution, video duration etc.)
4. Create a [metadataset](../basic/metadataset) that specifies what auxiliary data to load for each primary dataset
    * For more details read about [crude data](crude-data)

(create-jsonl-dataset)=
## Steps to Create a JSONL Dataset

A JSONL dataset is a simplified alternative to a full-blown WebDataset with tar files.
It has fewer features, but can easily be read using a standard editor.

```{admonition} Good to know
:class: tip
A JSONL dataset cannot contain media files, but it can reference media files elsewhere (auxiliary data).
It does not have a train/val/test split.
It cannot be used as an auxiliary dataset by other primary datasets.
It cannot be mounted using `energon mount`.
```

A single JSONL file will contain all of your text-based data, one JSON entry per line. For example:

```
{"id": 0, "question": "What is 1+2?", "answer": "3"}
{"id": 1, "question": "Who is Jensen Huang?", "answer": "The CEO of NVIDIA."}
```

And it is essentially equivalent to using a WebDataset with files

```
0.json
1.json
```

each file containing the JSON from one of the lines above.

None of the JSON fields is mandatory. The data is considered to be crude data and will be interpreted by your custom [cooker](crude-data).
If you want to include media, you should include file names of the media files in the JSON.
A metadataset with [auxiliary data](aux-data) can then be used to load the media on the fly.

Here's an example of how a polylithic JSONL dataset with images might look like:

```
{"image": "computer_01.jpg", "caption": "A desktop computer with two monitors."}
{"image": "mountains_123.jpg", "caption": "A beautiful landscape with mountains on a sunny day."}
```

Steps needed:

1. Create the JSONL file according to your needs
2. Run `energon prepare /path/to/my_dataset.jsonl` to create an index next to it
3. Optionally create a [metadataset](../basic/metadataset) that specifies what auxiliary data to load for each primary dataset
    * For more details read about [crude data](crude-data)

The metadataset would then refer to the JSONL dataset while specifying the auxiliary data source:

```yaml
__module__: megatron.energon
__class__: MetadatasetV2
splits:
  train:
    path: /path/to/my_dataset.jsonl
    aux:
      foo_bar_source: ./aux_ds123
      image_source: filesystem://./relative_image_folder
```

An auxiliary data source can be a local or remote folder, or other energon-prepared webdatasets. Even multiple auxiliary sources can be used.
For all the options and to see how to specify a matching cooker, please check out the section on [auxiliary data](aux-data).

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
This shows a monolithic dataset, for polylithic you would drop the JPGs in the primary dataset.

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
* Ask you about the sample type (optionally crude)
* Ask you how to decode the data if not using crude data (field map or sample_loader.py)
* Store all this information in a subfolder `.nv-meta/`, see details [below](data-on-disk).

(media-metadata)=
### Media Metadata

If you are preparing a dataset with media files, energon can retrieve and store additional information about the media (like image size, resolution, video duration etc.).
This information will be stored inside an SQLite database file next to the dataset.
Later, inside the [cooker](crude-data), you can access this information using the {py:meth}`get_media_metadata <megatron.energon.FileStore.get_media_metadata>` method of the {py:class}`FileStore <megatron.energon.FileStore>`.

#### During normal initial preparation of a WebDataset

```sh
> energon prepare --media-metadata --media-by-extension /path/to/dataset
```

#### Adding media metadata to an existing dataset
```sh
> energon prepare-media --media-by-extension /path/to/dataset
```

```{admonition} Good to know
:class: tip
That also works for filesystem datasets. I.e. you can run `energon prepare-media` on a normal folder with media files and it will create the media metadata database file next to the dataset.
```

#### Customizing the selection of media files

You can customize the selection of media files by using the `--media-by-glob`, `--media-by-header` and `--media-by-extension` options.
You must specify exactly one of the options.

To select media files by our default extension list (reocmmended), you can use the `--media-by-extension` option.
```sh
> energon prepare --media-metadata --media-by-extension /path/to/dataset
```

The list can be found in the [extractor.py](https://github.com/NVIDIA/Megatron-Energon/blob/develop/src/megatron/energon/media/extractor.py) file.

To select all media files with the extensions `.jpg`, `.png` and `.webp`, you can use the following command:
```sh
> energon prepare --media-metadata --media-by-glob '*.jpg,*.png,*.webp' /path/to/dataset
```

To select media files by reading their contents/header, you can use the `--media-by-header` option.
```sh
> energon prepare --media-metadata --media-by-header /path/to/dataset
```
Note that this option may be slower than the other options, as it needs to read the contents of the files.


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

Note that the pattern matching syntax uses regexes, so for arbitrary characters insert `.*` not just `*`

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

```{admonition} Good to know
:class: tip
You can inspect your prepared dataset like a normal file system by using the [`energon mount`](energon-mount) feature.
```

### Sample Types

After the split is set up, the assistant will ask you which sample type you want to use.
We provide a set of common sample types such as for image captioning or visual question answering, they are listed below.

This will be sufficient in a simple scenario and if none of these fits, you may even create your own new sample type.
Here are your options:

* You have a new type sample which is rather common but not in our list below
  * Please add your type to energon and create a pull request so we can add it
* Your sample type is experimental very special or used temporarily only
  * You can add the sample type class in your code repository and create the `dataset.yaml` manually, referring to your class with `__class__`
  * You can add the sample type class in your code repository, use a crude dataset and cookers (no need to put the sample type in `dataset.yaml`)


(sect-sample-types)=
#### Available Sample Types

These are the possible integrated types you can currently choose from:

* {py:class}`Sample <megatron.energon.Sample>`: Base dataclass for samples from source webdatasets.
  * Attributes:
    * {py:attr}`__key__: str <megatron.energon.Sample.__key__>`: Unique identifier of the sample within the dataset. Useful for backtracking the source of a single sample.
    * {py:attr}`__key__: str <megatron.energon.Sample.__restore_key__>`: Structured key of the sample, which can be used to regenerate the sample without storing the whole sample.
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

When you actually use and load your dataset, the data stored in the tar files needs to be converted to an instance of your chosen sample type.
There are three options:

1. The conversion is a simple 1:1 mapping of files to fields of the sample type class
    * You can use a simple field map
2. Otherwise the now preferred way is to use a CrudeWebdataset and do the conversion inside a [cooker](crude-data).
3. There is another (now legacy) way, i.e. to create a custom `sample_loader.py` file next to your dataset.
    * This option will continue to work, but we encourage to move to crude datasets in the future.

When running `energon prepare`, you can choose "Crude sample" as the sample type and the assistant will end.
If you picked another sample type, the assistant will ask if you want to use a "simple field map" or a "sample loader".

#### Simple Field Map

If your data consists of simple text, json and images that can be decoded by the standard [webdataset auto decoder](https://rom1504.github.io/webdataset/api/webdataset/autodecode.html),
and they map directly to the attributes of your chosen sample type from the list above, use a "field map".
The field map stores which file extension in the webdataset shall be mapped to which attribute of the sample class.

#### Sample Loader (Deprecated)

If your data needs some custom decoding code to compute the sample attributes from the data in the tar, you can use a custom sample loader.
However, starting from Energon 7, we recommend to use crude datasets and a [cooker](crude-data) instead.

If you use a `sample_loader.py`, its code shall only contain the dataset-specific decoding, no project-specific decoding.

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


(data-on-disk)=
## Dataset Format on Disk (WebDataset)

The energon library supports loading large multi-modal datasets from disk.
To load the dataset, it must comply with the format described in this section unless it's a JSONL dataset.

A valid energon dataset must contain an `.nv-meta` folder with certain files as shown below.

```
my_dataset
├── .nv-meta
│   ├── dataset.yaml
│   ├── split.yaml
│   ├── .info.json
│   ├── index.sqlite
│   └── index.uuid
├── shards
│   ├── shard_000.tar
│   ├── shard_001.tar
│   ├── ...
```

Note that the `shards` folder is just an example. The shards and their folder can be named differently, but the `.nv-meta` structure is always the same.


### Files in `.nv-meta`
#### dataset.yaml (user editable)
The `dataset.yaml` contains the dataset definition, i.e. the dataset class to use as loader, optional decoders.
If you want to create such a file, you should consider using the [CLI preparation tool](energon-prepare).

Here's an example:
```yaml
sample_type:
  __module__: megatron.energon
  __class__: CaptioningSample
field_map:
  image: jpg
  caption: txt
```

For a crude dataset the `dataset.yaml` will simply be
```yaml
__module__: megatron.energon
__class__: CrudeWebdataset
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

#### split.yaml (user editable)
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

#### .info.json (read-only)
The hidden info file is auto-generated and contains a list of all shards and the number of samples in each.

Example:
```json
{
  "energon_version": "7.1.0",
  "shard_counts": {
    "shards/000.tar": 1223,
    "shards/001.tar": 1420,
    "shards/002.tar": 1418,
    "shards/003.tar": 1358
  }
}
```

The order of tar files is important, as it's used by the sqlite database below.

#### index.sqlite and index.uuid (read-only)

The sqlite database was introduced in Energon 7 and allows for fully random access of samples and files by their names.
This is a precondition for [polylithic datasets](aux-data) and for the [`energon mount`](energon-mount) command.
Later, media metadata was added to the database to allow for fast access to the media metadata (audio length, image resolution etc.) of the media files.

Below there is some detailed information for the interested reader. Note that the internal table structure can
change in any release without notice.

The database contains an entry for each sample and sample part including their byte offsets and sizes in the tar files.

Example `samples` table:
| tar_file_id | sample_key | sample_index | byte_offset | byte_size |
| --- | --- | --- | --- | --- |
| 0 | 00000 | 0 | 0 | 35840 |
| 0 | 00001 | 1 | 35840 | 35840 |
| 0 | 00002 | 2 | 71680 | 35840 |
| 0 | ... | | | |

The byte offsets describe the range around all the tar entries that are part of that sample including the tar headers.

Corresponding example `sample_parts` table:

| tar_file_id | sample_index | part_name | content_byte_offset | content_byte_size |
| --- | --- | --- | --- | --- |
| 0 | 0 | json | 1536 | 31 |
| 0 | 0 | png | 3584 | 30168 |
| 0 | 0 | txt | 35328 | 16 |
| 0 | 1 | json | 37376 | 31 |
| 0 | 1 | png | 39424 | 30168 |
| 0 | 1 | txt | 71168 | 16 |
| 0 | ... | | | |

The byte offsets in the `sample_parts` table refer to the byte ranges of the actual file content and can be used to
directly access the content without parsing the tar header.

Both tables can be joined over the `tar_file_id` and the `sample_index`. Note that the `tar_file_id` refers to the list
of tar files in the `.info.json` file.

Newer versions of Energon also add `media_filters` and `media_metadata` tables to the database:

The filters table is used to store the media filters that were used to select the media files.
The media metadata will be stored *for the union* of all the media files that were selected by the filters.

| filter_id | strategy | patterns | created_at_utc |
| --- | --- | --- | --- |
| 1 | EXTENSION |  | 2025-01-01 12:00:00 |
| 2 | GLOB | \*.jpg,\*.png,\*.webp | 2025-01-01 12:00:00 |
| 3 | HEADER |  | 2025-01-01 12:00:00 |
| 4 | ... | ... | ... |


The `media_metadata` table is used to store the media metadata for the selected media files:

| entry_key | metadata_type | metadata_json |
| --- | --- | --- |
| 00000.jpg | image | {"width": 1024, "height": 768, "format": "jpg"} |
| 00001.wav | av | {"audio_duration": 39.0 , "audio_channels": 1, "audio_sample_rate": 16000} |
| 00002.mp4 | ... | ... |


(data-on-disk-jsonl)=
## Dataset Format on Disk for JSONL Datasets

For the simpler JSONL option, you will still need to run `energon prepare`, but this will not create a full `.nv-meta` folder.
Instead, only an index file with the same base filename will be created.

So if your dataset is named `my_dataset.jsonl`, a new file `my_dataset.jsonl.idx` will appear next to it when preparing it.

That's all. The dataset type will always be `CrudeWebdataset` and the split part is `train` by default. However, when loading the dataset
you can change the split type to `val` or `test`.

(data-on-disk-filesystem)=
## Dataset Format on Disk for Filesystem Datasets

Filesystem datasets are datasets that are stored on disk as individual files in a folder.
They are not indexed and cannot be accessed randomly. They are only used as auxiliary datasets.

They can be used without an `.nv-meta` folder, but if you run `energon prepare-media` on them, an sqlite database file will be created inside an `.nv-meta` folder.
The database will contain just the `media_filters` and `media_metadata` tables as explained above.
