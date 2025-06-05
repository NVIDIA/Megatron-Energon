<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(custom-sample-loader)=
# Custom Sample Loader

```{warning}
The custom sample loader is a legacy feature and using [crude datasets](../advanced/crude_datasets.md) with cookers is usually the preferred way.
This feature might be deprecated at some point in the future.
```
Instead of using a `field_map` in your `dataset.yaml`, you can create custom python code for sample loading
right next to your dataset inside the `.nv-meta` folder.
One reason for why we are deprecating this feature, is that you cannot easily version-control the code inside this folder.
In contrast, cookers live inside your code repository together with the task encoder.

Here's an example for your updated `dataset.yaml` if you want to use a sample loader:
```yaml
sample_type:
  __module__: megatron.energon
  __class__: OCRSample
sample_loader: sample_loader.py:sample_loader
part_filter: sample_loader.py:part_filter
```

In addition, you need to create a python file inside the `.nv-meta` folder. In this case it's
called `sample_loader.py`.
That file needs to contain the two methods referenced above:

```python
import torch


def sample_loader(raw: dict) -> dict:
    data = raw["bbox.json"]

    return dict(
        __key__=raw["__key__"],
        image=raw["jpg"],
        text=raw["gt.txt"],
        lines_boxes=torch.tensor([box["bbox"] for box in data], dtype=torch.int64),
        lines_text=[box["text"] for box in data],
    )


def part_filter(part: str) -> bool:
    return part in ("bbox.json", "gt.txt", "jpg")
```

Your `sample_loader` method must accept a dictionary as argument and return a dictionary. It directly operates on 
the webdataset samples and the resulting dictionary keys should map to the corresponding sample class members.
In this case an {py:class}`OCRSample <megatron.energon.OCRSample>`.

With the optional `part_filter` method, you can prevent some webdataset fields from being loaded.
Given a field name, the method should return True if the field is to be kept.


(interleaved-sample-loader)=
## Example: Interleaved Data and Arbitrary Image Count

### The Webdataset Structure

If you need multiple files with an arbitrary number of data per sample, e.g. multiple image / video / audio files, this shows a blueprint for how to setup your webdataset tar files and how to load that webdataset with Energon.

The structure of the shard files could be like this:

`tar -tvf shard_0.tar`:
```python
sample_000001.2345ew.jpg
sample_000001.json
sample_000002.35tags.jpg
sample_000002.as23ds.jpg
sample_000002.gd1dtg.jpg
sample_000002.gds233.jpg
sample_000002.json
sample_000002.sdag42.jpg
sample_000003.json
sample_000004.asf234.jpg
sample_000004.json
```

where the structure of a json file is:

`tar -xf shard_0.tar sample_000001.json -O`:
```json
{
    "images": [null, "2345ew.jpg", null],
    "texts": ["This is some text, an image is following.", null, "More text after the image."],
}
```
Note that the image path corresponds to the filename of the image after the first "." in the sample. This is all part of the extension as defined by webdataset. Everything before the first "." is part of the sample key and must be equal to match into the same group.


### Usage with Energon

To make this work with Energon, in the `energon prepare` [CLI preparation tool](energon_data_prepare), you can either tell the wizard to create a custom sample loader template for you, or change the files accordingly. Here is the example with the structure above:

`.nv-meta/dataset.yaml`:
```yaml
sample_type:
  __module__: megatron.energon
  __class__: InterleavedSample
part_filter: sample_loader.py:part_filter
sample_loader: sample_loader.py:sample_loader
```

`.nv-meta/sample_loader.py`:
```python
import torch


def sample_loader(raw: dict) -> dict:
    # Note that the images are already decoded, as well as the json part.
    return dict(
        __key__=raw["__key__"],
        sequence=[
            (raw[image] if text is None else text)
            for image, text in zip(raw["json"]["images"], raw["json"]["texts"])
        ],
    )


def part_filter(part: str) -> bool:
    # Need to load all parts
    return True
```

When iterating, you'll get those samples as `InterleavedSample` which either contains the image tensor, or the strings for text. The first sample would look like this:

```python
InterleavedSample(
    sequence=["This is some text, an image is following.", torch.Tensor(...) or PIL.Image.Image(), "More text after the image."]
)
``` 
