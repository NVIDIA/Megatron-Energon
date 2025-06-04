<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->


(crude-data)=
# Crude Datasets and Auxiliary Data

As explained in [](sample-loading), the user has several options to choose how energon converts the raw (crude) data inside the tar files into Python sample objects (instances of a `Sample` dataclass) during loading.

When using crude datasets, this conversion happens through so-called "cookers", i.e. user-defined functions defined in the task encoder, as explained below.
In this case, the dataset on disk will specify neither the resulting sample type nor a sample loader for conversion, hence we call it "crude".
All of the conversion will happen in the user's code base.

## Setting Up a Crude Dataset with Cookers
Let's try it. 
When you run `energon prepare` to prepare your dataset, you can pick "Crude sample" as the sample type.
If you already have an existing energon-compliant data set, you can modify it as follows (or create a copy of your `dataset.yaml` specifically for this use case and keep the original).

Original `dataset.yaml`:
```yaml
sample_type:
  __module__: megatron.energon
  __class__: OCRSample
sample_loader: sample_loader.py:sample_loader
part_filter: sample_loader.py:part_filter
```

New `dataset.yaml` (or `crude.yaml`):
```yaml
__module__: megatron.energon
__class__: CrudeWebdataset
subflavors:
  example_crude_selector: foobar
```

We recommend that you assign [subflavors](sect-subflavors) to differentiate the data.
This helps, if you decide to mix different crude datasets or crude with non-crude data using a [Metadataset](../basic/metadataset).
You can then use the subflavors to assign how each sample shall be processed.
If you like, you can specify any subflavors in the metadataset later instead.

Let's see how we set up our [Task Encoder](../basic/task_encoder) to cook the crude samples and turn them into real samples like {py:class}`TextSample <megatron.energon.TextSample>`.

A crude sample is essentially a dictionary ({py:class}`CrudeSample <megatron.energon.CrudeSample>` inherits from `dict`) and the loader will look for a {py:class}`Cooker <megatron.energon.task_encoder.Cooker>` that can handle and convert your crude samples.
You can provide cookers as follows:

```python
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys

# ...

def cook_text(sample: dict) -> TextSample:
    return TextSample(
        **basic_sample_keys(sample),
        text=f">{sample['txt'].decode()}<",
    )


class MyTaskEncoder(DefaultTaskEncoder[TextSample, TextSample, TextRawBatch, TextBatch]):
    cookers = [
        Cooker(cook_text, has_subflavors={"example_crude_selector": "foobar"}),
        Cooker(...)  # other cookers for other crude data if needed
    ]

    # ...
```

In the example above, the cooker acts on all crude samples that have a subflavor `example_crude_selector` set to `foobar`.
If you leave out the `has_subflavors` argument, the cooker will apply to any sample.

The cooker will convert the dictionary to a {py:class}`TextSample <megatron.energon.TextSample>` by decoding the raw bytes and decorating the text with some nice angle brackets.
Probably you noticed the {py:meth}`basic_sample_keys <megatron.energon.task_encoder.cooking.basic_sample_keys>` helper that we inserted.
All it does, is to forward the key, restore key and flavors from the dict to the real sample. You will always need to forward these, or your dataset will not be restorable.

In a real use-case you will want to do a lot more here and we recommend keeping the cook methods in separate files and importing them where you define your TaskEncoder.

(aux-data)=
## Auxiliary Data for Polylithic Datasets

Using a crude dataset allows you to benefit from two other features of energon:

* Auxiliary Data
* Cache Pools

Both of which are often used in combination. A typical use case is online packing.

An **auxiliary data source** is an additional data source that supports random access and can be used to load data on-demand using its filename.
It is typically used with polylithic datasets where you have one primary dataset that contains only the text-based sample data
and one or more additional auxiliary data sources that contain the (larger) media data such as images or videos.

An auxiliary data source can be either

* A folder on the local or a remote file system
* Another energon-prepared WebDataset

You can specify it in your [metadataset](../basic/metadataset) yaml as follows (look at the `aux:` section)
```yaml
__module__: megatron.energon
__class__: MetadatasetV2
splits:
  train:
    path: ./my_pimary_ds
    aux:
      foo_bar_source: ./aux_ds123
      fs_source: filesystem://./images
      remote_source: msc://mybucket/path/ds
    subflavors:
      crude_type: my_dual_aux_example
```

The format is like
```yaml
aux:
    NAME: PATH_OR_URL
    NAME: PATH_OR_URL
    ...
```

You can specify multiple aux sources each of which can be one of

* Relative or absolute path to a local prepared energon dataset
* Relative or absolute path to a local folder (use the prefix `filesystem://`)
* Path to a remote prepared energon dataset (use prefix `msc://`)
* *[Planned future feature]*: Path to a remote folder (use prefix `filesystem+msc://`)

In your code, the cooker will automatically receive a {py:class}`FileStore <megatron.energon.FileStore>` reference to the data source as a keyword argument:

```python
from megatron.energon import FileStore

# ...

def cook_text(sample: dict, foo_bar_source: FileStore) -> TextSample:
    additional_text = foo_bar_source.get(sample['add_txt_fname'])
    return TextSample(
        **basic_sample_keys(sample),
        text=f"{sample['txt'].decode()} + {additional_text.decode()}",
    )

# ...
```

You can use multiple sources, but you don't have to specify an arg for each source that was defined in the metadataset.

For easier debugging, you should always keep track of all the sources you used. The `get` method takes care of this if you pass it the sample like this:

```python
additional_text = foo_bar_source.get(sample['add_txt_fname'], sample)
```

This will update the sample-internal `__sources__` list with the aux dataset you used.

If you want, you can even use your primary dataset as an auxiliary dataset and look up files by name, yes! If you want to do that, you specify it in the cooker decorator and add an arg:

```python
from megatron.energon import cooker, FileStore

# ...

@cooker(need_primary=True)
def cook_text(sample: dict, primary: FileStore, foo_bar_source: FileStore) -> TextSample:
    # ...
```

You can then retrieve files by their names from the primary dataset.

(cache-pools)=
## Cache Pools

Cache pools allow the user to defer the data transfer if the content will be needed at some point in the future but not immediately.
This is only needed if the media data is rather large and does not reside on a local disk, but rather on a network file system (e.g. lustre) or a remote file system (e.g. object storage).

For example, when using [online packing](../advanced/packing), we may need to keep a buffer of several thousand samples to optimize for the best packing,
but we cannot keep several thousand images in memory, also we don't need the actual image content to optimize the packing.
Hence we will use auxiliary datasets as explained above.

However, at the time of filling the buffer, **we already know** that we **will need the image content in the future**, so cache pools can be used to **prefetch it in the background**.

Initially we want to load some information about the sample and its image but not the actual image pixels.
Later, when the packing is computed, we need to retrieve the pixel values.

In practice, this means the cooker will use a cache pool to queue the data retrieval from an auxiliary data source and obtain a lazy object (a handle to this future data). In a later stage (like {py:meth}`pack_selected_samples <megatron.energon.TaskEncoder.pack_selected_samples>`), the lazy object can be used to retrieve the content.
Ideally, in the mean-time the cache pool has already downloaded the data to a local SSD.

### Using a Cache Pool

When calling {py:func}`get_savable_loader <megatron.energon.get_savable_loader>`,
we pass a cache pool as an additional argument:

```python
from megatron.energon import FileStoreCachePool

# ...

loader = get_savable_loader(
    my_ds,
    ...,
    cache_pool=FileStoreCachePool(
        parent_cache_dir="/local_scratch/cache",
        num_workers=1,
    ),
)
```

Then we tell the cooker decorator that we need access to the cache pool and use it to get a lazy object:

```python
from megatron.energon import cooker, FileStore, CachePool

# ...

@edataclass
class LazyTextSample(Sample):
    text: str
    secondary_text: Lazy[str]


@cooker(need_cache=True)
def cook_text(sample: dict, foo_bar_source: FileStore, cache: CachePool) -> LazyTextSample:
    # Previous non-cached version:
    # additional_text = foo_bar_source.get(sample['add_txt_fname'])

    # Cached version:
    additional_text = cache.get_lazy(foo_bar_source, sample['add_txt_fname'])

    return TextSample(
        **basic_sample_keys(sample),
        text=f"{sample['txt'].decode()} + {additional_text.decode()}",
        secondary_text=additional_text,  # Pass the lazy object on
    )
```

Later down the data processing pipeline, we can retrieve the data, for example here:

```python
@stateless
def pack_selected_samples(self, samples: List[LazyTextSample]) -> TextSample:
    # Get the real object now:
    secondary_txt = samples[0].secondary_txt.get(samples[0])
    
    return TextSample.derive_from(
        samples[0],
        text=samples[0].txt + "|" + secondary_txt,
    )
```