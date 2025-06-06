<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Task Encoder

The "Task Encoder" is an Energon-specific concept. It is a class that describes how each sample is going to be processed at different stages of the pipeline.
Please also take a look at [](basics_flow) to see the pipeline.

If you don't specify any task encoder, a default version, the {py:class}`DefaultTaskEncoder <megatron.energon.DefaultTaskEncoder>` will be used.

For writing your own task encoder, create a class based on {py:class}`DefaultTaskEncoder <megatron.energon.DefaultTaskEncoder>`
and override one or more of the following methods. The data flow of {py:func}`get_train_dataset <megatron.energon.get_train_dataset>` or {py:func}`get_val_dataset <megatron.energon.get_val_dataset>` is as follows:

- {py:meth}`def cook_crude_sample(self, sample: Union[T_sample, CrudeSample]) -> T_sample <megatron.energon.TaskEncoder.cook_crude_sample>`
  - Optional. Define when using [crude data](crude-data).
- {py:func}`def encode_sample(self, sample: T_sample) -> T_encoded_sample <megatron.energon.DefaultTaskEncoder.encode_sample>`
  - Transform the raw data from the dataset (e.g. augment/transform images, tokenize a single sample).
  - Alternatively, you can use {py:meth}`preencode_sample <megatron.energon.TaskEncoder.preencode_sample>` and {py:meth}`postencode_sample <megatron.energon.TaskEncoder.postencode_sample>`. This is especially useful when you use packing and you want to defer loading to when the packed samples are known.
- {py:meth}`def preencode_sample(self, sample: T_sample) -> T_sample <megatron.energon.TaskEncoder.preencode_sample>`
  - Can be used as a replacement to {py:func}`encode_sample <megatron.energon.DefaultTaskEncoder.encode_sample>`
  - Only used together with {py:meth}`postencode_sample <megatron.energon.TaskEncoder.postencode_sample>` (below). Use it if you need to instantiate lazy data before packing the selected samples. But it will also work when not using packing.
- {py:meth}`def select_samples_to_pack(self, samples: List[T_encoded_sample]) -> List[List[T_encoded_sample]] <megatron.energon.TaskEncoder.select_samples_to_pack>`
  - Optional. Allows for efficient sample packing. See [](../advanced/packing).
- {py:meth}`def postencode_sample(self, sample: T_sample) -> T_encoded_sample <megatron.energon.TaskEncoder.postencode_sample>`
  - Only used together with {py:meth}`preencode_sample <megatron.energon.TaskEncoder.preencode_sample>`. Use it if you need to instantiate lazy data before packing the selected samples. But it will also work when not using packing.
- {py:meth}`def pack_selected_samples(self, samples: List[T_encoded_sample]) -> T_batch_sample] <megatron.energon.TaskEncoder.pack_selected_samples>`
  - Required if select_samples_to_pack is used. Compresses a group of samples to a single sample.
- (samples are collected for a batch)
- {py:meth}`def batch(self, batch: List[T_encoded_sample]) -> T_raw_batch <megatron.energon.DefaultTaskEncoder.batch>`
  - Collate the batch to a single sample, defaults to padded batching for tensors, lists for everything else.
- {py:meth}`def encode_batch(self, batch_data: T_raw_batch) -> T_batch <megatron.energon.DefaultTaskEncoder.encode_batch>`
  -  Transform the batched data (e.g. tokenize the whole batch).
- (optionally limit the dataset size, based on the `limit` argument)
- (optionally epochize the dataset)
- (move data from the worker to the main process through the {py:class}`torch.data.DataLoader` via {py:func}`get_loader <megatron.energon.get_loader>`)
- For batches based on {py:class}`Batch <megatron.energon.Batch>`, call {py:meth}`def pin_memory(self, batch: T_batch) -> T_batch <megatron.energon.Batch.pin_memory>`, or if not a dataclass, use default torch pinning (this must happen in the main process, thus after data loading)

If a sample or batch is to be ignored, any of these methods may raise {py:class}`IgnoreSample <megatron.energon.IgnoreSample>` to skip the sample being processed.

The types `T_sample`, `T_encoded_sample`, `T_raw_batch` and `T_batch` are generics and depend on your task. You do not necessarily have to specify them, it's only used for proper typing in your IDE.

```python
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from megatron.energon import Batch, CaptioningSample, DefaultTaskEncoder, batch_list, batch_stack


# Type for intermediate batch, after batching operation
@dataclass
class CaptioningRawBatch(Batch):
    # (n, c, h, w)
    image: torch.Tensor
    # (n,)
    caption: List[str]


# Typing for the resulting batch data
@dataclass
class CaptioningBatch(Batch):
    # (n, c, h, w)
    images: torch.Tensor
    # (n, c)
    text_tokens: torch.Tensor
    # (n, c, c)
    text_attn_mask: torch.Tensor


# All the typing is optional
class CaptioningTaskEncoder(
    DefaultTaskEncoder[CaptioningSample, CaptioningSample, CaptioningRawBatch, CaptioningBatch]
):
    """A simple task encoder for captioning."""

    decoder = SampleDecoder(image_decode="torchrgb")

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        max_length: int = 128,
    ):
        # Specify the batch_type for default batching (batching is performed here "manually" by overwriting the `batch`
        # method)
        super().__init__(batch_type=CaptioningRawBatch)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

    def encode_sample(self, sample: CaptioningSample) -> CaptioningSample:
        sample.image = self.image_transform(sample.image)
        return sample
    
    def batch(self, samples: List[CaptioningSample]) -> CaptioningRawBatch:
        # Batch the samples
        # By default, `batch_pad_stack` is used for all tensor fields, and `batch_list` is used for all non-tensor 
        # fields. This example matches the default implementation (not overwriting the `batch` method).
        return CaptioningRawBatch.from_samples(samples)

    def encode_batch(self, batch_data: CaptioningRawBatch) -> CaptioningBatch:
        # Run the encoder on the batch of captions.
        tokenized = self.tokenizer(batch_data.caption)
        # Return the final batch, going into the network
        return CaptioningBatch.derive_from(
            batch_data,
            images=batch_data.image,
            text_tokens=tokenized["input_ids"],
            text_attn_mask=tokenized["attention_mask"],
        )

```

If you're wondering about the `decoder` assignment, check out [](../basic/data_decoding).

Usage in your training script:
```python
    
from torchvision import transforms
from transformers import AutoTokenizer
from megatron.energon import get_loader, get_train_dataset

    
train_img_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
)

train_loader = get_loader(get_train_dataset(
    '/my/dataset/path',
    batch_size=32,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
    task_encoder=CaptioningTaskEncoder(
        tokenizer=AutoTokenizer.from_pretrained('gpt2'),
        image_transform=train_img_transform,
    ),
))

for data in train_loader:
    # data is a CaptioningBatch
    pass

```
