<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Packing (Slicing and Batching)

If you need to pack samples efficiently, e.g. for language models where padding can be reduced
easily, this section is for you.

## Concept of efficient Slicing during Batching

The task encoder offer the method {py:meth}`slice_batch <megatron.energon.TaskEncoder.slice_batch>`,
which gets the samples of a batch and returns a new list of multiple batches. Internally,
never more than {py:func}`batch_size <megatron.energon.get_train_dataset>` samples are kept, so
after each yielded batch-slice, new samples will be fetched from the inner dataset to avoid lag.

It is important, to mark custom functions like `encode_sample` as `@stateless` to allow saving
samples for slicing. If augmentations happen, it should be marked with
`@stateless(restore_seeds=True)`, to deterministically set the seeds based on the
`TaskEncoder.current_sample_index`.

Example for padding for a large language model extending the example from the [Task Encoders](task_encoders) section:

```python
class PackingCaptioningTaskEncoder(CaptioningTaskEncoder):
    """This class extends the CaptioningTaskEncoder and adds slice_batch for recomputing the batch
    slice efficiently on-the-fly.
    Set the batch_size of the get_(train|val)_dataset to an accordingly large number to get a
    properly sized input samples buffer with some choice.
    """

    @stateless(restore_seeds=True)
    def encode_sample(self, ...):
        # Added `stateless` decorator to allow saving samples for slicing. Will set the seed
        # deterministically based on the self.current_sample_index.
        ...

    def slice_batch(self, samples: List[CaptioningSample]) -> List[List[CaptioningSample]]:
        # Do something intelligent here, e.g. sort by caption length and concat where possible.
        # This could be better, but it's just an example.
        samples.sort(key=lambda x: len(x.caption))
        batches = []
        while len(samples) > 0:
            batch = []
            caption_len = 0
            while len(samples) > 0 and caption_len + len(samples[0].caption) < self.max_length:
                sample = samples.pop(0)
                batch.append(sample)
                caption_len += len(sample.caption)
            batches.append(batch)
        return batches
    
    def batch(self, samples: List[CaptioningSample]) -> CaptioningRawBatch:
        # Adapt the tokenizer accordingly to concat the samples from the slicing.
        ...

```
