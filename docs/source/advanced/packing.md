<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Packing

Packing (sometimes also called sequence packing), enables you to selectively compress multiple
input samples into a single sample, for example depending on their length.

This technique is commonly used with large language models, if the input samples have very different
lengths leading to lots of padding and hence wasted compute.

This section explains how you can pack samples together and utilize the full context length.

## Buffered packing

To use buffered packing, you need to implement the TaskEncoder methods {py:meth}`select_samples_to_pack <megatron.energon.TaskEncoder.select_samples_to_pack>`
and {py:meth}`pack_selected_samples <megatron.energon.TaskEncoder.pack_selected_samples>`.
Furthermore, you need to initialize the loader with the `packing_buffer_size` argument set to a non-zero number.

The `select_samples_to_pack` method will receive a list of samples (size according to the selected `packing_buffer_size`),
and should partition those samples into groups that shall be packed together. Hence the function typically returns
a list of lists of samples. Alternatively it may return {py:class}`PackedSamplesOutput <megatron.energon.PackedSamplesOutput>` with a ``pushback`` sequence: those samples are appended back to the reading buffer before the next fill from the dataset.

For each group, the second method `pack_selected_samples` will be called. You need to implement how a group of
samples will be mapped to a single sample. In terms of LLMs for example, this method might concatenate the input tokens.

## Streaming packing

Buffered packing is useful when the selector needs a window of samples, for example to sort or search
for a good combination. If your algorithm greedily emits one pack at a time and can slice the last
sample, use streaming packing instead. Streaming packing calls
{py:meth}`select_next_pack <megatron.energon.TaskEncoder.select_next_pack>` with an iterator and lets
the task encoder pull only the samples needed for the next pack:

```python
get_train_dataset(
    path,
    batch_size=None,
    packing_buffer_size="stream",
    task_encoder=MyTaskEncoder(...),
)
```

`select_next_pack` returns the same shapes as buffered packing, but it must return at most one
non-empty pack. Return a plain list when there is no carryover:

```python
def select_next_pack(self, samples):
    pack = []
    remaining = self.max_tokens
    for sample in samples:
        sample_len = len(sample.tokens)
        if sample_len <= remaining:
            pack.append(sample)
            remaining -= sample_len
            if remaining == 0:
                return [pack]
            continue

        split = remaining
        pack.append(PartialSample(sample=sample, slice=(0, split)))
        return PackedSamplesOutput(
            packs=[pack],
            pushback=(PartialSample(sample=sample, slice=(split, sample_len)),),
        )
    return [pack] if pack else []
```

As with buffered packing, {py:meth}`postencode_sample <megatron.energon.TaskEncoder.postencode_sample>`
runs on each selected pack member before
{py:meth}`pack_selected_samples <megatron.energon.TaskEncoder.pack_selected_samples>` is called.

`packing_buffer_size` may also be a per-group dict when using Metadataset V2 groups. Values may be
an integer for buffered packing, `"stream"` for streaming packing, or `None` to disable packing for
that group.


```{admonition} Note
:class: important
You can set the `__restore_key__` of the packed sample to an empty tuple, since energon will set the correct
restore key afterwards, based on the samples that went in.
```

```{warning}
To handle attention masks and tokenized inputs, you will want to operate on a different sample type.
The `pack_selected_samples` method may return a different sample type that is expected as the input for the `batch` method.
```

It is important, to mark custom functions like `encode_sample` and `pack_selected_samples` as `@stateless` to allow saving
samples for packing. If augmentations happen, it should be marked with
`@stateless(restore_seeds=True)`, to deterministically set the seeds based on the `TaskEncoder.current_sample_index`.
You have to make sure the methods are actually stateless, meaning that they will produce the same output when invoked
with the same input and random states.

## Carrying over partial samples

Sometimes the next sample only partially fits into the remaining packed context. In that case,
{py:meth}`select_samples_to_pack <megatron.energon.TaskEncoder.select_samples_to_pack>` or
{py:meth}`select_next_pack <megatron.energon.TaskEncoder.select_next_pack>` can return a
{py:class}`PartialSample <megatron.energon.PartialSample>` for the part that fits and push back another
`PartialSample` for the remainder via {py:class}`PackedSamplesOutput <megatron.energon.PackedSamplesOutput>`.

`PartialSample` stores the original sample and a task-defined slice payload:

```python
@edataclass
class PartialSample(Generic[T_sample, T_slice]):
    sample: T_sample
    slice: T_slice
```

The `slice` object is stored as-is in loader state and restore keys, so it must be serializable by
the same mechanism you use for checkpointing loader state. A `tuple[int, int]` using normal Python
`(start, stop)` slicing semantics is a typical choice.

The slicing semantics are user-defined. Energon preserves and restores the `slice` payload, but your
task encoder applies it. If you override
{py:meth}`postencode_sample <megatron.energon.TaskEncoder.postencode_sample>` and produce partials,
`postencode_sample` must accept both full samples and `PartialSample` inputs:

```python
def select_samples_to_pack(
    self,
    samples: list[TokenizedSample | PartialSample[TokenizedSample, tuple[int, int]]],
) -> PackedSamplesOutput[TokenizedSample | PartialSample[TokenizedSample, tuple[int, int]]]:
    sample = samples[0]
    if isinstance(sample, PartialSample):
        base_sample = sample.sample
        token_start, token_stop = sample.slice
    else:
        base_sample = sample
        token_start = 0
        token_stop = len(sample.tokens)

    return PackedSamplesOutput(
        packs=[
            [
                PartialSample(
                    sample=base_sample,
                    slice=(token_start, token_start + self.remaining_context),
                )
            ]
        ],
        pushback=(
            PartialSample(
                sample=base_sample,
                slice=(token_start + self.remaining_context, token_stop),
            ),
        ),
    )


@stateless
def postencode_sample(
    self,
    sample: TokenizedSample | PartialSample[TokenizedSample, tuple[int, int]],
) -> TokenizedSample:
    if isinstance(sample, PartialSample):
        token_start, token_stop = sample.slice
        sample = sample.sample
        return TokenizedSample.derive_from(
            sample,
            tokens=sample.tokens[token_start:token_stop],
        )
    return sample
```

If you do not use `postencode_sample`, then
{py:meth}`pack_selected_samples <megatron.energon.TaskEncoder.pack_selected_samples>` receives the
`PartialSample` values directly and must apply the slice there. In that mode, type the final packer
to accept the same union of full and partial samples returned by `select_samples_to_pack` or
`select_next_pack`.

Example packing for a large language model extending the example from the [](../basic/task_encoder) section:

```python
class PackingCaptioningTaskEncoder(CaptioningTaskEncoder):
    """This class extends the CaptioningTaskEncoder and adds select_samples_to_pack and pack_selected_samples for packing samples
    efficiently on-the-fly.
    Set the `packing_buffer_size` of the get_(train|val)_dataset to an accordingly large number to get a
    properly sized input sample buffer with good diversity.
    """

    @stateless(restore_seeds=True)
    def encode_sample(self, ...):
        # Added `stateless` decorator to allow saving samples for packing. Will set the seed
        # deterministically based on the self.current_sample_index.
        ...

    def select_samples_to_pack(self, samples: List[CaptioningSample]) -> List[List[CaptioningSample]]:
        # Do something intelligent here, e.g. sort by caption length and concat where possible.
        # This could be better, but it's just an example.
        samples.sort(key=lambda x: len(x.caption))
        groups = []
        while len(samples) > 0:
            batch = []
            caption_len = 0
            while len(samples) > 0 and caption_len + len(samples[0].caption) < self.max_length:
                sample = samples.pop(0)
                batch.append(sample)
                caption_len += len(sample.caption)
            groups.append(batch)
        return groups
    
    @stateless
    def pack_selected_samples(self, samples: List[CaptioningSample]) -> CaptioningSample:
        # Construct a new CaptioningSample by concatenating the captions
        ...

```
