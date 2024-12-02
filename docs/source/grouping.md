<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Grouping

While [packing](packing) joins multiple samples into one sample, grouping allows for rule-based batching of samples into
one batch on the fly.

That means, while packing is done in addition to batching (it's done before), grouping replaces standard batching.

## Example use cases

* Select samples to batch based on image resolution, so that only samples of the same size are in one batch
* Select blended samples based on their dataset origin, so that one batch does not mix different tasks or data types

## How to group

To use grouping, you need to define the method `batch_group_criterion` in your custom task encoder.
This method gets a sample and returns a hashable value that will be used to cluster/group the samples
and it also returns the batch size for that group.
Samples with the same batch group criterion will be batched together. Once enough samples for one group
have been collected (reached the batch size for that group), they will be batched and pushed down the pipeline
to the next processing step.

Here's and example task encoder that batches samples based on the file name in their sample key.

```python
class GroupingTaskEncoder(
    TaskEncoder[CaptioningSample, CaptioningSample, CaptioningSample, CaptioningSample]
):
    @stateless
    def encode_sample(self, sample: CaptioningSample) -> CaptioningSample:
        sample.caption = sample.__key__.split("/")[-2]
        return sample

    def batch_group_criterion(self, sample: CaptioningSample) -> Tuple[Hashable, int]:
        if sample.caption == "data-0.tar":
            return "shard1", 4
        elif sample.caption == "data-1.tar":
            return "shard2", 8
        else:
            assert False
```
