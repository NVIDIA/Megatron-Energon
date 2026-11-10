<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Customized Blending

Recipe `blend` weights target sample counts by default. A task encoder can expose another unit for
weighted blending by registering a named sample size metric:

```py
from megatron.energon import DefaultTaskEncoder, TextSample, sample_size_metric, stateless


class TextTaskEncoder(DefaultTaskEncoder):
    @sample_size_metric("tokens")
    @stateless
    def token_count(self, sample: TextSample) -> int:
        return len(sample.text)
```

The recipe selects that unit with `blend_weight_unit: tokens` (see
{ref}`blend weight units in recipes <blend-weight-unit>`). The reserved unit `samples` keeps
sample-count blending and does not call a sample size metric.

The metric must not return negative values. A dataset whose samples all have size zero never reaches
its share and is selected more and more often, so make sure that every blended dataset yields
samples with a non-zero size.

If {py:meth}`build_packing_groups <megatron.energon.TaskEncoder.build_packing_groups>` returns more
than one group, the output streams of the groups are blended as well. This blend needs a second
metric for the same unit, registered with `stage="encoded"`. It receives the samples the groups
yield: the packed samples returned by
{py:meth}`pack_selected_samples <megatron.energon.TaskEncoder.pack_selected_samples>`, or the
encoded samples of a group without packing:

```py
class PackedTextTaskEncoder(DefaultTaskEncoder):
    @sample_size_metric("tokens")
    @stateless
    def token_count(self, sample: TextSample) -> int:
        return len(sample.text)

    @sample_size_metric("tokens", stage="encoded")
    @stateless
    def packed_token_count(self, sample: PackedTextSample) -> int:
        return sum(len(text) for text in sample.texts)
```

A metric is only required for a blend that is built: the default metric if a group contains more
than one dataset, and the encoded metric if there is more than one group. If a required metric is
missing, loading the dataset raises an error.

In your Task Encoder you could customize the blend of datasets by overriding the `build_train_datasets` method as shown below.


```{warning}
This interface is not stable and may be subject of changes quite often for new features we add. So if you change
how the datasets are plugged together, consider that this may have to be adapted to future changes.
```

```py

class CaptioningTaskEncoder(
    DefaultTaskEncoder[CaptioningSample, CaptioningSample, CaptioningRawBatch, CaptioningBatch]
):
    ...
    
    def build_train_datasets(
        self,
        *,
        datasets: List[Tuple[BaseCoreDatasetFactory[T_sample], float]],
        worker_config: WorkerConfig,
        batch_size: Optional[int],
        batch_drop_last: bool = False,
        packing_buffer_size: Optional[int | Literal["stream"]] = None,
        virtual_epoch_length: int = 0,
        shuffle_buffer_size: Optional[int] = None,
    ) -> SavableDataset[T_batch]:
        # The default implementation uses MixDataset, which mixes the datasets according to their weights
        # This could be customized, e.g. to batch the datasets first (i.e. each batch only contains data from a single datset)
        # and then blend, which would yield the same distribution.
        dataset = BlendDataset(
            *datasets,
            worker_config=worker_config,
        )
        # Apply post-encoding, buffered packing, or streaming packing.
        dataset = self._build_packing_postencode(
            dataset,
            packing_buffer_size=packing_buffer_size,
            worker_config=worker_config,
        )
        # Build batches from blended samples
        dataset = self.build_batch(
            dataset,
            batch_size=batch_size,
            batch_drop_last=batch_drop_last,
            worker_config=worker_config,
        )
        # Optionally epochize
        if virtual_epoch_length > 0:
            dataset = EpochizeDataset(
                dataset,
                length=virtual_epoch_length,
                worker_config=worker_config,
            )
        return dataset

```