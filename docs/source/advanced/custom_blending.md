# Advanced Customized Blending

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
        packing_buffer_size: Optional[int] = None,
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