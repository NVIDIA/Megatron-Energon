## Epochized Blending

As an alternative to blending with a weight for each dataset, blending can be made accurate and
iterating the dataset can follow epochs (i.e. interrupt iteration after an epoch) with this concept.

Here is a modified example of the above `metadataset.yaml` config file that changes to epochized blending:
```yaml
__module__: megatron.energon
__class__: MetadatasetV2
splits:
  train:
    # Blend the following datasets, repeating coco 5 times, coyo-train 2 times and coyo-val 1 times
    blend_epochized:
      - repetitions: 5
        path: ./coco
        # ... Other parameters
      - repetitions: 2
        path: ./coyo
      - repetitions: 1
        path: ./coyo
        split_part: val
```

Now, the call to `get_train_dataset` requires the additional parameter `repeat=False` to stop interrupt after one epoch.
This would also work without this, but then the shuffle buffer will shuffle samples at bounderies of epochs:

```py
from megatron.energon import get_train_dataset, get_loader, WorkerConfig


loader = get_loader(get_train_dataset(
    'metadataset.yaml',
    batch_size=2,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
    worker_config=WorkerConfig.default_worker_config(),
    repeat=False,
))

# This will now stop iterating after the datasets have been iterated (coco 5 times, coyo-train 2
# times and coyo-val 1 times). Of course, the data is still being shuffled between all those
# datasets.
for batch in loader:
    print(batch)

```