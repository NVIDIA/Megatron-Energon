# Joining Datasets / Adding Columns to Metadataset

This document explains how to use the feature to add columns in a Metadataset using the `megatron.energon` module.

Adding columns is analogous to webdataset's add column, i.e., merging two datasets which require the same files within the tar files.

## Minimal Example

Below is a minimal example of a `metadataset_minimal.yaml` file:

```yaml
__module__: megatron.energon
__class__: MetadatasetV2
splits:
  train:
    join:
      - path: ds1
      - path: ds2
    joiner:
      __module__: megatron.energon
      __class__: CaptioningSample
```
(note that the mixing is omitted here, as it is optional; the joining is of course also optional)

## Extensive Example

Here is a more extensive example of a `metadataset_extended.yaml` file:

```yaml
__module__: megatron.energon
__class__: MetadatasetV2
splits:
  train:
    mix:
      - weight: 1
        join:
          - path: ds1
            dataset_config: dataset.yaml  # If override is needed
          - path: ds2
            dataset_config: dataset.yaml
            subflavor: ds1  # If needed, overrides the 'ds2' subflavor
            subflavors: # If needed, will be merged(overriding) with parent subflavor
              ds2_extra: 2
            split_config: split.yaml  # Sets this for all joined datasets
        joiner:
          __module__: my_module
          __class__: JoinedSample # Type should implement from_joined(ds1, ds2)
        split_config: split.yaml  # Sets this for all joined datasets
        split_part: train  # Sets this for all joined datasets
        subflavor: ds1  # Sets this for all joined datasets
        subflavors:  # Sets this for all joined datasets (it will be merged with their individual subflavors)
          source: metadataset.yaml
```

## Custom Join Type

To define a custom join type, you can create a Python class as shown below in `my_module.py`:

```python
from dataclasses import dataclass
import torch
from megatron.energon import Sample, TextSample

@dataclass
class JoinedSample(Sample):
    text1: torch.Tensor
    text2: torch.Tensor

    @staticmethod
    def from_joined(ds1: TextSample, ds2: TextSample) -> "JoinedSample":
        return JoinedSample.derive_from(
            ds1,
            text1=ds1.text,
            text2=ds2.text,
        )
```

This class should implement the `from_joined` method to combine samples from `ds1` and `ds2`.

## Example Structure

Here is an example structure of the datasets within the tar files:

```
ds1
├── .nv-meta
│   ├── .info.yaml
│   ├── split.yaml
│   └── dataset.yaml
├── shard1.tar
│   ├── 0001.txt
│   ├── 0002.txt
│   └── 0003.txt
├── shard1.idx

ds2
├── .nv-meta
│   ├── .info.yaml
│   ├── split.yaml
│   └── dataset.yaml
├── shard1.tar
│   ├── 0001.txt
│   ├── 0002.txt
│   └── 0003.txt
├── shard1.idx
```

In this example, `ds1/shard1.tar` and `ds1/shard1.tar` contain files with the same names. When adding columns, the files from both datasets are joined based on their keys, which must be in the same order. Each dataset must be prepared (i.e. .nv-meta created).

By following these examples, you can configure and extend your Metadataset to include additional columns and custom join types.
