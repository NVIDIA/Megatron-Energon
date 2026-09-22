<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(linear-data-processing)=
# Linear Dataset Processing

Energon can also run finite, one-pass data-processing jobs. Examples include
building manifests, validating or hashing payloads, extracting metadata,
selectively transcoding media, and materializing a derived dataset. These jobs
usually want each input sample once, without training-time shuffle, batching,
or repetition.

Setting a Task Encoder's `decoder` class attribute to `None` is useful when the
processing code needs the stored payload rather than Energon's automatically
decoded representation:

```python
from megatron.energon import DefaultTaskEncoder


class ProcessingTaskEncoder(DefaultTaskEncoder):
    decoder = None
```

For a WebDataset, ordinary payload parts then remain `bytes`. For example, a
`txt` part is not converted to `str`, a `json` part is not parsed, and a `jpg`
part is not converted to a PIL image or tensor. Energon's own metadata, such as
`__key__`, `__restore_key__`, `__tags__`, and `__sources__`, remains structured.
The same setting also leaves payloads obtained from primary and auxiliary
{py:class}`FileStore <megatron.energon.FileStore>` objects supplied to cookers undecoded.

This is particularly useful when you want to:

- copy, checksum, validate, or index payloads without decoding them;
- preserve compressed image, audio, or video bytes in a derived dataset;
- decode only selected formats or records with application-specific settings;
- use a library or parser other than Energon's automatic decoder; or
- avoid decoding media that the processing result does not consume.

`decoder = None` changes the representation of loaded parts; it does not avoid
reading the tar member, remote object, or selected dataset column. Use a cooker
`part_filter` to avoid loading unused parts. It is also not inherently faster if
your code immediately decodes every payload in the same way.

## Example: Build a JSONL Metadata Manifest

Suppose a prepared WebDataset contains one `txt` caption and one `jpg` image per
sample. The following job reads every sample once, decodes only the caption,
and writes the image byte size and SHA-256 digest to a JSONL file. The JPEG is
never decoded into pixels.

Configure the dataset as crude data so the cooker receives the stored parts
directly:

```yaml
# /data/input/.nv-meta/dataset.yaml
__module__: megatron.energon
__class__: CrudeWebdataset
```

Then define and run the processing pipeline:

```python
import json
from hashlib import sha256
from pathlib import Path

from megatron.energon import (
    Cooker,
    CrudeSample,
    DefaultTaskEncoder,
    Sample,
    WorkerConfig,
    basic_sample_keys,
    edataclass,
    get_loader,
    get_processing_dataset,
    reraise_exception,
)


@edataclass
class ManifestRecord(Sample):
    caption: str
    image_nbytes: int
    image_sha256: str


def keep_manifest_parts(part: str) -> bool:
    return part in {"txt", "jpg"}


def cook_manifest_record(sample: CrudeSample) -> ManifestRecord:
    caption_bytes = sample["txt"]
    image_bytes = sample["jpg"]
    if not isinstance(caption_bytes, bytes) or not isinstance(image_bytes, bytes):
        raise TypeError("decoder=None requires raw txt and jpg payloads")

    return ManifestRecord(
        **basic_sample_keys(sample),
        caption=caption_bytes.decode("utf-8"),
        image_nbytes=len(image_bytes),
        image_sha256=sha256(image_bytes).hexdigest(),
    )


class ManifestTaskEncoder(DefaultTaskEncoder):
    # Keep WebDataset parts and FileStore results as bytes. The cooker decides
    # what, if anything, to decode.
    decoder = None
    cookers = (
        Cooker(cook=cook_manifest_record, part_filter=keep_manifest_parts),
    )


def build_manifest(input_path: Path, output_path: Path) -> None:
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=0,
        global_error_handler=reraise_exception,
    )
    dataset = get_processing_dataset(
        input_path,
        worker_config=worker_config,
        task_encoder=ManifestTaskEncoder(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for record in get_loader(dataset):
            output.write(
                json.dumps(
                    {
                        "key": record.__key__,
                        "caption": record.caption,
                        "image_nbytes": record.image_nbytes,
                        "image_sha256": record.image_sha256,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


build_manifest(Path("/data/input"), Path("/data/output/manifest.jsonl"))
```

{py:func}`get_processing_dataset <megatron.energon.get_processing_dataset>` fixes
the input pipeline to a finite, unshuffled, unpacked, and unbatched pass. It
still applies cookers, `preencode_sample` or `encode_sample`, and
`postencode_sample`. `num_workers=0` runs this work in the main process, which is
easiest to debug. `reraise_exception` stops on an error instead of logging and
skipping the failed sample.

### Separate Outputs for Recipe Leaves

For a recipe, the singular API concatenates all resolved leaves in recipe order.
Use {py:func}`get_processing_datasets
<megatron.energon.get_processing_datasets>` when each recipe leaf should have a
separate output:

```python
from megatron.energon import get_processing_datasets


for dataset, source_factory in get_processing_datasets(
    "/data/input/recipe.yaml",
    split_part="train",
    worker_config=worker_config,
    task_encoder=ManifestTaskEncoder(),
):
    # Open a user-specific writer for source_factory, then consume this leaf.
    for record in get_loader(dataset):
        ...
```

The result contains one `(dataset, source_factory)` pair per resolved leaf
reference. Both processing APIs ignore blend weights and epoch repetitions: each
leaf reference is traversed once. If a recipe references the same physical
dataset twice, those are still two leaf references and are therefore processed
twice.

Energon deliberately leaves the output writer user-specific. Coordinated
restart requires a writer contract that can flush or commit and expose writer
state alongside the state from a {py:func}`get_savable_loader
<megatron.energon.get_savable_loader>` instance. Without transactional or
idempotent writer support, no generic runner can promise exactly-once external
output.

For more throughput, increase `num_workers` so pure cooker and Task Encoder work
runs in worker processes, but keep output writes in the main loop. Worker code
may run ahead because of prefetching or be replayed during restore, so side
effects in `cook_manifest_record` or `encode_sample` are not an exactly-once
output mechanism.

## Ordering, Recipes, and Restartability

The example uses one rank and a direct prepared dataset. With multiple ranks,
Energon partitions the input stream; each rank should write a distinct output
file and the files can be merged afterwards. Processing traversal is exhaustive,
so an ordinary weighted `blend` does not sample according to its weights here.

The finite loader does not make the external JSONL write transactional with
Energon's checkpoint. For restartable processing, make the sink idempotent by
`__key__`, or coordinate sink commits with saved loader state. Filters,
`SkipSample`, or a non-raising error handler can intentionally reduce the output
count.

If downstream processing actually wants strings, parsed JSON, image tensors, or
lazy AV objects, keep a {py:class}`SampleDecoder
<megatron.energon.SampleDecoder>` configured instead of setting it to `None`.
See [](../basic/data_decoding) for the available automatic decoder options.
