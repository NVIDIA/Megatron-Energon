<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(packing-blending)=

# Packing and Blending Internals

Packing and blending both consume ahead of the visible output stream and both own random or buffered
state. They are therefore sensitive to ordering and checkpoint changes.

## Pipeline position

For an ordinary training branch, the relevant order is:

```text
leaf -> cook -> blend -> shuffle -> encode/pre-encode -> pack -> post-encode -> batch
```

Moving blend or pack across encoding changes what constitutes one emitted item.

## Blend selection

`BlendDataset` is designed for repeated, effectively infinite children. It stores child exhaustion and
`WorkerRng`.

The configured weights directly define target selection proportions. The child is sampled with the saved
worker RNG.

Changing the probability formula or RNG consumption changes iteration order.

## Buffered packing

`PackingDataset` owns a buffer from which it selects packs. The same state principle applies: if input has
been consumed but not emitted, the buffer is checkpoint state. Save and restore must preserve sample
payloads, restore keys, ordering, and any selector state.

When changing packing logic, checkpoint at these points:

- an empty buffer;
- a partially filled buffer;
- immediately after a selection;
- just before child exhaustion;
- after reset.

## Review checklist

For a change to blending or packing, verify:

1. the exact stage at which the selector sees a sample;
2. whether the stage consumes random numbers and which `WorkerRng` owns them;
3. whether input can be consumed without immediate output;
4. whether all pending samples retain restore keys and provenance;
5. whether empty packs or multiple packs per call are rejected;
6. whether child exhaustion changes selection probabilities correctly;
7. whether exact restore reproduces the uninterrupted stream;
8. whether the change is iteration-order or checkpoint breaking.

Public configuration and examples are covered in [Packing](../advanced/packing.md),
[Grouping](../advanced/grouping.md), and [Custom Blending](../advanced/custom_blending.md).
