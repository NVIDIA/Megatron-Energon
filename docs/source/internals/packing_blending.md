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

Packing groups create a separate branch per group:

```text
blend -> shuffle -> encode -> select group -> post-encode -> pack
```

The already packed group streams are blended afterward. Moving blend or pack across encoding changes what
constitutes one emitted item.

## Blend selection

`BlendDataset` is designed for repeated, effectively infinite children. It stores child exhaustion and
`WorkerRng`.

The configured weights directly define target selection proportions. The child is sampled with the saved
worker RNG.

Changing the probability formula or RNG consumption changes iteration order.

## Streaming packing

`StreamingPackingDataset` delegates pack formation to a selector that pulls from an input iterator. For
each call, the selector returns either:

- a list containing at most one non-empty pack; or
- `PackedSamplesOutput`, which can additionally push a sample back for the next pack.

Pushback is stored in a `SavablePartialSampleBuffer`. Its sample payload and restore key must survive a
checkpoint because the input stream has already advanced past it.

The dataset also saves separate `SampleIndex` scopes for selection, per-sample encoding, and final packing.
These scopes keep random seeding and restore keys aligned.

## Buffered packing

`PackingDataset` owns a buffer from which it selects packs. The same state principle applies: if input has
been consumed but not emitted, the buffer is checkpoint state. Save and restore must preserve sample
payloads, restore keys, ordering, and any selector state.

When changing packing logic, checkpoint at these points:

- an empty buffer;
- a partially filled buffer;
- immediately after a selection;
- with a pushed-back sample;
- just before child exhaustion;
- after reset.

## Grouped packing

A packing-group selector routes encoded samples into group-specific branches. Each branch can have its own
packing behavior. Since the final blend sees already packed outputs, its weights apply to packed items,
not directly to original leaf samples.

A selector is state-defining. If selection uses sample fields created by an
encoding hook, keep that hook before selection and include any relevant random state in the normal
savability mechanism.

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
