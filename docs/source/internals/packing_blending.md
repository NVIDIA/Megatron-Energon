<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(packing-blending)=

# Packing and Blending Internals

Packing and blending both consume ahead of the visible output stream and both own random or buffered
state. They are therefore sensitive to ordering, skip mode, and checkpoint changes.

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
the weighting metric observes and what constitutes one emitted item.

## Blend selection

`BlendDataset` is designed for repeated, effectively infinite children. It stores child exhaustion,
`WorkerRng`, and the emitted size per child.

With the sample-count metric, configured weights directly define target selection proportions. With a
custom size metric, the selector compensates for the amount already emitted. In simplified form:

```text
target_i  = normalized configured weight_i
deficit_i = target_i * total_emitted - emitted_i
credit_i  = max(deficit_i, 0)
score_i   = target_i * (epsilon + credit_i) ** alpha
```

The scores are normalized to probabilities and sampled with the saved worker RNG. The size metric must
return a nonnegative value.

The metric is evaluated on the loaded or cooked sample at the blend point, before `encode_sample`,
pre-encoding, padding, or packing. If token count is the desired metric, that count must already be exposed
by the cooker or source sample metadata. A token count produced only inside `encode_sample` is not visible
to the blend metric.

Changing the metric, its evaluation point, the probability formula, or RNG consumption changes iteration
order.

## Streaming packing

`StreamingPackingDataset` delegates pack formation to a selector that pulls from an input iterator. For
each call, the selector returns either:

- a list containing at most one non-empty pack; or
- `PackedSamplesOutput`, which can additionally push a sample back for the next pack.

Pushback is stored in a `SavablePartialSampleBuffer`. Its sample payload and restore key must survive a
checkpoint because the input stream has already advanced past it.

The dataset also saves separate `SampleIndex` scopes for selection, per-sample encoding, and final packing.
These scopes keep random seeding and restore keys aligned even when stages are skipped.

In skip mode:

- selection still executes;
- per-sample encoding can be elided only when both the sample encoder and the final packer are skip-safe;
- the final packer can be elided independently when it is skip-safe;
- the corresponding sample indexes still advance.

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
- after reset and after logical-worker skip mode.

## Grouped packing

A packing-group selector routes encoded samples into group-specific branches. Each branch can have its own
packing behavior. Since the final blend sees already packed outputs, its weights apply to packed items or
to the metric reported for those items, not directly to original leaf samples.

A selector is state-defining and cannot be marked skip-safe. If selection uses sample fields created by an
encoding hook, keep that hook before selection and include any relevant random state in the normal
savability mechanism.

## Review checklist

For a change to blending or packing, verify:

1. the exact stage at which the metric or selector sees a sample;
2. whether the stage consumes random numbers and which `WorkerRng` owns them;
3. whether input can be consumed without immediate output;
4. whether all pending samples retain restore keys and provenance;
5. whether empty packs, multiple packs per call, or negative metric values are rejected;
6. whether child exhaustion changes selection probabilities correctly;
7. whether skip mode advances all indexes without omitting stateful work;
8. whether exact restore reproduces the uninterrupted stream;
9. whether recipe migration should retain or reset wrapper-local buffers;
10. whether the change is iteration-order or checkpoint breaking.

Public configuration and examples are covered in [Packing](../advanced/packing.md),
[Grouping](../advanced/grouping.md), and [Custom Blending](../advanced/custom_blending.md).
