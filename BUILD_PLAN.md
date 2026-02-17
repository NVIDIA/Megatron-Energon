---
name: Dynamic blend weight schedules
overview: Analyze current blending pipeline and add support for per-batch dynamic blend weights (e.g. piecewise linear/step schedules) that are deterministic, checkpoint/resume-safe, and work with existing SavableDataLoader/WorkerConfig behavior.
todos:
  - id: trace-weight-flow
    content: Audit all call sites that read/assume `LoadedDataset.weight` is float/probability and adjust to `WeightSpec` + runtime normalization where needed.
    status: completed
  - id: add-weight-spec
    content: Introduce typed weight schedule representation + parsing helpers for YAML configs (linear/step with points).
    status: completed
  - id: metadataset-v2-config
    content: Update `BlendWeightMixin`/`LoadedDataset` typing and metadataset blending logic to propagate weight specs (schedule can be outermost), forbid schedule×schedule nesting, and avoid static normalization when schedules are present.
    status: completed
  - id: blend-dataset-runtime
    content: Extend `BlendDataset` to evaluate weights per active batch index, allow zero weights, and normalize dynamically with caching per batch index.
    status: completed
  - id: tests-determinism
    content: Add unit tests for schedule evaluation, zero-weight behavior, and checkpoint/resume determinism (pattern after `tests/test_dataset_det.py::test_determinism_taskencoder_save_restore`).
    status: completed
isProject: false
---

## Findings (current behavior)

- **Where weights come from**: In `MetadatasetV2`, `splits.train.blend[*].weight` is currently a `float` via `BlendWeightMixin`.
  - See `BlendWeightMixin` and `MetadatasetBlend.get_datasets()` in `/home/pfischer/source/megatron-energon/src/megatron/energon/metadataset/metadataset_v2.py`.
- **How weights are applied**: The training pipeline eventually reaches `TaskEncoder.build_train_datasets()` which, for `DatasetBlendMode.DATASET_WEIGHT`, wraps each inner dataset in `RepeatDataset` and then constructs a `BlendDataset(*[(dataset, weight)], worker_config=...)`.
  - See `/home/pfischer/source/megatron-energon/src/megatron/energon/task_encoder/base.py` around the `BlendDataset(...)` call.
- **How iteration counters exist today**: `SavableDatasetWrapper` activates `WorkerConfig` once per *outer yielded item* (your chosen x-axis: **per-batch**), via `worker_config.worker_activate(sample_index, ...)`.
  - Inside the worker, any dataset wrapper (including `BlendDataset`) can read a deterministic per-rank/per-worker “batch index” from `WorkerConfig.active_worker_config.active_worker_batch_index` (this is exactly what `TaskEncoder.current_batch_index` exposes).
  - This means we can implement schedules that change **between batches**, not within a batch.

## Recommended approach (low-invasive, deterministic)

Implement **scheduled weights evaluated from the active batch index** and normalize at sampling time.

### 1) Add a typed “weight spec” that can be constant or scheduled

- **New dataclasses** (recommended location: new module like `/home/pfischer/source/megatron-energon/src/megatron/energon/weights.py`):
  - `WeightSpec`: either `ConstantWeight(value: float)` or `ScheduledWeight(kind: Literal["linear","step"], points: dict[int,float])`, optionally with a `scale: float = 1.0`.
  - `evaluate(batch_idx: int) -> float` returning a non-negative float.
  - **Piecewise linear**: interpolate between nearest bracketing points; clamp to endpoints outside range.
  - **Step**: use last point with key \le batch_idx; clamp before first.

### 2) Update metadataset config types to accept schedules

- Change `BlendWeightMixin.weight` in `/home/pfischer/source/megatron-energon/src/megatron/energon/metadataset/metadataset_v2.py` from `float` to `Union[float, WeightSpecConfig]` (where `WeightSpecConfig` is the YAML-facing structure).
- Update `LoadedDataset.weight` in `/home/pfischer/source/megatron-energon/src/megatron/energon/metadataset/loader_interface.py` to allow `WeightSpec` (instead of only `float|int|None`).
- **Forbid nested schedules, but allow schedule at any level (outermost typical)**:
  - Allow `schedule × constant` (represented as a scaled schedule).
  - Reject `schedule × schedule` anywhere in the metadataset hierarchy with a clear config error.
  - This supports “outer scheduled, inner constant” as well as “inner scheduled, outer constant”, but not both scheduled on the same path.
- **Where to enforce `schedule × schedule**`:
  - Enforce during **weight composition across metadataset levels**, i.e. in `get_datasets()` when combining an outer blend entry’s weight with each inner `LoadedDataset.weight` (conceptually: `combined = inner_weight * entry_weight`).
  - Rationale: this is the one place where weights from multiple levels are multiplied; YAML parsing alone can’t reliably detect “stacked schedules” once nested metadatasets/join loaders are involved.
  - Practical implementation: a shared helper (e.g. `compose_weights(a, b)`) used by:
    - `MetadatasetBlend.get_datasets()` in `metadataset_v2.py`
    - `MetadatasetBlender.get_datasets()` in `metadataset.py` (V1), if we want schedules there too / for consistency
    - any join loader that multiplies weights similarly
  - Behavior: if both operands are scheduled → raise `ValueError` with a message pointing to the offending dataset path(s); else return a `WeightSpec` (possibly scaled) or a float.

### 3) Make `BlendDataset` accept dynamic weights and compute probs per batch

- In `/home/pfischer/source/megatron-energon/src/megatron/energon/wrappers/blend_dataset.py`:
  - Accept `dataset_weights: Tuple[SavableDataset, WeightSpecLike]`.
  - Allow weights to be **0** (drop the current `assert weight > 0`).
  - Compute a probability vector by:
    - reading current batch index via `WorkerConfig.active_worker_config.active_worker_batch_index` (fallback to 0 if not active),
    - evaluating each dataset’s current weight,
    - zeroing weights for datasets that are empty on that worker or exhausted,
    - normalizing (if sum>0), then sampling via `WorkerRng.choice_idx`.
  - Cache the computed tensor keyed by the last seen batch index to avoid per-sample recomputation within the same batch.

### 4) Adjust metadataset normalization semantics

- Today `MetadatasetBlend.get_datasets()` normalizes weights by `sum_weight` at config time.
- With schedules, **sum changes over time**, so normalization must move to runtime (inside `BlendDataset`).
- Proposed change: return **raw (unnormalized) weights** from metadataset blend and let `BlendDataset` normalize at sampling time.
  - Pros: supports schedules naturally; preserves “relative weights” interpretation.
  - Cons: any external code that assumed `LoadedDataset.weight` sums to 1 would need updating (we’ll locate those call sites and fix if any).

### 5) Tests / validation (determinism & semantics)

- Add focused unit tests (wherever existing tests live) to cover:
  - constant weights unchanged behavior
  - step schedule (e.g. {0:100, 100:10, 1000:0}) including weight==0
  - linear interpolation correctness
  - checkpoint/resume determinism: same restored state yields same dataset-choice sequence, patterned after `tests/test_dataset_det.py::test_determinism_taskencoder_save_restore`

## Alternatives considered (and why not first)

- **Rebuild dataloader each step / epoch**: exact but heavy; breaks worker persistence and complicates resume.
- **Main-process scheduler controlling workers**: would require IPC or shared state; complexity/perf risk.
- **Piecewise-constant phases via `ConcatDataset` + `LimitDataset**`: simple and precise in sample counts, but only approximates linear schedules (unless many phases) and balloons config/state.

## Notes on your stated semantics

- Because schedule is evaluated from the **active per-batch index** in `WorkerConfig`, the weight function is stable while a batch is being assembled (good) and advances between batches.
- Across ranks, as long as each rank yields the same number of batches (usual DP assumption), the schedules stay in sync without extra communication.

