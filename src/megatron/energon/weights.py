# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Weight specifications for dataset blending.

This module provides a small abstraction (`WeightSpec`) for blend weights that may be constant or
scheduled over training. Scheduled weights are evaluated as a pure function of an integer
`batch_idx`, which in Energon is typically the per-rank batch index
(`WorkerConfig.active_worker_batch_index`). Keeping evaluation stateless ensures determinism and
checkpoint/resume safety when used inside savable datasets.

The repository-wide strict YAML parser is optimized for mappings with string keys. Therefore,
schedule point keys may arrive as strings and are converted to integers during parsing.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence, TypeAlias, Union, cast

WeightScheduleKind: TypeAlias = Literal["linear", "step"]
WeightSchedulePoints: TypeAlias = Mapping[int, float]

# YAML-facing schedule config:
#   weight:
#     step:
#       0: 100
#       5: 0
# or
#   weight:
#     linear:
#       0: 100
#       100: 10
#
# NOTE: the project-wide strict YAML->dataclass parser (`JsonParser`) only has special handling for
# `dict[...]` with `str` keys. Therefore, we intentionally type schedule configs as `dict[str, dict[str, float]]`
# even though YAML may use integer keys for the points. We convert keys to int during parsing.
WeightConfig: TypeAlias = Union[float, int, dict[str, dict[str, float]]]


class WeightSpec:
    """A non-negative weight function for blending.

    Implementations must be deterministic and side-effect free. In the dataset pipeline, the
    `batch_idx` argument is expected to be a stable counter (typically per-rank batch index).
    """

    is_scheduled: bool = True

    def evaluate(self, batch_idx: int) -> float:  # pragma: no cover (interface)
        """Evaluate the weight at a given batch index (must return a non-negative finite float)."""
        raise NotImplementedError

    def scale_by(self, factor: float) -> "WeightSpec":
        """Return a new weight scaled by a non-negative constant factor."""
        if not math.isfinite(factor):
            raise ValueError(f"Invalid weight scale factor: {factor!r}")
        if factor < 0:
            raise ValueError(f"Weight scale factor must be >= 0, got {factor}")
        return ScaledWeight(inner=self, scale=factor)


WeightLike: TypeAlias = Union[float, WeightSpec]


def _as_float_weight(value: float | int) -> float:
    """Validate and coerce a numeric weight to a non-negative finite float."""
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"Invalid weight: {value!r}")
    if f < 0:
        raise ValueError(f"Weight must be >= 0, got {value!r}")
    return f


def eval_weight(weight: WeightLike, batch_idx: int) -> float:
    """Evaluate a `WeightLike` (float or `WeightSpec`) to a validated float at `batch_idx`."""
    if isinstance(weight, WeightSpec):
        v = weight.evaluate(batch_idx)
    else:
        v = float(weight)
    if not math.isfinite(v):
        raise ValueError(f"Invalid evaluated weight at batch_idx={batch_idx}: {v!r}")
    if v < 0:
        raise ValueError(f"Evaluated weight must be >= 0 at batch_idx={batch_idx}, got {v!r}")
    return v


def is_scheduled_weight(weight: WeightLike) -> bool:
    """Return True if the weight is a scheduled (time-varying) `WeightSpec`."""
    return isinstance(weight, WeightSpec) and weight.is_scheduled


@dataclass(frozen=True, slots=True)
class ConstantWeight(WeightSpec):
    """A constant (time-invariant) weight."""

    value: float
    is_scheduled: bool = False

    def __post_init__(self) -> None:
        _as_float_weight(self.value)

    def evaluate(self, batch_idx: int) -> float:
        return self.value

    def scale_by(self, factor: float) -> WeightSpec:
        # Keep constants as constants where possible
        return ConstantWeight(_as_float_weight(self.value * _as_float_weight(factor)))


@dataclass(frozen=True, slots=True)
class ScaledWeight(WeightSpec):
    """A weight scaled by a constant non-negative factor."""

    inner: WeightSpec
    scale: float

    def __post_init__(self) -> None:
        _as_float_weight(self.scale)

    def evaluate(self, batch_idx: int) -> float:
        return eval_weight(self.inner, batch_idx) * self.scale


@dataclass(frozen=True, slots=True)
class ScheduledWeight(WeightSpec):
    """A piecewise scheduled weight.

    Args:
        kind: Either `\"step\"` or `\"linear\"`.
        points: Sorted `(batch_idx, weight)` control points. Keys must be unique and >= 0.
        scale: Optional non-negative multiplier applied to the evaluated value.

    Semantics:
        - `step`: value at the last point with key <= batch_idx (knot points inclusive).
        - `linear`: linear interpolation between points; clamped outside endpoints.
    """

    kind: WeightScheduleKind
    points: tuple[tuple[int, float], ...]
    scale: float = 1.0
    is_scheduled: bool = True

    def __post_init__(self) -> None:
        _as_float_weight(self.scale)
        if len(self.points) == 0:
            raise ValueError("Scheduled weight must have at least one point")
        xs = [x for x, _ in self.points]
        if xs != sorted(xs):
            raise ValueError(f"Schedule points must be sorted by key, got keys={xs!r}")
        if len(set(xs)) != len(xs):
            raise ValueError(f"Schedule points must have unique keys, got keys={xs!r}")
        for x, y in self.points:
            if x < 0:
                raise ValueError(f"Schedule point key must be >= 0, got {x}")
            _as_float_weight(y)

    @staticmethod
    def from_points(
        kind: WeightScheduleKind, points: Mapping[Any, Any], *, scale: float = 1.0
    ) -> "ScheduledWeight":
        """Create from a mapping of point_key -> weight (keys are coerced to int)."""
        items = sorted((int(k), float(v)) for k, v in points.items())
        return ScheduledWeight(kind=kind, points=tuple(items), scale=float(scale))

    def evaluate(self, batch_idx: int) -> float:
        if batch_idx < 0:
            raise ValueError(f"batch_idx must be >= 0, got {batch_idx}")

        xs = [x for x, _ in self.points]
        ys = [y for _, y in self.points]

        if batch_idx <= xs[0]:
            return ys[0] * self.scale
        if batch_idx >= xs[-1]:
            return ys[-1] * self.scale

        if self.kind == "step":
            # Step schedule semantics: take the last point with key <= batch_idx.
            # This makes knot points inclusive, e.g. {100: 10} applies at batch_idx=100.
            idx = bisect.bisect_right(xs, batch_idx) - 1
            return ys[idx] * self.scale

        # linear interpolation: xs[lo] < batch_idx <= xs[hi]
        hi = bisect.bisect_left(xs, batch_idx)
        lo = hi - 1

        x0, y0 = xs[lo], ys[lo]
        x1, y1 = xs[hi], ys[hi]

        # linear interpolation
        assert self.kind == "linear"
        if x1 == x0:
            return y1 * self.scale
        t = (batch_idx - x0) / (x1 - x0)
        return (y0 + t * (y1 - y0)) * self.scale


class NormalizedWeightGroup:
    """Node-level normalizer for a set of sibling blend weights.

    Given raw sibling weights (constants and/or schedules), this computes normalized weights that
    sum to 1 for a given batch index. Results are cached per batch index.
    """

    def __init__(self, raw_weights: Sequence[WeightLike]) -> None:
        if len(raw_weights) == 0:
            raise ValueError("Cannot normalize empty weight group")
        self._raw_weights = tuple(raw_weights)
        self._cached_batch_idx: int | None = None
        self._cached_norm: tuple[float, ...] | None = None

    @property
    def raw_weights(self) -> tuple[WeightLike, ...]:
        return self._raw_weights

    def normalized(self, batch_idx: int) -> tuple[float, ...]:
        """Return normalized sibling weights for the given batch index (sums to 1)."""
        if self._cached_batch_idx == batch_idx and self._cached_norm is not None:
            return self._cached_norm

        vals = tuple(eval_weight(w, batch_idx) for w in self._raw_weights)
        denom = float(sum(vals))
        if denom <= 0:
            raise RuntimeError(
                f"Blend weight schedule evaluated to all zeros at batch_idx={batch_idx}."
            )
        norm = tuple(v / denom for v in vals)
        self._cached_batch_idx = batch_idx
        self._cached_norm = norm
        return norm


@dataclass(frozen=True, slots=True)
class NormalizedWeight(WeightSpec):
    """A scheduled weight that represents one entry of a normalized sibling weight group."""

    group: NormalizedWeightGroup
    index: int
    is_scheduled: bool = True

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be >= 0, got {self.index}")

    def evaluate(self, batch_idx: int) -> float:
        norm = self.group.normalized(batch_idx)
        if self.index >= len(norm):
            raise IndexError(f"index {self.index} out of bounds for norm size {len(norm)}")
        return norm[self.index]


def weight_to_config(weight: WeightLike) -> Any:
    """Convert a WeightLike to a JSON-serializable object for logging/config dumps."""
    if isinstance(weight, WeightSpec):
        if isinstance(weight, ConstantWeight):
            return float(weight.value)
        if isinstance(weight, ScheduledWeight):
            return {
                "kind": weight.kind,
                "points": {int(k): float(v) for k, v in weight.points},
                "scale": float(weight.scale),
            }
        if isinstance(weight, ScaledWeight):
            return {"scale": float(weight.scale), "inner": weight_to_config(weight.inner)}
        if isinstance(weight, NormalizedWeight):
            return {
                "normalized_index": int(weight.index),
                "raw_weights": [weight_to_config(w) for w in weight.group.raw_weights],
            }
        return {"type": type(weight).__name__, "repr": repr(weight)}
    return float(weight)


def weight_from_config(weight: WeightConfig | WeightLike | None) -> WeightLike:
    """Convert a YAML-facing weight config to a WeightLike.

    - numeric -> float
    - {\"linear\"|\"step\": {<point_key>: float, ...}} -> ScheduledWeight (point_key coerced to int)
    - WeightSpec -> returned as-is
    - None -> 1.0 (default)
    """
    if weight is None:
        return 1.0
    if isinstance(weight, WeightSpec):
        return weight
    if isinstance(weight, (int, float)):
        return _as_float_weight(weight)
    if isinstance(weight, Mapping):
        # Expect exactly one schedule kind key
        if len(weight) != 1:
            raise ValueError(
                f"Invalid weight schedule config (expected single key), got keys={list(weight.keys())!r}"
            )
        kind_any, points_any = next(iter(weight.items()))
        kind = cast(WeightScheduleKind, kind_any)
        if kind not in ("linear", "step"):
            raise ValueError(f"Unknown schedule kind {kind_any!r}; expected 'linear' or 'step'")
        if not isinstance(points_any, Mapping):
            raise ValueError(
                f"Invalid schedule points for kind {kind!r}: expected mapping, got {type(points_any)}"
            )
        return ScheduledWeight.from_points(kind, cast(Mapping[Any, Any], points_any))
    raise ValueError(f"Unsupported weight config type: {type(weight)}")


def compose_weights(inner: WeightLike, outer: WeightLike) -> WeightLike:
    """Compose two weights via multiplication, forbidding schedule×schedule.

    This is meant for metadataset hierarchy composition: combined = inner * outer.
    """
    inner = weight_from_config(inner)
    outer = weight_from_config(outer)

    inner_sched = is_scheduled_weight(inner)
    outer_sched = is_scheduled_weight(outer)
    if inner_sched and outer_sched:
        raise ValueError(
            "Nested scheduled blend weights are not supported (schedule×schedule). "
            "Please keep at most one scheduled blend node along any dataset path."
        )

    # numeric × numeric
    if not isinstance(inner, WeightSpec) and not isinstance(outer, WeightSpec):
        return _as_float_weight(float(inner) * float(outer))

    # spec × numeric or numeric × spec
    if isinstance(inner, WeightSpec) and not isinstance(outer, WeightSpec):
        return inner.scale_by(_as_float_weight(outer))
    if not isinstance(inner, WeightSpec) and isinstance(outer, WeightSpec):
        return outer.scale_by(_as_float_weight(inner))

    # unreachable due to schedule×schedule check, but keep for safety
    assert False, "Unhandled weight composition case"


def make_node_entry_weights(
    entry_weights: Sequence[WeightConfig | WeightLike | None],
) -> list[WeightLike]:
    """Turn a list of raw entry weights into *node-normalized* weights.

    This preserves hierarchical blending semantics by normalizing at each blend node.

    - If any entry is scheduled, returns `NormalizedWeight` specs (shared group) so they sum to 1
      per `batch_idx`.
    - Otherwise returns constant normalized floats summing to 1.
    """
    raw = [weight_from_config(w) for w in entry_weights]
    if any(is_scheduled_weight(w) for w in raw):
        group = NormalizedWeightGroup(raw_weights=raw)
        return [NormalizedWeight(group=group, index=i) for i in range(len(raw))]

    # all constant -> normalize once
    raw_f = [_as_float_weight(w) if isinstance(w, (int, float)) else eval_weight(w, 0) for w in raw]  # type: ignore[arg-type]
    denom = float(sum(raw_f))
    if denom <= 0:
        raise ValueError("Blend weights must sum to > 0")
    return [_as_float_weight(v / denom) for v in raw_f]
