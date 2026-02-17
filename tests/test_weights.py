# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

from megatron.energon.weights import (
    ConstantWeight,
    ScheduledWeight,
    compose_weights,
    eval_weight,
    make_node_entry_weights,
    weight_from_config,
)


class TestWeights(unittest.TestCase):
    def test_weight_from_config_constant(self):
        assert weight_from_config(3) == 3.0
        assert weight_from_config(0.5) == 0.5
        assert eval_weight(ConstantWeight(2.5), 0) == 2.5

    def test_scheduled_weight_step(self):
        w = ScheduledWeight.from_points("step", {0: 100.0, 100: 10.0, 1000: 0.0})
        assert w.evaluate(0) == 100.0
        assert w.evaluate(1) == 100.0
        assert w.evaluate(99) == 100.0
        assert w.evaluate(100) == 10.0
        assert w.evaluate(999) == 10.0
        assert w.evaluate(1000) == 0.0
        assert w.evaluate(5000) == 0.0

    def test_scheduled_weight_linear(self):
        w = ScheduledWeight.from_points("linear", {0: 100.0, 100: 0.0})
        assert w.evaluate(0) == 100.0
        assert w.evaluate(50) == 50.0
        assert w.evaluate(100) == 0.0
        assert w.evaluate(1000) == 0.0

    def test_weight_from_config_schedule_yaml_shape(self):
        # YAML parser provides dict[str, dict[str, float]]; keys are strings.
        w = weight_from_config({"step": {"0": 2.0, "5": 0.0}})
        assert isinstance(w, ScheduledWeight)
        assert w.evaluate(0) == 2.0
        assert w.evaluate(4) == 2.0
        assert w.evaluate(5) == 0.0

    def test_compose_weights_forbids_schedule_schedule(self):
        w1 = ScheduledWeight.from_points("step", {0: 1.0})
        w2 = ScheduledWeight.from_points("linear", {0: 1.0})
        with self.assertRaises(ValueError):
            compose_weights(w1, w2)

    def test_compose_weights_allows_schedule_constant(self):
        sched = ScheduledWeight.from_points("step", {0: 10.0, 5: 0.0})
        composed = compose_weights(1.0, sched)  # constant × schedule
        assert eval_weight(composed, 0) == 10.0
        assert eval_weight(composed, 5) == 0.0

        composed2 = compose_weights(sched, 0.5)  # schedule × constant
        assert eval_weight(composed2, 0) == 5.0
        assert eval_weight(composed2, 5) == 0.0

    def test_make_node_entry_weights_constant_normalized(self):
        w = make_node_entry_weights([1.0, 3.0])
        assert w == [0.25, 0.75]

    def test_make_node_entry_weights_scheduled_normalized(self):
        # For scheduled weights, returned entries are normalized per batch idx to sum to 1.
        w = make_node_entry_weights(
            [
                {"step": {"0": 100.0, "5": 0.0}},
                1.0,
            ]
        )
        v0 = [eval_weight(x, 0) for x in w]
        assert abs(sum(v0) - 1.0) < 1e-6
        assert v0[0] > v0[1]

        v5 = [eval_weight(x, 5) for x in w]
        assert abs(sum(v5) - 1.0) < 1e-6
        # at batch_idx=5 the first schedule evaluates to 0 -> second dominates
        assert v5[0] == 0.0
        assert v5[1] == 1.0


if __name__ == "__main__":
    unittest.main()

