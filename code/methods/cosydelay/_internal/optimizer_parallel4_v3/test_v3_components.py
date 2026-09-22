"""Unit tests for the isolated v3 components."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from methods.cosydelay._internal.optimizer_jacobian.test_jacobian_fitter import (
    _two_lane_payload,
)
from methods.cosydelay._internal.optimizer_parallel3.parallel_fitter import (
    _generate_payloads as _generate_official_payloads,
)

from .fitter import (
    _mse_and_selective_log_gradient,
    _r9_probe_restart,
    _to_search_space,
    _wide_bound_mask,
    generate_mixed_restart_payloads,
)
from .physics_audit import audit_fitted_physics, audit_search_physics
from .run_training_only_search import _jsonable
from .training_selection import (
    balanced_regression_folds,
    score_predictions,
    select_search_candidates,
)


class MixedRestartTest(unittest.TestCase):
    def test_r9_probe_distinguishes_weak_and_severe_coefficients(self):
        payload = {
            "expression": "Cycle_Time * a1 * flow_lane * log(1 + a2 / GR_phase)",
            "block_lanes": ["A_T"],
            "coefficient_names": ["a1", "a2"],
        }
        weak, weak_delays = _r9_probe_restart(
            np.asarray([1.0, 1e-4]), payload
        )
        severe, severe_delays = _r9_probe_restart(
            np.asarray([1000.0, 1e-4]), payload
        )
        self.assertFalse(weak)
        self.assertTrue(severe)
        self.assertLess(weak_delays["A_T"], 10_000.0)
        self.assertGreater(severe_delays["A_T"], 10_000.0)

    def test_selective_log_chain_rule_matches_finite_difference(self):
        _, _, payload = _two_lane_payload()
        log_mask = _wide_bound_mask(payload["bounds"])
        payload["log_parameter_mask"] = log_mask
        raw = np.asarray(payload["initials"][0], dtype=float)
        search = _to_search_space(raw, log_mask)
        _, analytic = _mse_and_selective_log_gradient(search, payload)
        finite = np.zeros_like(search)
        for index, value in enumerate(search):
            step = 1e-6 * max(1.0, abs(float(value)))
            left = search.copy()
            right = search.copy()
            left[index] -= step
            right[index] += step
            left_value, _ = _mse_and_selective_log_gradient(left, payload)
            right_value, _ = _mse_and_selective_log_gradient(right, payload)
            finite[index] = (right_value - left_value) / (2.0 * step)
        scale = np.maximum(1.0, np.maximum(np.abs(analytic), np.abs(finite)))
        self.assertLess(float(np.max(np.abs(analytic - finite) / scale)), 1e-5)

    def test_r10_layout_replays_nine_local_and_one_wide_start(self):
        expression, _, source = _two_lane_payload()
        rows = len(source["target_values"])
        prepared = {
            "lane_contexts": source["lane_contexts"],
            "approach_lanes": {"A": source["block_lanes"]},
            "approach_weights": {
                "A": {
                    "lanes": source["block_lanes"],
                    "weights": source["weights"],
                    "total_flow": np.ones(rows),
                    "valid_mask": source["valid"],
                    "zero_flow_policy": "na",
                    "zero_flow_rows": 0,
                }
            },
            "n_rows": rows,
            "zero_flow_policy": "na",
        }
        kwargs = {
            "universal_expr": expression,
            "lanes": source["block_lanes"],
            "approach_targets": {"A": pd.Series(source["target_values"])},
            "prepared": prepared,
            "n_restarts": 10,
            "maxiter": 20,
            "maxfun": 500,
            "coefficient_bounds_override": None,
        }
        _, first, seed_a = generate_mixed_restart_payloads(
            rng=np.random.default_rng(91), **kwargs
        )
        _, second, seed_b = generate_mixed_restart_payloads(
            rng=np.random.default_rng(91), **kwargs
        )
        payload = first[0]
        self.assertEqual(
            payload["restart_kinds"],
            ["official_local_replay"] * 9
            + ["sobol_role_wide"],
        )
        self.assertEqual(payload["initials"].shape[0], 10)
        self.assertEqual(seed_a, seed_b)
        np.testing.assert_allclose(payload["initials"], second[0]["initials"])
        _, official = _generate_official_payloads(
            rng=np.random.default_rng(91), **kwargs
        )
        np.testing.assert_allclose(
            payload["initials"][:9], official[0]["initials"][:9]
        )
        lower = np.asarray([item[0] for item in payload["bounds"]])
        upper = np.asarray([item[1] for item in payload["bounds"]])
        self.assertTrue(np.all(payload["initials"] >= lower))
        self.assertTrue(np.all(payload["initials"] <= upper))

    def test_parent_values_are_clipped_and_used_once(self):
        expression, names, source = _two_lane_payload()
        rows = len(source["target_values"])
        prepared = {
            "lane_contexts": source["lane_contexts"],
            "approach_lanes": {"A": source["block_lanes"]},
            "approach_weights": {
                "A": {
                    "lanes": source["block_lanes"],
                    "weights": source["weights"],
                    "total_flow": np.ones(rows),
                    "valid_mask": source["valid"],
                    "zero_flow_policy": "na",
                    "zero_flow_rows": 0,
                }
            },
        }
        warm = {
            lane: {name: 1e9 for name in names}
            for lane in source["block_lanes"]
        }
        _, payloads, _ = generate_mixed_restart_payloads(
            universal_expr=expression,
            lanes=source["block_lanes"],
            approach_targets={"A": pd.Series(source["target_values"])},
            prepared=prepared,
            rng=np.random.default_rng(5),
            n_restarts=10,
            maxiter=20,
            maxfun=500,
            coefficient_bounds_override=None,
            warm_parameters=warm,
        )
        payload = payloads[0]
        self.assertEqual(payload["restart_kinds"][0], "parent_warm")
        self.assertEqual(payload["restart_kinds"].count("official_local_replay"), 8)
        self.assertEqual(payload["restart_kinds"][-1], "sobol_role_wide")
        self.assertEqual(
            payload["warm_values_inherited"],
            len(source["block_lanes"]) * len(names),
        )
        np.testing.assert_allclose(
            payload["initials"][0],
            np.asarray([bound[1] for bound in payload["bounds"]]),
        )

    def test_mixed_restart_requires_local_and_wide_slots(self):
        expression, _, source = _two_lane_payload()
        rows = len(source["target_values"])
        prepared = {
            "lane_contexts": source["lane_contexts"],
            "approach_weights": {
                "A": {
                    "lanes": source["block_lanes"],
                    "weights": source["weights"],
                    "valid_mask": source["valid"],
                    "zero_flow_rows": 0,
                    "zero_flow_policy": "na",
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "at least two"):
            generate_mixed_restart_payloads(
                universal_expr=expression,
                lanes=source["block_lanes"],
                approach_targets={"A": pd.Series(source["target_values"])},
                prepared=prepared,
                rng=np.random.default_rng(5),
                n_restarts=1,
                maxiter=20,
                maxfun=500,
                coefficient_bounds_override=None,
            )


class TrainingSelectionTest(unittest.TestCase):
    def test_incumbent_reserves_one_top_k_slot(self):
        records = [
            {
                "expression": "generated_best",
                "candidate_source": "generated_v3",
                "standard_physical_pass": True,
                "enhanced_physical_pass": True,
                "search_composite": 0.9,
            },
            {
                "expression": "generated_second",
                "candidate_source": "generated_v3",
                "standard_physical_pass": True,
                "enhanced_physical_pass": True,
                "search_composite": 0.8,
            },
            {
                "expression": "incumbent",
                "candidate_source": "retained_v2_incumbent",
                "standard_physical_pass": True,
                "enhanced_physical_pass": True,
                "search_composite": 0.1,
            },
        ]
        selected = select_search_candidates(records, top_k=2)
        self.assertEqual(
            [item["expression"] for item in selected],
            ["incumbent", "generated_best"],
        )

    def test_balanced_folds_are_deterministic_and_balanced(self):
        targets = {
            "N": pd.Series(np.linspace(1.0, 100.0, 41)),
            "S": pd.Series(np.linspace(4.0, 90.0, 41) ** 1.1),
        }
        first = balanced_regression_folds(targets, n_folds=4, seed=17)
        second = balanced_regression_folds(targets, n_folds=4, seed=17)
        np.testing.assert_array_equal(first, second)
        counts = np.bincount(first, minlength=4)
        self.assertLessEqual(int(np.max(counts) - np.min(counts)), 1)

    def test_prediction_metrics_reward_exact_predictions(self):
        truth = {"A": pd.Series([1.0, 2.0, 3.0, 4.0])}
        exact = score_predictions(truth, {"A": np.asarray([1.0, 2.0, 3.0, 4.0])})
        self.assertAlmostEqual(exact["macro_r2"], 1.0)
        self.assertAlmostEqual(exact["macro_rmse"], 0.0)
        self.assertAlmostEqual(exact["macro_mae"], 0.0)

    def test_json_serialization_replaces_python_and_numpy_infinity(self):
        converted = _jsonable(
            {
                "python": float("inf"),
                "numpy": np.float64(-np.inf),
                "array": np.asarray([1.0, np.inf]),
                "ok": 2.0,
            }
        )
        self.assertIsNone(converted["python"])
        self.assertIsNone(converted["numpy"])
        self.assertEqual(converted["array"], [1.0, None])
        self.assertEqual(converted["ok"], 2.0)


class EnhancedPhysicsTest(unittest.TestCase):
    def test_search_audit_requires_standard_and_symbolic_endpoints(self):
        passing = audit_search_physics(
            "Cycle_Time * a1 * flow_lane / GR_phase",
            standard_joint_pass=True,
        )
        self.assertTrue(passing["joint_pass"])
        self.assertTrue(passing["dense_audit_deferred_to_cv"])
        standard_failure = audit_search_physics(
            "Cycle_Time * a1 * flow_lane / GR_phase",
            standard_joint_pass=False,
        )
        self.assertFalse(standard_failure["joint_pass"])

    def test_exact_r8_and_symbolic_r9_pass_without_guard(self):
        expression = "Cycle_Time * a1 * flow_lane / GR_phase"
        parameters = {"N_T": {"a1": 1.0}}
        audit = audit_fitted_physics(expression, parameters, ["N_T"])
        self.assertTrue(audit["joint_pass"], audit["errors"])
        self.assertTrue(audit["symbolic_r8_exact"])
        self.assertTrue(audit["symbolic_r9_positive_infinity"])
        self.assertFalse(audit["fixed_r9_guard_applied"])

    def test_finite_zero_green_limit_fails_enhanced_r9(self):
        expression = "Cycle_Time * a1 * flow_lane / (GR_phase + a2)"
        parameters = {"N_T": {"a1": 1.0, "a2": 0.1}}
        audit = audit_fitted_physics(expression, parameters, ["N_T"])
        self.assertFalse(audit["joint_pass"])
        self.assertFalse(audit["symbolic_r9_positive_infinity"])


if __name__ == "__main__":
    unittest.main()
