"""Correctness tests for the prospective analytic Jacobian."""

from __future__ import annotations

import unittest

import numexpr as ne
import numpy as np
import pandas as pd

from methods.cosydelay_lbfgsb_r10_parallel3.parallel_fitter import (
    _generate_payloads,
)

from .jacobian_fitter import (
    _mse_and_gradient,
    _symbolic_derivative_strings,
    fit_lane_parameters_with_jacobian_parallel,
)


def _two_lane_payload():
    rows = 31
    lanes = ["A_L", "A_T"]
    flow_left = np.linspace(0.25, 2.0, rows)
    flow_through = np.linspace(0.5, 3.2, rows)
    total = flow_left + flow_through
    green = np.linspace(0.28, 0.72, rows)
    cycle = np.linspace(50.0, 110.0, rows)
    contexts = {
        "A_L": {
            "flow_lane": flow_left,
            "GR_phase": green,
            "Cycle_Time": cycle,
        },
        "A_T": {
            "flow_lane": flow_through,
            "GR_phase": green,
            "Cycle_Time": cycle,
        },
    }
    weights = {
        "A_L": flow_left / total,
        "A_T": flow_through / total,
    }
    expression = (
        "Cycle_Time * (a1 * flow_lane**a2 / GR_phase**a3 * "
        "exp(a4 / GR_phase) + a5 * flow_lane**a6 * "
        "(1 - GR_phase)**a7)"
    )
    target = pd.Series(np.linspace(5.0, 24.0, rows))
    prepared = {
        "lane_contexts": contexts,
        "approach_lanes": {"A": lanes},
        "approach_weights": {
            "A": {
                "lanes": lanes,
                "weights": weights,
                "total_flow": total,
                "valid_mask": np.ones(rows, dtype=bool),
                "zero_flow_policy": "na",
                "zero_flow_rows": 0,
            }
        },
        "n_rows": rows,
        "zero_flow_policy": "na",
    }
    names, payloads = _generate_payloads(
        universal_expr=expression,
        lanes=lanes,
        approach_targets={"A": target},
        prepared=prepared,
        rng=np.random.default_rng(41),
        n_restarts=2,
        maxiter=50,
        maxfun=1000,
        coefficient_bounds_override=None,
    )
    payloads[0]["derivative_expressions"] = (
        _symbolic_derivative_strings(expression, tuple(names))
    )
    return expression, names, payloads[0]


class AnalyticJacobianTest(unittest.TestCase):
    def test_gradient_matches_centered_finite_difference(self):
        ne.set_num_threads(1)
        _, _, payload = _two_lane_payload()
        values = np.linspace(0.21, 0.79, len(payload["initials"][0]))
        _, analytic = _mse_and_gradient(values, payload)
        finite = np.zeros_like(values)
        for index, value in enumerate(values):
            step = 1e-6 * max(1.0, abs(float(value)))
            left = values.copy()
            right = values.copy()
            left[index] -= step
            right[index] += step
            left_value, _ = _mse_and_gradient(left, payload)
            right_value, _ = _mse_and_gradient(right, payload)
            finite[index] = (right_value - left_value) / (2.0 * step)
        scale = np.maximum(1.0, np.maximum(np.abs(analytic), np.abs(finite)))
        relative_error = np.max(np.abs(analytic - finite) / scale)
        self.assertLess(relative_error, 1e-5)

    def test_fit_consumes_same_random_stream_as_frozen_start_generator(self):
        expression, _, payload = _two_lane_payload()
        lanes = list(payload["block_lanes"])
        rows = len(payload["target_values"])
        prepared = {
            "lane_contexts": payload["lane_contexts"],
            "approach_lanes": {"A": lanes},
            "approach_weights": {
                "A": {
                    "lanes": lanes,
                    "weights": payload["weights"],
                    "total_flow": np.ones(rows),
                    "valid_mask": payload["valid"],
                    "zero_flow_policy": "na",
                    "zero_flow_rows": 0,
                }
            },
            "n_rows": rows,
            "zero_flow_policy": "na",
        }
        targets = {"A": pd.Series(payload["target_values"])}
        fitter_rng = np.random.default_rng(887)
        reference_rng = np.random.default_rng(887)
        _generate_payloads(
            universal_expr=expression,
            lanes=lanes,
            approach_targets=targets,
            prepared=prepared,
            rng=reference_rng,
            n_restarts=3,
            maxiter=20,
            maxfun=500,
            coefficient_bounds_override=None,
        )
        diagnostics = {}
        result = fit_lane_parameters_with_jacobian_parallel(
            universal_expr=expression,
            df=pd.DataFrame(index=np.arange(rows)),
            lanes=lanes,
            lane_to_approach={lane: "A" for lane in lanes},
            approach_targets=targets,
            intersection_id=1,
            prepared_context=prepared,
            rng=fitter_rng,
            n_restarts=3,
            diagnostics=diagnostics,
            parallel_workers=1,
            maxiter=20,
            maxfun=500,
        )
        self.assertEqual(set(result), set(lanes))
        self.assertEqual(fitter_rng.random(), reference_rng.random())
        self.assertEqual(diagnostics["jacobian"], "analytic_sympy_numexpr")


if __name__ == "__main__":
    unittest.main()
