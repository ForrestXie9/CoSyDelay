"""Tests for deterministic best-restart polishing."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from methods.cosydelay._internal.optimizer_jacobian.jacobian_fitter import (
    fit_lane_parameters_with_jacobian_parallel,
)
from methods.cosydelay._internal.optimizer_jacobian.test_jacobian_fitter import (
    _two_lane_payload,
)
from optimization_lane import unpack_lane_parameters

from .polish_fitter import (
    DEFAULT_POLISH_MAXITER,
    fit_lane_parameters_with_jacobian_polish_parallel,
    polish_lane_parameters_parallel,
)


def _problem():
    expression, names, payload = _two_lane_payload()
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
    return (
        expression,
        names,
        payload,
        lanes,
        prepared,
        targets,
        pd.DataFrame(index=np.arange(rows)),
    )


class PolishFitterTest(unittest.TestCase):
    def test_selected_default_is_polish_200(self):
        self.assertEqual(DEFAULT_POLISH_MAXITER, 200)

    def test_polish_never_selects_a_worse_fit_objective(self):
        expression, names, payload, lanes, prepared, targets, df = _problem()
        starting = unpack_lane_parameters(
            np.asarray(payload["initials"][0], dtype=float),
            lanes,
            names,
        )
        diagnostics = {}
        result = polish_lane_parameters_parallel(
            universal_expr=expression,
            df=df,
            lanes=lanes,
            lane_to_approach={lane: "A" for lane in lanes},
            approach_targets=targets,
            intersection_id=1,
            lane_parameters=starting,
            prepared_context=prepared,
            diagnostics=diagnostics,
            polish_maxiter=20,
            parallel_workers=1,
        )
        self.assertEqual(set(result), set(starting))
        self.assertLessEqual(
            diagnostics["objective_sum_after"],
            diagnostics["objective_sum_before"],
        )
        self.assertEqual(diagnostics["rng_values_consumed"], 0)

    def test_combined_fitter_preserves_base_rng_consumption(self):
        expression, _, _, lanes, prepared, targets, df = _problem()
        base_rng = np.random.default_rng(921)
        polish_rng = np.random.default_rng(921)
        base_diagnostics = {}
        polish_diagnostics = {}
        fit_lane_parameters_with_jacobian_parallel(
            universal_expr=expression,
            df=df,
            lanes=lanes,
            lane_to_approach={lane: "A" for lane in lanes},
            approach_targets=targets,
            intersection_id=1,
            prepared_context=prepared,
            rng=base_rng,
            n_restarts=2,
            diagnostics=base_diagnostics,
            parallel_workers=1,
            maxiter=15,
            maxfun=500,
        )
        fit_lane_parameters_with_jacobian_polish_parallel(
            universal_expr=expression,
            df=df,
            lanes=lanes,
            lane_to_approach={lane: "A" for lane in lanes},
            approach_targets=targets,
            intersection_id=1,
            prepared_context=prepared,
            rng=polish_rng,
            n_restarts=2,
            diagnostics=polish_diagnostics,
            parallel_workers=1,
            base_maxiter=15,
            base_maxfun=500,
            polish_maxiter=5,
            polish_maxfun=500,
        )
        self.assertEqual(base_rng.random(), polish_rng.random())
        self.assertLessEqual(
            polish_diagnostics["objective_sum"],
            polish_diagnostics["base"]["objective_sum"],
        )


if __name__ == "__main__":
    unittest.main()
