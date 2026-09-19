"""Regression tests for the retained approach-parallel fitter."""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager

import numexpr as ne
import numpy as np
import pandas as pd

from optimization_lane import fit_lane_parameters_to_approaches

from .parallel_fitter import fit_lane_parameters_to_approaches_parallel


@contextmanager
def _optimizer_budget(maxiter: int, maxfun: int):
    requested = {
        "COSY_OPTIMIZER_MAXITER": str(maxiter),
        "COSY_OPTIMIZER_MAXFUN": str(maxfun),
    }
    previous = {key: os.environ.get(key) for key in requested}
    os.environ.update(requested)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _synthetic_problem():
    rows = 24
    lanes = ["A_L", "B_L", "C_L"]
    lane_to_approach = {"A_L": "A", "B_L": "B", "C_L": "C"}
    coefficients = {"A": 0.22, "B": 0.37, "C": 0.51}
    lane_contexts = {}
    approach_weights = {}
    targets = {}
    for index, (lane, approach) in enumerate(lane_to_approach.items()):
        flow = np.linspace(0.4 + index, 2.1 + index, rows)
        green = np.linspace(0.31, 0.67, rows)
        cycle = np.linspace(55.0, 105.0, rows)
        lane_contexts[lane] = {
            "flow_lane": flow,
            "GR_phase": green,
            "Cycle_Time": cycle,
        }
        approach_weights[approach] = {
            "lanes": [lane],
            "weights": {lane: np.ones(rows)},
            "total_flow": flow.copy(),
            "valid_mask": np.ones(rows, dtype=bool),
            "zero_flow_policy": "na",
            "zero_flow_rows": 0,
        }
        targets[approach] = pd.Series(
            cycle * coefficients[approach] * flow / green
        )
    prepared = {
        "lane_contexts": lane_contexts,
        "approach_lanes": {
            approach: value["lanes"]
            for approach, value in approach_weights.items()
        },
        "approach_weights": approach_weights,
        "n_rows": rows,
        "zero_flow_policy": "na",
    }
    return (
        pd.DataFrame(index=np.arange(rows)),
        lanes,
        lane_to_approach,
        targets,
        prepared,
    )


class ParallelFitterEquivalenceTest(unittest.TestCase):
    def test_matches_serial_parameters_objectives_and_rng_state(self):
        ne.set_num_threads(1)
        df, lanes, lane_map, targets, prepared = _synthetic_problem()
        expression = "Cycle_Time * a1 * flow_lane / GR_phase"
        serial_rng = np.random.default_rng(20260731)
        parallel_rng = np.random.default_rng(20260731)
        serial_diagnostics = {}
        parallel_diagnostics = {}

        with _optimizer_budget(60, 1000):
            serial = fit_lane_parameters_to_approaches(
                universal_expr=expression,
                df=df,
                lanes=lanes,
                lane_to_approach=lane_map,
                approach_targets=targets,
                intersection_id=1,
                prepared_context=prepared,
                rng=serial_rng,
                n_restarts=4,
                diagnostics=serial_diagnostics,
            )
        parallel = fit_lane_parameters_to_approaches_parallel(
            universal_expr=expression,
            df=df,
            lanes=lanes,
            lane_to_approach=lane_map,
            approach_targets=targets,
            intersection_id=1,
            prepared_context=prepared,
            rng=parallel_rng,
            n_restarts=4,
            diagnostics=parallel_diagnostics,
            parallel_workers=3,
            maxiter=60,
            maxfun=1000,
        )

        self.assertEqual(serial.keys(), parallel.keys())
        for lane in lanes:
            self.assertEqual(serial[lane].keys(), parallel[lane].keys())
            for name in serial[lane]:
                self.assertAlmostEqual(
                    serial[lane][name], parallel[lane][name], places=12
                )
        self.assertAlmostEqual(
            serial_diagnostics["objective_sum"],
            parallel_diagnostics["objective_sum"],
            places=12,
        )
        self.assertEqual(
            [item["chosen_restart"] for item in serial_diagnostics["approaches"]],
            [
                item["chosen_restart"]
                for item in parallel_diagnostics["approaches"]
            ],
        )
        self.assertEqual(serial_rng.random(), parallel_rng.random())


if __name__ == "__main__":
    unittest.main()
