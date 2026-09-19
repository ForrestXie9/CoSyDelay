"""Four-approach equivalence regression for the retained fitter."""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager

import numexpr as ne
import numpy as np
import pandas as pd

from optimization_lane import fit_lane_parameters_to_approaches

from .parallel_fitter import (
    OFFICIAL_PARALLEL_WORKERS,
    fit_lane_parameters_to_approaches_parallel,
)


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


def _four_approach_problem():
    rows = 18
    approaches = ("A", "B", "C", "D")
    lanes = [f"{approach}_L" for approach in approaches]
    lane_to_approach = {
        lane: approach for lane, approach in zip(lanes, approaches)
    }
    lane_contexts = {}
    approach_weights = {}
    targets = {}
    for index, (lane, approach) in enumerate(lane_to_approach.items()):
        flow = np.linspace(0.3 + 0.1 * index, 1.5 + 0.1 * index, rows)
        green = np.linspace(0.25, 0.75, rows)
        cycle = np.linspace(60.0, 120.0, rows)
        coefficient = 0.2 + 0.1 * index
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
            cycle * coefficient * flow / green
        )
    prepared = {
        "lane_contexts": lane_contexts,
        "approach_lanes": {
            approach: item["lanes"]
            for approach, item in approach_weights.items()
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


class ParallelFourEquivalenceTest(unittest.TestCase):
    def test_default_cap_is_four_and_matches_serial(self):
        self.assertEqual(OFFICIAL_PARALLEL_WORKERS, 4)
        ne.set_num_threads(1)
        df, lanes, lane_map, targets, prepared = _four_approach_problem()
        expression = "Cycle_Time * a1 * flow_lane / GR_phase"
        serial_rng = np.random.default_rng(20260803)
        parallel_rng = np.random.default_rng(20260803)
        serial_diagnostics = {}
        parallel_diagnostics = {}

        with _optimizer_budget(50, 600):
            serial = fit_lane_parameters_to_approaches(
                universal_expr=expression,
                df=df,
                lanes=lanes,
                lane_to_approach=lane_map,
                approach_targets=targets,
                intersection_id=1,
                prepared_context=prepared,
                rng=serial_rng,
                n_restarts=2,
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
            n_restarts=2,
            diagnostics=parallel_diagnostics,
            maxiter=50,
            maxfun=600,
        )

        self.assertEqual(parallel_diagnostics["parallel_workers"], 4)
        self.assertEqual(len(parallel_diagnostics["approaches"]), 4)
        for lane in lanes:
            for name, value in serial[lane].items():
                self.assertAlmostEqual(value, parallel[lane][name], places=12)
        self.assertAlmostEqual(
            serial_diagnostics["objective_sum"],
            parallel_diagnostics["objective_sum"],
            places=12,
        )
        self.assertEqual(serial_rng.random(), parallel_rng.random())


if __name__ == "__main__":
    unittest.main()
