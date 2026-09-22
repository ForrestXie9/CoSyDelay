"""Non-Test-data checks for the frozen two-stage evaluation runner."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from .run_locked_test_evaluation import (
    _aggregate,
    _frozen_manifest,
    _metrics,
)


class LockedTestProtocolTest(unittest.TestCase):
    def test_metrics_report_raw_and_error_metrics(self):
        result = _metrics(
            np.asarray([0.0, 1.0, 2.0]),
            np.asarray([0.0, 1.0, 2.0]),
        )
        self.assertAlmostEqual(result["r2"], 1.0)
        self.assertAlmostEqual(result["rmse"], 0.0)
        self.assertAlmostEqual(result["mae"], 0.0)
        self.assertEqual(result["rows"], 3)

    def test_aggregate_preserves_raw_test_r2(self):
        evaluations = [
            {
                "intersection_id": 1,
                "physical_joint_pass": True,
                "train_r2": 0.8,
                "test_macro": {
                    "r2": -0.2,
                    "r2_clipped": 0.0,
                    "rmse": 2.0,
                    "mae": 1.0,
                    "mape": 5.0,
                },
                "test_pooled": {"r2": -0.1, "rmse": 2.2},
            }
        ]
        result = _aggregate(evaluations)
        self.assertAlmostEqual(result["groups"][0]["test_macro_r2"]["mean"], -0.2)
        self.assertAlmostEqual(
            result["groups"][0]["test_macro_r2_clipped"]["mean"], 0.0
        )

    def test_freeze_manifest_precedes_test_access(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = _frozen_manifest(
                Path(directory),
                SimpleNamespace(
                    intersections=[1, 6],
                    runs=3,
                    base_seed=20260712,
                ),
            )
        self.assertFalse(manifest["test_files_opened_at_freeze"])
        self.assertEqual(manifest["runs_per_intersection"], 3)
        self.assertTrue(manifest["freeze_id"])


if __name__ == "__main__":
    unittest.main()
