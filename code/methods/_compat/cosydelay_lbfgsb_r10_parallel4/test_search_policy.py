"""Regression tests for the opt-in P4/G2 adaptive search policy."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import unittest

from .search_policy import P4G2_ADAPTIVE_SEARCH


class AdaptiveSearchPolicyTest(unittest.TestCase):
    def test_policy_is_training_only_and_has_no_fixed_guard(self):
        policy = P4G2_ADAPTIVE_SEARCH
        self.assertEqual(policy.population, 4)
        self.assertEqual(policy.generations, 2)
        self.assertEqual(policy.early_stop_feasible_count, 2)
        self.assertAlmostEqual(policy.early_stop_train_r2_gap, 0.02)
        self.assertEqual(policy.selection_source, "training")
        self.assertFalse(policy.validation_accessed_during_evolution)
        self.assertFalse(policy.fixed_r9_guard)
        self.assertTrue(policy.retention_status.startswith("official"))

    def test_evolution_kwargs_enable_archive_feedback_and_training_stop(self):
        self.assertEqual(
            P4G2_ADAPTIVE_SEARCH.evolution_kwargs(),
            {
                "use_feasible_archive": True,
                "early_stop_feasible_count": 2,
                "early_stop_train_gap": 0.02,
                "targeted_physical_feedback": True,
            },
        )

    def test_main_and_manifest_retain_p4g2_without_test_execution(self):
        import main

        signature = inspect.signature(main.main)
        self.assertEqual(signature.parameters["generations_list"].default, [2])
        self.assertEqual(signature.parameters["pop_size_list"].default, [4])
        manifest = json.loads(
            (Path(__file__).with_name("method_config.json")).read_text(
                encoding="utf-8"
            )
        )
        retained = manifest["retained_search_policy"]
        self.assertEqual(retained["population"], 4)
        self.assertEqual(retained["generations"], 2)
        self.assertFalse(
            retained["training_only_evidence"]["test_files_opened"]
        )


if __name__ == "__main__":
    unittest.main()
