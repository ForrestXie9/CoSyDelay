"""Regression tests for the retained conservative pre-fit gate."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from .prefit_gate import evaluate_structural_prefit_gate
from .prefit_integration import install_structural_prefit_gate


class StructuralPrefitGateTest(unittest.TestCase):
    def test_accepts_exact_zero_flow_expression(self):
        result = evaluate_structural_prefit_gate(
            "Cycle_Time * a1 * flow_lane / GR_phase"
        )
        self.assertTrue(result.passed, result.reason)
        self.assertTrue(result.checks["r8_zero_flow_identity"])
        self.assertFalse(
            result.diagnostics["neutral_coefficient_r9_probe_run"]
        )
        self.assertFalse(result.diagnostics["fixed_postfit_guard_applied"])

    def test_rejects_nonzero_zero_flow_offset(self):
        result = evaluate_structural_prefit_gate(
            "Cycle_Time * (a1 * flow_lane / GR_phase + a2)"
        )
        self.assertFalse(result.passed)
        self.assertIn("R8", result.reason)

    def test_does_not_hard_filter_neutral_r9(self):
        result = evaluate_structural_prefit_gate(
            "Cycle_Time * a1 * flow_lane / (1 + GR_phase)"
        )
        self.assertTrue(result.passed, result.reason)
        self.assertEqual(
            result.diagnostics["r9_prefit_policy"],
            "deferred_to_postfit_fitted_verifier",
        )

    def test_rejects_missing_time_dimension(self):
        result = evaluate_structural_prefit_gate(
            "a1 * flow_lane / GR_phase"
        )
        self.assertFalse(result.passed)
        self.assertIn("structural", result.reason)

    def test_rejection_is_fed_back_and_regenerated(self):
        import expression_adaptation_lane as adaptation
        import population_evolution_lane as population

        old_prompt = adaptation.PHYSICAL_REQUIREMENTS_STANDARD
        old_generator = population.safe_generate_universal_lane_expression
        responses = [
            "### Expression\ny = Cycle_Time * (a1 * flow_lane / GR_phase + a2)\n"
            "### Explanation\ninvalid offset",
            "### Expression\ny = Cycle_Time * a1 * flow_lane / GR_phase\n"
            "### Explanation\ncorrected",
        ]
        with patch.object(adaptation, "run_llm", side_effect=responses):
            with install_structural_prefit_gate(
                population, adaptation
            ) as audit:
                expression, _, _ = (
                    population.safe_generate_universal_lane_expression(
                        feature_explanations={},
                        universal_features=[
                            "flow_lane",
                            "GR_phase",
                            "Cycle_Time",
                        ],
                        max_retries=2,
                    )
                )
                self.assertEqual(
                    expression,
                    "Cycle_Time * a1 * flow_lane / GR_phase",
                )
                self.assertEqual([item["passed"] for item in audit], [False, True])
                self.assertIn(
                    "y(0, GR_phase, Cycle_Time) = 0 exactly",
                    adaptation.PHYSICAL_REQUIREMENTS_STANDARD,
                )
        self.assertEqual(adaptation.PHYSICAL_REQUIREMENTS_STANDARD, old_prompt)
        self.assertIs(
            population.safe_generate_universal_lane_expression,
            old_generator,
        )


if __name__ == "__main__":
    unittest.main()
