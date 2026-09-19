import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from expression_adaptation_lane import parse_llm_response, safe_generate_universal_lane_expression
from expression_rules import validate_candidate_expression
from optimization_lane import (
    build_approach_weights,
    build_parameter_bounds,
    calculate_approach_delays_from_universal,
    extract_coefficients,
)


class ExpressionPipelineTests(unittest.TestCase):
    def test_parses_multiline_expression_without_truncation(self):
        response = """### Expression
```
y = Cycle_Time * (
    a1 * flow_lane / GR_phase
    + a2 * flow_lane**2
)
```
### Explanation
multiline formula
"""
        expression, _, explanation = parse_llm_response(response)
        self.assertEqual(
            expression,
            "Cycle_Time * ( a1 * flow_lane / GR_phase + a2 * flow_lane**2 )",
        )
        self.assertEqual(explanation, "multiline formula")

    def test_parses_last_expression_when_llm_includes_retry_commentary(self):
        response = """### Expression
y = Cycle_Time * ( a1 * flow_lane / GR_phase + a2 * flow_lane**a3 * (1 / GR_phase)**a4 + a5 * log(1 + a6 * flow_lane) / (1 + a7 * GR_phase) )
--- This expression is prohibited because it matches one of the listed duplicates exactly.
--- ### Expression
y = Cycle_Time * ( a1 * flow_lane / GR_phase + a2 * flow_lane**a3 * exp(a4 * (1/GR_phase - 1)) + a5 * log(1 + a6 * flow_lane) * (1 / (GR_phase + a7)) )
--- This is also prohibited.
--- **Now, a truly novel expression:**
--- ### Expression
y = Cycle_Time * ( a1 * flow_lane / GR_phase + a2 * flow_lane**a3 * (1 + a4 / GR_phase)**a5 + a6 * log(1 + a7 * flow_lane / GR_phase) )
--- Check if this is novel:
- None of the banned expressions use (1 + a4 / GR_phase) raised to a power a5.
### Explanation
novel shifted-inverse-green structure
"""
        expression, _, explanation = parse_llm_response(response)
        self.assertEqual(
            expression,
            "Cycle_Time * ( a1 * flow_lane / GR_phase + a2 * flow_lane**a3 * (1 + a4 / GR_phase)**a5 + a6 * log(1 + a7 * flow_lane / GR_phase) )",
        )
        self.assertEqual(explanation, "novel shifted-inverse-green structure")

    def test_rejects_symbolically_proven_wrong_green_direction(self):
        expression = (
            "Cycle_Time*(a1*flow_lane**a2/(1+a3*(1/GR_phase)**a4)"
            "+a5*flow_lane**a6*log(1+a7*flow_lane)**a8)"
        )
        valid, reason = validate_candidate_expression(expression)
        self.assertFalse(valid)
        self.assertIn("non-decreasing with GR_phase", reason)

    def test_rejects_movable_pole_but_allows_nonzero_low_demand_limit(self):
        expression = (
            "a1*Cycle_Time*(flow_lane**a2)*((1/GR_phase-1)**a3)"
            "/(1-a4*flow_lane/GR_phase)"
            "+a5*Cycle_Time*exp(a6*flow_lane/GR_phase)"
        )
        valid, reason = validate_candidate_expression(expression)
        self.assertFalse(valid)
        self.assertIn("denominator", reason)

        positive_intercept = "Cycle_Time*(a1+a2*flow_lane)/GR_phase"
        self.assertTrue(validate_candidate_expression(positive_intercept)[0])

    def test_allows_well_formed_exponential_growth(self):
        expression = "a1*Cycle_Time*flow_lane*exp(a2*flow_lane/GR_phase)"
        self.assertTrue(validate_candidate_expression(expression)[0])

    def test_accepts_positive_safe_structures(self):
        expressions = (
            "a1*Cycle_Time*(flow_lane/GR_phase)**a2",
            "a1*Cycle_Time*flow_lane*log(1+a2*flow_lane/GR_phase)",
            "a1*Cycle_Time*flow_lane*((1/GR_phase)-1)**a2",
        )
        for expression in expressions:
            self.assertTrue(validate_candidate_expression(expression)[0], expression)

    def test_rejects_unit_failure_and_allows_finite_small_green_limit(self):
        unit_failure = (
            "a1*flow_lane**a2/GR_phase**a3*(1+a4*Cycle_Time)"
            "+a5*flow_lane*log(1+a6/GR_phase)"
        )
        finite_small_green = (
            "a1*Cycle_Time*flow_lane**a2/(GR_phase+a3)"
        )
        self.assertIn("time dimension", validate_candidate_expression(unit_failure)[1])
        self.assertTrue(validate_candidate_expression(finite_small_green)[0])

    def test_ast_parameter_roles_cover_nested_forms(self):
        expression = (
            "a1*Cycle_Time*(flow_lane/GR_phase)**(a2+a3)"
            "+a4*Cycle_Time*flow_lane*exp(a5*flow_lane)"
        )
        names = extract_coefficients(expression)
        bounds = dict(zip(names, build_parameter_bounds(expression, names)))
        self.assertEqual(bounds["a2"], (0.05, 5.0))
        self.assertEqual(bounds["a3"], (0.05, 5.0))
        self.assertEqual(bounds["a5"], (0.0001, 1.0))

    def test_generator_retries_instead_of_returning_unsafe_offspring(self):
        unsafe = (
            "a1*Cycle_Time*flow_lane/(1-a2*flow_lane/GR_phase)"
        )
        safe = "a1*Cycle_Time*(flow_lane/GR_phase)**a2"
        responses = [
            f"### Expression\ny = {unsafe}\n### Explanation\nunsafe",
            f"### Expression\ny = {safe}\n### Explanation\nsafe",
        ]
        with patch("expression_adaptation_lane.run_llm", side_effect=responses) as mocked:
            expression, _, _ = safe_generate_universal_lane_expression(
                {}, ["flow_lane", "GR_phase", "Cycle_Time"], max_retries=2
            )
        self.assertEqual(expression, safe)
        self.assertEqual(mocked.call_count, 2)
        second_prompt = mocked.call_args_list[1].args[0]
        self.assertIn("RETRY CORRECTION", second_prompt)
        self.assertIn("denominator can become zero", second_prompt)
        self.assertIn("Cycle_Time times a dimensionless function", second_prompt)
        self.assertNotIn("VALID example", second_prompt)
        self.assertNotIn("INVALID examples", second_prompt)

    def test_generator_reprompts_canonical_duplicates_with_novelty_context(self):
        duplicate = "a1*Cycle_Time*flow_lane/GR_phase"
        novel = "a1*Cycle_Time*(flow_lane/GR_phase)**a2"
        responses = [
            f"### Expression\ny = {duplicate}\n### Explanation\nduplicate",
            f"### Expression\ny = {novel}\n### Explanation\nnovel",
        ]

        with patch("expression_adaptation_lane.run_llm", side_effect=responses) as mocked:
            expression, _, _ = safe_generate_universal_lane_expression(
                {},
                ["flow_lane", "GR_phase", "Cycle_Time"],
                max_retries=2,
                excluded_expressions=[duplicate],
            )

        self.assertEqual(expression, novel)
        self.assertEqual(mocked.call_count, 2)
        first_prompt = mocked.call_args_list[0].args[0]
        second_prompt = mocked.call_args_list[1].args[0]
        self.assertIn("NOVELTY REQUIREMENT", first_prompt)
        self.assertIn(duplicate, first_prompt)
        self.assertIn("algebraically equivalent", second_prompt)

    def test_zero_total_flow_is_na_by_default_and_uniform_only_for_legacy(self):
        lane_contexts = {
            "S_L": {"flow_lane": np.asarray([0.0, 0.2])},
            "S_T": {"flow_lane": np.asarray([0.0, 0.8])},
        }
        approach_lanes = {"S": ["S_L", "S_T"]}
        primary = build_approach_weights(lane_contexts, approach_lanes)["S"]
        legacy = build_approach_weights(
            lane_contexts, approach_lanes, zero_flow_policy="uniform"
        )["S"]

        self.assertEqual(primary["zero_flow_policy"], "na")
        self.assertEqual(primary["zero_flow_rows"], 1)
        np.testing.assert_array_equal(primary["valid_mask"], [False, True])
        self.assertEqual(primary["weights"]["S_L"][0], 0.0)
        self.assertAlmostEqual(primary["weights"]["S_T"][1], 0.8)
        self.assertEqual(legacy["weights"]["S_L"][0], 0.5)

    def test_unknown_zero_flow_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            build_approach_weights({}, {}, zero_flow_policy="guess")

    def test_primary_aggregation_emits_na_for_zero_total_flow(self):
        lane_contexts = {
            "S_T": {
                "flow_lane": np.asarray([0.0, 0.2]),
                "GR_phase": np.asarray([0.5, 0.5]),
                "Cycle_Time": np.asarray([90.0, 90.0]),
            }
        }
        prepared = {
            "lane_contexts": lane_contexts,
            "approach_lanes": {"S": ["S_T"]},
            "approach_weights": build_approach_weights(
                lane_contexts, {"S": ["S_T"]}
            ),
            "n_rows": 2,
            "zero_flow_policy": "na",
        }
        predictions = calculate_approach_delays_from_universal(
            df=pd.DataFrame(index=range(2)),
            universal_expr="a1*Cycle_Time*flow_lane/GR_phase",
            lane_parameters={"S_T": {"a1": 1.0}},
            lanes=["S_T"],
            lane_to_approach={"S_T": "S"},
            intersection_id=1,
            prepared_context=prepared,
            strict=True,
        )["S"]
        self.assertTrue(np.isnan(predictions[0]))
        self.assertAlmostEqual(predictions[1], 36.0)


if __name__ == "__main__":
    unittest.main()
