import unittest

from expression_adaptation_lane import (
    build_universal_lane_init_prompt,
    build_universal_lane_mutation_prompt,
)
from expression_rules import validate_candidate_expression, validate_candidate_legality
from expression_validation_lane import (
    LEGACY_RULE_NAME_TO_CURRENT_ID,
    PHYSICAL_RULE_NAMES,
    PHYSICAL_RULE_SCHEMA_ID,
    PhysicalVerifierConfig,
    format_physical_validation_feedback,
    score_fitted_lanes_principlewise,
)


FEATURES = ["flow_lane", "GR_phase", "Cycle_Time"]


class OperationalDomainScoringTests(unittest.TestCase):
    def test_all_rules_pass_for_positive_inverse_green_formula(self):
        result = score_fitted_lanes_principlewise(
            "a1*Cycle_Time*flow_lane/GR_phase",
            {"S_T": {"a1": 1.0}, "N_T": {"a1": 2.0}},
            ["S_T", "N_T"],
            FEATURES,
        )
        self.assertTrue(result.joint_pass)
        self.assertEqual(result.score, 1.0)
        self.assertTrue(all(value == 1.0 for value in result.rule_scores.values()))

    def test_finite_zero_green_offset_fails_r9(self):
        result = score_fitted_lanes_principlewise(
            "a1*Cycle_Time*flow_lane/(GR_phase+a2)",
            {"S_T": {"a1": 1.0, "a2": 0.2}},
            ["S_T"],
            FEATURES,
        )
        self.assertEqual(result.rule_scores["R9_zero_green_limit"], 0.0)
        self.assertFalse(result.joint_pass)
        self.assertEqual(
            result.diagnostics["zero_green_methods_by_lane"]["S_T"],
            "numerical_near_zero_green_magnitude",
        )

    def test_inverse_green_formula_passes_r9_numerically(self):
        result = score_fitted_lanes_principlewise(
            "a1*Cycle_Time*flow_lane/GR_phase",
            {"S_T": {"a1": 1.0}},
            ["S_T"],
            FEATURES,
        )
        self.assertEqual(result.rule_scores["R9_zero_green_limit"], 1.0)
        self.assertEqual(
            result.diagnostics["zero_green_methods_by_lane"]["S_T"],
            "numerical_near_zero_green_magnitude",
        )
        self.assertEqual(
            result.diagnostics["zero_green_symbolic_proof"]["verification_type"],
            "numerical_near_zero_green_magnitude",
        )

    def test_positive_intercept_fails_r8_zero_flow(self):
        result = score_fitted_lanes_principlewise(
            "Cycle_Time*(a1 + a2*flow_lane/GR_phase)",
            {"S_T": {"a1": 1.0, "a2": 1.0}},
            ["S_T"],
            FEATURES,
        )
        self.assertEqual(result.rule_scores["R8_zero_flow_boundary"], 0.0)
        self.assertFalse(result.joint_pass)

    def test_negative_delay_fails_binary_nonnegative_rule(self):
        result = score_fitted_lanes_principlewise(
            "Cycle_Time*(flow_lane-a1)/GR_phase",
            {"S_T": {"a1": 1.0}},
            ["S_T"],
            FEATURES,
        )
        self.assertEqual(result.rule_scores["R6_nonnegative_delay"], 0.0)
        self.assertFalse(result.joint_pass)

    def test_nonfinite_operational_values_fail_r6(self):
        result = score_fitted_lanes_principlewise(
            "Cycle_Time*flow_lane*exp(a1/GR_phase)",
            {"S_T": {"a1": 1000.0}},
            ["S_T"],
            FEATURES,
        )
        self.assertNotIn("R5_finite_real_delay", result.rule_scores)
        self.assertEqual(result.rule_scores["R6_nonnegative_delay"], 0.0)
        self.assertFalse(result.joint_pass)

    def test_extreme_finite_delay_no_longer_fails_removed_r5(self):
        result = score_fitted_lanes_principlewise(
            "Cycle_Time*exp(a1/GR_phase)",
            {"S_T": {"a1": 1.0}},
            ["S_T"],
            FEATURES,
        )
        self.assertNotIn("R5_finite_real_delay", result.rule_scores)
        # Large but finite domain values are allowed after R5 removal; R9 still
        # requires severe near-zero-green delay, which this form satisfies.
        self.assertEqual(result.rule_scores["R9_zero_green_limit"], 1.0)

    def test_grid_monotonicity_rules_are_strict_binary(self):
        result = score_fitted_lanes_principlewise(
            "a1*Cycle_Time*flow_lane*(1-a2*flow_lane)/GR_phase",
            {"S_T": {"a1": 1.0, "a2": 0.9}},
            ["S_T"],
            FEATURES,
        )
        self.assertEqual(result.rule_scores["R2_nondecreasing_flow"], 0.0)
        self.assertFalse(result.joint_pass)
        self.assertEqual(
            result.diagnostics["monotonicity_methods_by_lane"]["S_T"][
                "flow_lane"
            ],
            "sampled_analytic_derivative",
        )

    def test_decidable_derivative_sign_avoids_monotonicity_sampling(self):
        result = score_fitted_lanes_principlewise(
            "a1*Cycle_Time*flow_lane/GR_phase",
            {"S_T": {"a1": 1.0}},
            ["S_T"],
            FEATURES,
        )
        self.assertEqual(
            result.diagnostics["monotonicity_methods_by_lane"]["S_T"],
            {
                "flow_lane": "symbolic_derivative_sign",
                "GR_phase": "symbolic_derivative_sign",
            },
        )

    def test_derivative_timeout_falls_back_to_finite_differences(self):
        result = score_fitted_lanes_principlewise(
            "a1*Cycle_Time*flow_lane/GR_phase",
            {"S_T": {"a1": 1.0}},
            ["S_T"],
            FEATURES,
            config=PhysicalVerifierConfig(
                symbolic_derivative_timeout_seconds=1e-12
            ),
        )
        self.assertTrue(result.joint_pass)
        self.assertEqual(
            result.diagnostics["monotonicity_methods_by_lane"]["S_T"],
            {
                "flow_lane": "finite_difference_fallback",
                "GR_phase": "finite_difference_fallback",
            },
        )
        self.assertIn(
            "TimeoutError", result.diagnostics["derivative_fallback_reason"]
        )

    def test_flat_response_fails_operational_responsiveness(self):
        result = score_fitted_lanes_principlewise(
            "Cycle_Time*(a1 + 0*flow_lane + 0*GR_phase)",
            {"S_T": {"a1": 1.0}},
            ["S_T"],
            FEATURES,
        )
        self.assertEqual(
            result.rule_scores["R7_operational_responsiveness"], 0.0
        )

    def test_inverse_green_exponential_still_evaluates_r9(self):
        expression = (
            "Cycle_Time*(a1*flow_lane**a2/"
            "(a3+log(1+a4/GR_phase)) + "
            "a5*exp(a6/GR_phase)*(1-exp(-a7*flow_lane)) + a8)"
        )
        parameters = {
            "S_T": {
                "a1": 0.2,
                "a2": 1.3,
                "a3": 0.4,
                "a4": 0.2,
                "a5": 0.1,
                "a6": 0.1,
                "a7": 0.5,
                "a8": 0.0,
            }
        }
        result = score_fitted_lanes_principlewise(
            expression, parameters, ["S_T"], FEATURES
        )
        self.assertNotIn("fatal_error", result.diagnostics)
        self.assertTrue(result.diagnostics["endpoint_limits_evaluated"])
        self.assertEqual(result.rule_scores["R9_zero_green_limit"], 1.0)

    def test_serialized_rule_order_and_schema_match_operational_protocol(self):
        self.assertEqual(
            PHYSICAL_RULE_NAMES,
            (
                "R1_required_variables",
                "R2_nondecreasing_flow",
                "R3_nonincreasing_green",
                "R4_time_dimension",
                "R6_nonnegative_delay",
                "R7_operational_responsiveness",
                "R8_zero_flow_boundary",
                "R9_zero_green_limit",
            ),
        )
        result = score_fitted_lanes_principlewise(
            "a1*Cycle_Time*flow_lane/GR_phase",
            {"S_T": {"a1": 1.0}},
            ["S_T"],
            FEATURES,
        ).to_dict()
        self.assertEqual(
            PHYSICAL_RULE_SCHEMA_ID,
            "operational_domain_binary_hybrid_2026_07_14_v12",
        )
        self.assertEqual(result["rule_schema_id"], PHYSICAL_RULE_SCHEMA_ID)
        self.assertEqual(result["rule_order"], list(PHYSICAL_RULE_NAMES))
        self.assertEqual(list(result["rule_scores"]), list(PHYSICAL_RULE_NAMES))

    def test_legacy_name_map_is_identification_only(self):
        self.assertEqual(
            LEGACY_RULE_NAME_TO_CURRENT_ID["R2_zero_flow_boundary"],
            "R8_zero_flow_boundary",
        )
        self.assertEqual(
            LEGACY_RULE_NAME_TO_CURRENT_ID["R7_zero_green_limit"],
            "R9_zero_green_limit",
        )
        self.assertNotIn("R2_zero_flow_boundary", PHYSICAL_RULE_NAMES)

    def test_legality_filter_does_not_leak_physical_rules(self):
        expression = "a1 + a2*flow_lane"
        self.assertTrue(validate_candidate_legality(expression)[0])
        self.assertFalse(validate_candidate_expression(expression)[0])

    def test_generic_prompt_withholds_verifier_knowledge(self):
        prompt = build_universal_lane_init_prompt(
            {}, FEATURES, include_physical_knowledge=False
        )
        mutation = build_universal_lane_mutation_prompt(
            "a1+a2*flow_lane",
            "",
            "",
            (False, "R2: delay decreases with flow"),
            "large",
            FEATURES,
            include_physical_knowledge=False,
        )
        self.assertNotIn("operational domain", prompt.lower())
        self.assertNotIn("R2", mutation)
        self.assertIn("intentionally withheld", mutation)

    def test_knowledge_prompt_states_endpoint_conditions(self):
        prompt = build_universal_lane_init_prompt({}, FEATURES)
        self.assertIn("0.04<=GR_phase<=0.85", prompt)
        self.assertIn("flow_lane -> 0+", prompt)
        self.assertIn("finite, and is non-negative", prompt)
        self.assertIn("No condition requires this limit to equal zero", prompt)
        self.assertNotIn("y(0, GR_phase, Cycle_Time) = 0", prompt)
        self.assertIn("lim_{GR_phase -> 0+}", prompt)
        self.assertIn("+infinity", prompt)
        self.assertNotIn("1e4", prompt)
        self.assertNotIn("1e-6", prompt)
        self.assertIn("TRAFFIC INTERPRETABILITY", prompt)
        self.assertIn("simplest structure", prompt)
        self.assertIn("3--6 fitted coefficients", prompt)
        self.assertIn("If a term cannot be interpreted, remove it", prompt)

    def test_complex_inverse_green_sum_passes_r9_numerically(self):
        expression = (
            "Cycle_Time * a1 * flow_lane / (GR_phase**a2) * "
            "(exp(a3 * flow_lane / GR_phase) - 1) + "
            "Cycle_Time * a4 * flow_lane**a5 * (1 / GR_phase - 1)**a6"
        )
        parameters = {
            "S_T": {
                "a1": 0.5,
                "a2": 1.1,
                "a3": 0.2,
                "a4": 0.3,
                "a5": 1.2,
                "a6": 0.8,
            }
        }
        result = score_fitted_lanes_principlewise(
            expression, parameters, ["S_T"], FEATURES
        )
        self.assertEqual(result.rule_scores["R9_zero_green_limit"], 1.0)
        self.assertEqual(
            result.diagnostics["zero_green_methods_by_lane"]["S_T"],
            "numerical_near_zero_green_magnitude",
        )

    def test_complex_exp_passes_r9_numerically(self):
        expression = (
            "Cycle_Time * a1 * flow_lane / (GR_phase**a2) * "
            "(exp(a3 * flow_lane / GR_phase) - 1)"
        )
        result = score_fitted_lanes_principlewise(
            expression,
            {"S_T": {"a1": 0.5, "a2": 1.1, "a3": 0.2}},
            ["S_T"],
            FEATURES,
        )
        self.assertEqual(result.rule_scores["R9_zero_green_limit"], 1.0)
        self.assertEqual(
            result.diagnostics["zero_green_methods_by_lane"]["S_T"],
            "numerical_near_zero_green_magnitude",
        )

    def test_slow_log_growth_fails_r9_without_symbolic_rescue(self):
        result = score_fitted_lanes_principlewise(
            "a1*Cycle_Time*flow_lane*log(1+1/GR_phase)",
            {"S_T": {"a1": 1.0}},
            ["S_T"],
            FEATURES,
        )
        self.assertEqual(result.rule_scores["R9_zero_green_limit"], 0.0)
        self.assertEqual(
            result.diagnostics["zero_green_methods_by_lane"]["S_T"],
            "numerical_near_zero_green_magnitude",
        )

    def test_verifier_config_requires_exactly_eight_weights(self):
        with self.assertRaises(ValueError):
            PhysicalVerifierConfig(rule_weights=(1.0, 1.0))

    def test_physical_feedback_is_complete_and_mathematical(self):
        result = score_fitted_lanes_principlewise(
            "Cycle_Time*(flow_lane-a1)/GR_phase",
            {"S_T": {"a1": 1.0}, "E_T": {"a1": 1.0}},
            ["S_T", "E_T"],
            FEATURES,
        )
        feedback = format_physical_validation_feedback(result)
        self.assertIn("Failed joint rules:", feedback)
        self.assertIn("Green monotonicity:", feedback)
        self.assertNotIn("Lane-specific violations:", feedback)
        self.assertNotIn("Mathematical fixes to prioritize:", feedback)
        self.assertNotIn("131/131", feedback)


if __name__ == "__main__":
    unittest.main()
