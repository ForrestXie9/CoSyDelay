"""Unit tests for the isolated manuscript-aligned V16 foundation."""

from contextlib import contextmanager
import json
import math
from types import SimpleNamespace

import expression_validation_lane as shared

from . import integration
from . import fitter as v16_fitter
from .integration import (
    format_principle_feedback,
    principlewise_fitness,
    validate_unambiguous_coefficient_roles,
)
from .physics import (
    MANUSCRIPT_RULE_NAMES,
    MANUSCRIPT_RULE_SCHEMA_ID,
    ManuscriptVerifierConfig,
    score_fitted_lanes_manuscript_principlewise,
    _check_exact_time_dimension,
    _generate_v16_test_points,
)
from .policy import V16_POLICY
from .contract import (
    V16_CONTRACT,
    V16_MAX_PROMPT_EXCLUSIONS,
    validate_v16_contract,
)
from .prompt import (
    OUTPUT_SCHEMA,
    PHYSICAL_REQUIREMENTS,
    build_init_prompt,
    build_mutation_prompt,
    move_output_schema_last,
    output_schema_is_last,
    PROMPT_CONTRACT_ID,
)
from methods.cosydelay.engine.numeric_fitting.policy import V15_POLICY
from .source_manifest import V16_SOURCE_FILES, validate_v16_source_manifest
from .run_training_pilot import (
    _diagnostic_epsilon_parsimony,
    _force_principlewise_evolution,
)
from .run_p10g10_training import (
    _audit_content_integrity,
    _restart_composition_audit,
)
from .run_offline_training_smoke import EXPRESSION_PROFILES
from .evaluate_frozen_test import (
    DEFAULT_BASELINE_DIR,
    FROZEN_FIXED17_SUMMARY_SHA256,
    pooled_metrics,
)
from .launch_p10g10_i1_i6 import validate_formal_result


FEATURES = ["flow_lane", "GR_phase", "Cycle_Time"]


def _valid_formal_physics_payload():
    scores = {name: 1.0 for name in MANUSCRIPT_RULE_NAMES}
    movements = (
        "S_L", "S_T", "S_R", "E_L", "E_T", "E_R",
        "N_L", "N_T", "N_R", "W_L", "W_T", "W_R",
    )
    return {
        "rule_schema_id": MANUSCRIPT_RULE_SCHEMA_ID,
        "rule_order": list(MANUSCRIPT_RULE_NAMES),
        "score": 1.0,
        "joint_pass": True,
        "rule_scores": dict(scores),
        "lane_rule_scores": {lane: dict(scores) for lane in movements},
        "config": V16_CONTRACT.to_dict()["verifier"],
    }


def _valid_formal_parameters():
    return {
        lane: {"a1": 1.0}
        for lane in (
            "S_L", "S_T", "S_R", "E_L", "E_T", "E_R",
            "N_L", "N_T", "N_R", "W_L", "W_T", "W_R",
        )
    }


def test_inverse_green_expression_passes_all_seven_rules():
    observed = score_fitted_lanes_manuscript_principlewise(
        "a1*Cycle_Time*flow_lane/GR_phase",
        {"S_T": {"a1": 1.0}},
        ["S_T"],
        FEATURES,
    )
    assert observed.joint_pass
    assert observed.score == 1.0
    assert tuple(observed.rule_scores) == MANUSCRIPT_RULE_NAMES


def test_positive_low_demand_limit_is_allowed():
    observed = score_fitted_lanes_manuscript_principlewise(
        "Cycle_Time*(a1+a2*flow_lane/GR_phase)",
        {"S_T": {"a1": 0.2, "a2": 1.0}},
        ["S_T"],
        FEATURES,
    )
    assert observed.rule_scores[
        "R5_finite_nonnegative_low_demand_limit"
    ] == 1.0
    assert observed.joint_pass
    assert "R8_zero_flow_boundary" not in observed.rule_scores
    displayed = observed.diagnostics["low_demand_limits_by_lane"]["S_T"]
    assert displayed != "0"
    assert "_positive_cycle" in displayed


def test_finite_zero_green_limit_gets_partial_score_not_hard_rejection():
    observed = score_fitted_lanes_manuscript_principlewise(
        "Cycle_Time*(a1+a2*flow_lane/(GR_phase+a3))",
        {"S_T": {"a1": 0.2, "a2": 1.0, "a3": 0.1}},
        ["S_T"],
        FEATURES,
    )
    assert observed.rule_scores["R7_zero_green_limit"] == 0.0
    assert not observed.joint_pass
    assert math.isclose(observed.score, 6.0 / 7.0)
    assert principlewise_fitness(0.8, observed.score) > 1.0
    feedback = format_principle_feedback(observed)
    assert "R1=1.000" in feedback
    assert "R7=0.000" in feedback
    assert "Strict joint pass: NO" in feedback
    assert "score=0.000" in feedback


def test_partial_monotonicity_is_fractional():
    observed = score_fitted_lanes_manuscript_principlewise(
        "Cycle_Time*flow_lane*(1-a1*flow_lane)/GR_phase",
        {"S_T": {"a1": 0.4}},
        ["S_T"],
        FEATURES,
    )
    score = observed.rule_scores["R2_nondecreasing_flow"]
    assert 0.0 < score < 1.0
    assert "R2:" in format_principle_feedback(observed)


def test_score_is_equal_mean_over_movements_and_rules():
    observed = score_fitted_lanes_manuscript_principlewise(
        "Cycle_Time*(a1-a2)*flow_lane/GR_phase",
        {
            "S_T": {"a1": 2.0, "a2": 1.0},
            "E_T": {"a1": 1.0, "a2": 2.0},
        },
        ["S_T", "E_T"],
        FEATURES,
    )
    expected = sum(
        score
        for lane_scores in observed.lane_rule_scores.values()
        for score in lane_scores.values()
    ) / (2 * len(MANUSCRIPT_RULE_NAMES))
    assert math.isclose(observed.score, expected)
    for name in MANUSCRIPT_RULE_NAMES:
        assert math.isclose(
            observed.rule_scores[name],
            sum(
                observed.lane_rule_scores[lane][name]
                for lane in ("S_T", "E_T")
            )
            / 2.0,
        )
    assert observed.diagnostics["aggregation"] == (
        "equal_mean_over_declared_movements_and_seven_rules"
    )


def test_schema_serialization_is_stable_and_shared_globals_are_restored():
    previous_names = shared.PHYSICAL_RULE_NAMES
    previous_schema = shared.PHYSICAL_RULE_SCHEMA_ID
    observed = score_fitted_lanes_manuscript_principlewise(
        "a1*Cycle_Time*flow_lane/GR_phase",
        {"S_T": {"a1": 1.0}},
        ["S_T"],
        FEATURES,
    ).to_dict()
    assert shared.PHYSICAL_RULE_NAMES == previous_names
    assert shared.PHYSICAL_RULE_SCHEMA_ID == previous_schema
    assert observed["rule_schema_id"] == MANUSCRIPT_RULE_SCHEMA_ID
    assert observed["rule_order"] == list(MANUSCRIPT_RULE_NAMES)


def test_config_requires_seven_weights():
    try:
        ManuscriptVerifierConfig(rule_weights=(1.0,) * 8)
    except ValueError as exc:
        assert "seven" in str(exc)
    else:
        raise AssertionError("eight weights unexpectedly accepted")


def test_exact_time_dimension_keeps_coefficients_symbolic():
    assert _check_exact_time_dimension(
        "Cycle_Time*(a1+a2*flow_lane/GR_phase)"
    ) == (True, None)
    power_ok, power_error = _check_exact_time_dimension(
        "Cycle_Time**a1*(a2+a3*flow_lane/GR_phase)"
    )
    assert not power_ok
    assert "degree one" in str(power_error)
    function_ok, function_error = _check_exact_time_dimension(
        "Cycle_Time*exp(a1*Cycle_Time)*flow_lane/GR_phase"
    )
    assert not function_ok
    assert "exp/log" in str(function_error)


def test_end_to_end_r4_rejects_fitted_cycle_power():
    observed = score_fitted_lanes_manuscript_principlewise(
        "Cycle_Time**a1*(a2+a3*flow_lane/GR_phase)",
        {"S_T": {"a1": 2.0, "a2": 0.2, "a3": 1.0}},
        ["S_T"],
        FEATURES,
    )
    assert observed.rule_scores["R4_time_dimension"] == 0.0
    assert not observed.joint_pass
    assert any(
        message.startswith("R4:")
        for message in observed.lane_errors["S_T"]
    )


def test_all_domain_corners_close_joint_boundary_blind_spot():
    points = _generate_v16_test_points(
        128, 42, (0.001, 2.0), (0.02, 0.95), (30.0, 240.0)
    )
    # The retained generator has 128 LHS points plus four anchors, including
    # the exact zero-flow point.  Six missing corners are added; the legacy
    # scorer subsequently excludes only the first zero-flow anchor.
    assert len(points) == 138
    observed_corners = {
        (
            float(point["flow_lane"]),
            float(point["GR_phase"]),
            float(point["Cycle_Time"]),
        )
        for point in points
    }
    expected_corners = {
        (flow, green, cycle)
        for flow in (0.001, 2.0)
        for green in (0.02, 0.95)
        for cycle in (30.0, 240.0)
    }
    assert expected_corners.issubset(observed_corners)

    # This response passes every pre-existing numerical point but has a
    # negative flow derivative at the previously absent high-flow/high-green
    # corner.  The completed corner grid must make R2 fractional.
    observed = score_fitted_lanes_manuscript_principlewise(
        "Cycle_Time*(a1+a2*flow_lane/GR_phase-"
        "a3*flow_lane**8*GR_phase**8)",
        {"S_T": {"a1": 100.0, "a2": 1.0, "a3": 0.002}},
        ["S_T"],
        FEATURES,
    )
    assert 0.0 < observed.rule_scores["R2_nondecreasing_flow"] < 1.0
    assert not observed.joint_pass
    assert observed.diagnostics["grid_points_per_lane"] == 137
    assert observed.diagnostics["grid_anchor_scheme"] == (
        "all_eight_joint_domain_corners"
    )


def test_fitted_limit_evaluation_uses_exact_unrounded_coefficients():
    passing = score_fitted_lanes_manuscript_principlewise(
        "Cycle_Time*(a1-a2+a3*flow_lane/GR_phase)",
        {"S_T": {"a1": 2.0001, "a2": 2.0, "a3": 1.0}},
        ["S_T"],
        FEATURES,
    )
    assert passing.rule_scores[
        "R5_finite_nonnegative_low_demand_limit"
    ] == 1.0
    diagnostics = passing.diagnostics["analytic_limit_fallback"]
    assert diagnostics["mode"] == (
        "always_exact_decimal_rational_fitted_coefficients"
    )
    assert diagnostics["legacy_two_significant_digit_result_overridden"]
    assert diagnostics["exact_fitted_limits_by_movement"]["S_T"]["R5"]
    symbolic = passing.diagnostics["symbolic_limit_evaluation"]
    assert not symbolic["rounded_coefficient_result_used_for_scoring"]
    assert not symbolic[
        "legacy_generic_positive_parameter_checks_executed"
    ]
    assert "fallback_coefficient_significant_digits" not in symbolic

    failing = score_fitted_lanes_manuscript_principlewise(
        "Cycle_Time*(a1-a2+a3*flow_lane/GR_phase)",
        {"S_T": {"a1": 2.0, "a2": 2.0001, "a3": 1.0}},
        ["S_T"],
        FEATURES,
    )
    assert failing.rule_scores[
        "R5_finite_nonnegative_low_demand_limit"
    ] == 0.0
    assert any(
        message.startswith("R5:")
        for message in failing.lane_errors["S_T"]
    )


def test_fitted_equality_can_change_r7_and_is_scored_exactly():
    expression = (
        "Cycle_Time*(a3+a4*flow_lane*GR_phase/"
        "(GR_phase**2+(a1-a2)**2))"
    )
    passing = score_fitted_lanes_manuscript_principlewise(
        expression,
        {"S_T": {"a1": 1.0, "a2": 1.0, "a3": 1.0, "a4": 1.0}},
        ["S_T"],
        FEATURES,
    )
    assert passing.rule_scores["R7_zero_green_limit"] == 1.0
    diagnostics = passing.diagnostics["analytic_limit_fallback"]
    assert not diagnostics[
        "legacy_generic_positive_parameter_checks_executed"
    ]
    assert diagnostics["exact_fitted_limits_by_movement"]["S_T"]["R7"] == (
        "oo"
    )

    failing = score_fitted_lanes_manuscript_principlewise(
        expression,
        {"S_T": {"a1": 2.0, "a2": 1.0, "a3": 1.0, "a4": 1.0}},
        ["S_T"],
        FEATURES,
    )
    assert failing.rule_scores["R7_zero_green_limit"] == 0.0


def test_exceptional_fitted_equality_can_change_r5_and_is_scored_exactly():
    expression = (
        "Cycle_Time*(a3+flow_lane**0.5/(a1-a2+flow_lane)+"
        "a4*flow_lane/GR_phase)"
    )
    failing = score_fitted_lanes_manuscript_principlewise(
        expression,
        {"S_T": {"a1": 1.0, "a2": 1.0, "a3": 1.0, "a4": 1.0}},
        ["S_T"],
        FEATURES,
    )
    diagnostics = failing.diagnostics["analytic_limit_fallback"]
    assert not diagnostics[
        "legacy_generic_positive_parameter_checks_executed"
    ]
    assert failing.rule_scores[
        "R5_finite_nonnegative_low_demand_limit"
    ] == 0.0
    assert diagnostics["exact_fitted_limits_by_movement"]["S_T"]["R5"] == (
        "oo"
    )


def test_serialized_config_contains_no_unused_numeric_r9_threshold():
    observed = score_fitted_lanes_manuscript_principlewise(
        "a1*Cycle_Time*flow_lane/GR_phase",
        {"S_T": {"a1": 1.0}},
        ["S_T"],
        FEATURES,
    ).to_dict()
    assert "zero_green_min_delay_seconds" not in observed["config"]
    assert "zero_green_low_probe_green" not in observed["config"]
    assert 10000 not in observed["config"].values()


def test_prompt_matches_manuscript_boundaries_without_training_feedback():
    init = build_init_prompt({}, FEATURES)
    mutation = build_mutation_prompt(
        "a1*Cycle_Time*flow_lane/GR_phase",
        "",
        "",
        (False, "R5: low-demand limit is not finite"),
        "large",
        FEATURES,
    )
    for text in (PHYSICAL_REQUIREMENTS, init, mutation):
        assert "finite,\n   nonnegative limit" in text
        assert "may be positive" in text
        assert "exact\n   zero-delay identity" in text
        assert "10000" not in text
        assert "RMSE" not in text
        assert "MAE" not in text
    for text in (init, mutation):
        assert "available observed targets are approach-average delays" in text
        assert "jointly calibrated through" in text
        assert "never supplies fitted numerical coefficient values" in text
    assert "PARENT EXPRESSION:" in mutation
    assert "R5: low-demand limit is not finite" in mutation
    assert "same coefficient both" in init


def test_programmatic_prompt_additions_precede_the_final_output_schema():
    base_prompt = build_mutation_prompt(
        "Cycle_Time*(a1+a2*flow_lane/GR_phase)",
        "",
        "",
        (False, "R7=0.000"),
        "large",
        FEATURES,
    )
    appended = base_prompt + (
        "\nNOVELTY REQUIREMENT:\n- banned-expression\n"
        "\nRETRY CORRECTION:\nRejection reason: duplicate\n"
    )
    reordered = move_output_schema_last(appended)
    assert output_schema_is_last(reordered)
    assert reordered.count(OUTPUT_SCHEMA) == 1
    assert reordered.count("PARENT EXPRESSION:") == 1
    assert reordered.count("R7=0.000") == 1
    assert reordered.count("banned-expression") == 1
    assert reordered.count("Rejection reason: duplicate") == 1
    assert reordered.index("NOVELTY REQUIREMENT:") < reordered.index(OUTPUT_SCHEMA)
    assert reordered.index("RETRY CORRECTION:") < reordered.index(OUTPUT_SCHEMA)


def test_prompt_install_reorders_at_api_boundary_and_restores_callable():
    from types import SimpleNamespace
    from .prompt import install_prompt_contract

    captured = []
    original_llm = lambda prompt: captured.append(prompt) or "response"
    module = SimpleNamespace(
        build_universal_lane_init_prompt=lambda *args, **kwargs: "old-init",
        build_universal_lane_mutation_prompt=lambda *args, **kwargs: "old-mutation",
        run_llm=original_llm,
    )
    with install_prompt_contract(adaptation_module=module):
        prompt = build_init_prompt({}, FEATURES) + "\nRETRY CORRECTION:\nfix\n"
        assert module.run_llm(prompt) == "response"
        assert output_schema_is_last(captured[-1])
    assert module.run_llm is original_llm


def test_mixed_nonlinear_coefficient_roles_are_rejected_before_fitting():
    passed, reason = validate_unambiguous_coefficient_roles(
        "Cycle_Time*(a1+(flow_lane/GR_phase)**a2*exp(a2*flow_lane))"
    )
    assert not passed
    assert "coefficient-role ambiguity" in reason
    assert "a2=exp_coefficient+power_exponent" in reason

    passed, reason = validate_unambiguous_coefficient_roles(
        "Cycle_Time*(a1+(flow_lane/GR_phase)**a2*exp(a3*flow_lane))"
    )
    assert passed
    assert "coefficient-role" in reason


def test_coefficient_identifiers_must_be_consecutive_from_a1():
    passed, reason = validate_unambiguous_coefficient_roles(
        "Cycle_Time*(a1+a3*flow_lane/GR_phase)"
    )
    assert not passed
    assert "consecutive beginning at a1" in reason

    passed, reason = validate_unambiguous_coefficient_roles(
        "Cycle_Time*(a1+a2*flow_lane/GR_phase)"
    )
    assert passed
    assert "coefficient-role" in reason


def test_policy_retains_optimizer_and_removes_hard_gate():
    assert V16_POLICY.optimizer == V15_POLICY.optimizer
    assert V16_POLICY.optimizer_restarts == V15_POLICY.optimizer_restarts == 10
    assert V16_POLICY.approach_workers_cap == V15_POLICY.approach_workers_cap == 4
    assert not V16_POLICY.validation_or_test_used_for_selection
    assert not V16_POLICY.fitted_enhanced_physics_is_hard_gate
    assert not V16_POLICY.coefficient_robust_r9_prefit_gate
    assert not V16_CONTRACT.legacy_numeric_r9_probe_executed
    assert V16_CONTRACT.population_update == (
        "mu_plus_lambda_global_elitist_top_m"
    )


def test_v16_worker_omits_and_scrubs_legacy_numeric_r9_probe(monkeypatch):
    previous_probe = v16_fitter.retained._r9_probe_restart
    observed_probe = []

    def fake_solver(payload):
        del payload
        observed_probe.append(v16_fitter.retained._r9_probe_restart(None, None))
        return {
            "r9_constrained_restart_selection": False,
            "r9_feasible_restart_count": 1,
            "r9_feasible_solution_available": True,
            "restarts": [
                {
                    "objective": 1.0,
                    "r9_probe_pass": True,
                    "r9_probe_delays_seconds": {"S_T": 10001.0},
                }
            ],
        }

    monkeypatch.setattr(v16_fitter, "_ORIGINAL_SOLVER", fake_solver)
    result = v16_fitter._solve_approach_without_legacy_probe({})
    assert observed_probe == [(True, {})]
    assert v16_fitter.retained._r9_probe_restart is previous_probe
    assert result["legacy_numeric_r9_probe_executed"] is False
    assert result["restart_selection"] == "minimum_training_mse"
    serialized = repr(result)
    for field in v16_fitter._LEGACY_DIAGNOSTIC_KEYS:
        assert field not in serialized


def test_v16_solver_substitution_is_scoped_and_restored():
    previous_solver = v16_fitter.retained._solve_approach_all_log
    with v16_fitter.legacy_numeric_r9_probe_disabled():
        assert v16_fitter.retained._solve_approach_all_log is (
            v16_fitter._solve_approach_without_legacy_probe
        )
    assert v16_fitter.retained._solve_approach_all_log is previous_solver


def test_integration_replaces_binary_bonus_with_fractional_score(monkeypatch):
    expression = "Cycle_Time*(a1+a2*flow_lane/(GR_phase+a3))"
    parameters = {"S_T": {"a1": 0.2, "a2": 1.0, "a3": 0.1}}
    evaluations = {
        expression: {
            "expression": expression,
            "fitness": 1.8,
            "details": {},
        }
    }
    runtime = SimpleNamespace(evaluations=evaluations)

    def binary_evaluator(*args, **kwargs):
        del args, kwargs
        return (
            1.0,
            0.8,
            1.8,
            parameters,
            "legacy binary pass",
            {
                "status": "evaluated",
                "selection_metrics": {"macro_nonnegative_r2": 0.8},
            },
        )

    generation_calls = []

    def old_generator(*args, **kwargs):
        del args
        generation_calls.append(dict(kwargs))
        return "generated"

    population = SimpleNamespace(
        score_fitted_lanes_principlewise=lambda *args, **kwargs: None,
        evaluate_expression_with_fitting=binary_evaluator,
        safe_generate_universal_lane_expression=old_generator,
    )
    adaptation = SimpleNamespace(
        validate_candidate_expression=lambda value: (True, "old"),
        build_universal_lane_init_prompt=lambda *args, **kwargs: "old init",
        build_universal_lane_mutation_prompt=lambda *args, **kwargs: "old mutation",
        run_llm=lambda prompt: prompt,
    )

    @contextmanager
    def fake_v15(**kwargs):
        del kwargs
        yield runtime

    monkeypatch.setattr(integration, "install_v15_candidate", fake_v15)
    with integration.install_v16_candidate(
        df_train=object(),
        targets={},
        lanes=["S_T"],
        lane_to_approach={"S_T": "S"},
        intersection_id=1,
        population_module=population,
        adaptation_module=adaptation,
    ):
        assert adaptation.validate_candidate_expression(
            "Cycle_Time*(flow_lane**a1+exp(a1/GR_phase))"
        )[0] is False
        population.safe_generate_universal_lane_expression(
            enforce_physical_prefilter=False
        )
        assert generation_calls[-1]["enforce_physical_prefilter"] is True
        observed = population.evaluate_expression_with_fitting(expression)

    assert population.safe_generate_universal_lane_expression is old_generator

    physical_component, accuracy, fitness, _, feedback, details = observed
    assert math.isclose(physical_component, 6.0 / 7.0)
    assert accuracy == 0.8
    assert math.isclose(fitness, 0.8 + 6.0 / 7.0)
    assert not details["verifier"]["joint_pass"]
    assert details["strict_joint_pass_is_diagnostic_only"]
    assert not details["physical_hard_gate"]
    assert details["v16_physics_wall_seconds"] >= 0.0
    assert details["physical_wall_seconds"] == details[
        "v16_physics_wall_seconds"
    ]
    assert details["enhanced_physics_wall_seconds"] == 0.0
    assert details["physical_timing_source"] == "v16_seven_rule_rescore"
    assert details["v16_score_reused_as_enhanced_audit"]
    assert "R7:" in feedback
    assert math.isclose(evaluations[expression]["fitness"], fitness)


def test_source_manifest_is_complete_and_unique():
    validate_v16_source_manifest()
    assert len(V16_SOURCE_FILES) == len(set(V16_SOURCE_FILES))
    names = {path.name for path in V16_SOURCE_FILES}
    assert {
        "physics.py",
        "contract.py",
        "prompt.py",
        "integration.py",
        "fitter.py",
        "run_training_pilot.py",
        "launch_authorized_pilot.py",
        "run_offline_training_smoke.py",
        "benchmark_persistent_limit_worker.py",
        "benchmark_maxiter_i1_i2.py",
        "benchmark_bounds_i1_i2.py",
        "benchmark_bound_components_i2.py",
        "run_p10g10_training.py",
        "launch_p10g10_i1_i6.py",
        "evaluate_frozen_test.py",
    }.issubset(names)


def test_frozen_test_pooled_metric_matches_global_sample_definition():
    targets = {
        "S": [1.0, 2.0, 3.0],
        "E": [10.0, 12.0],
    }
    predictions = {
        "S": [1.0, 1.0, 4.0],
        "E": [9.0, 13.0],
    }
    observed = pooled_metrics(targets, predictions)
    truth = [1.0, 2.0, 3.0, 10.0, 12.0]
    predicted = [1.0, 1.0, 4.0, 9.0, 13.0]
    mean_truth = sum(truth) / len(truth)
    sse = sum((a - b) ** 2 for a, b in zip(truth, predicted))
    sst = sum((a - mean_truth) ** 2 for a in truth)
    assert math.isclose(observed["r2"], 1.0 - sse / sst)
    assert math.isclose(observed["rmse"], math.sqrt(sse / len(truth)))
    assert math.isclose(observed["mae"], 4.0 / 5.0)
    assert observed["rows"] == 5


def test_epsilon_parsimony_remains_optional_without_strict_joint_pass():
    evaluations = {
        "expr": {
            "fitness": 1.75,
            "enhanced_physics": {"joint_pass": False},
        }
    }
    called = []

    def inherited(_evaluations):
        called.append(True)
        raise ValueError("no physically passing evaluations")

    observed = _diagnostic_epsilon_parsimony(evaluations, inherited)
    assert called == []
    assert observed["status"] == "unavailable_no_strict_joint_pass"
    assert observed["diagnostic_only_not_used_for_evolution"] is True
    assert observed["selected_expression"] is None


def test_epsilon_parsimony_delegates_when_strict_subset_exists():
    evaluations = {
        "expr": {
            "fitness": 1.8,
            "enhanced_physics": {"joint_pass": True},
        }
    }
    expected = {
        "diagnostic_only_not_used_for_evolution": True,
        "selected_expression": "expr",
    }
    assert _diagnostic_epsilon_parsimony(
        evaluations, lambda supplied: expected if supplied is evaluations else None
    ) == expected


def test_fixed17_ranking_summary_matches_precommitted_hash():
    from methods.cosydelay.engine.data_protocol.run_formal_training_search import (
        sha256_file,
    )

    path = DEFAULT_BASELINE_DIR / "raw16_per_intersection_method_metrics.csv"
    assert path.is_file()
    assert sha256_file(path) == FROZEN_FIXED17_SUMMARY_SHA256


def test_formal_integrity_check_rejects_any_test_selection_flag():
    protocol = {
        "policy": V16_POLICY.to_dict(),
        "method_contract": V16_CONTRACT.to_dict(),
        "source_sha256": {"f.py": "abc"},
        "training_sha256": {"1": {"sha256": "train-hash"}},
    }
    result = {
        "status": "completed_v16_p10g10_training_only_search",
        "not_a_formal_result": False,
        "formal_protocol_complete": True,
        "method_id": V16_POLICY.method_id,
        "intersection_id": 1,
        "population": 10,
        "generations": 10,
        "policy": V16_POLICY.to_dict(),
        "method_contract": V16_CONTRACT.to_dict(),
        "train_file_sha256": "train-hash",
        "selection_source": "full_training_evolution_only",
        "accessed_splits": ["train"],
        "outer_validation_accessed": False,
        "test_file_opened": False,
        "validation_or_test_used_for_selection": False,
        "post_evolution_cv_reranking": False,
        "post_evolution_refit": False,
        "external_incumbent_allowed": False,
        "initial_population_all_generated_in_current_run": True,
        "completed_generations": 10,
        "legacy_numeric_r9_probe_executed": False,
        "candidate_evaluations": 110,
        "v16_source_sha256": {"f.py": "abc"},
        "selection_order": [
            "paper_training_fitness",
            "raw_training_r2_on_exact_fitness_ties",
            "negative_training_rmse_on_remaining_ties",
        ],
        "population_update": V16_CONTRACT.population_update,
        "strict_joint_pass_role": "diagnostic_only_not_a_hard_gate",
        "paper_fitness_definition_changed": False,
        "selected_enhanced_physics": _valid_formal_physics_payload(),
        "training_metrics": {"macro_nonnegative_r2": 0.8},
        "training_fitness": 1.8,
        "prompt_audit": {
            "successful_prompt_calls": 10,
            "mutation_prompt_calls": 9,
            "single_physical_contract_per_prompt": True,
            "single_output_schema_per_prompt": True,
            "output_schema_is_last_in_every_prompt": True,
            "mutation_has_parent": True,
            "training_metric_feedback_absent": True,
            "finite_nonnegative_low_demand_present": True,
            "exact_zero_flow_absent": True,
            "numeric_r9_threshold_absent": True,
            "approach_supervision_disclosed": True,
            "llm_does_not_fit_coefficients": True,
        },
        "llm_runtime": {
            "content_archive_audit": {
                "status": "pass",
                "instantiated_prompt_content_archived": True,
                "normalized_assistant_response_content_archived": True,
                "prompt_and_response_sha256_verified": True,
                "api_key_or_authorization_fields_present": False,
                "model_identifier_present_in_every_successful_record": True,
                "exact_snapshot_inference_from_alias_permitted": False,
                "requested_model_aliases": ["gpt-4.1-mini"],
                "returned_model_identifiers": ["gpt-4.1-mini"],
            }
        },
        "optimizer_restart_composition_audit": {
            "status": "pass",
            "restarts_per_approach_block": 10,
            "parent_warm_start_replaces_one_local_slot": True,
            "external_warm_start_allowed": False,
        },
        "epsilon_parsimony_diagnostic": {
            "diagnostic_only_not_used_for_evolution": True,
        },
        "expression": "a1*Cycle_Time*flow_lane/GR_phase",
        "lane_parameters": _valid_formal_parameters(),
    }
    validate_formal_result(result, protocol, intersection=1)
    result["test_file_opened"] = True
    try:
        validate_formal_result(result, protocol, intersection=1)
    except RuntimeError as exc:
        assert "test_file_opened" in str(exc)
    else:
        raise AssertionError("Test access must invalidate a formal V16 search")


def test_formal_integrity_check_rejects_fitness_prompt_or_archive_drift():
    protocol = {
        "policy": V16_POLICY.to_dict(),
        "method_contract": V16_CONTRACT.to_dict(),
        "source_sha256": {"f.py": "abc"},
        "training_sha256": {"1": {"sha256": "train-hash"}},
    }
    base_result = {
        "status": "completed_v16_p10g10_training_only_search",
        "not_a_formal_result": False,
        "formal_protocol_complete": True,
        "method_id": V16_POLICY.method_id,
        "intersection_id": 1,
        "population": 10,
        "generations": 10,
        "policy": V16_POLICY.to_dict(),
        "method_contract": V16_CONTRACT.to_dict(),
        "train_file_sha256": "train-hash",
        "selection_source": "full_training_evolution_only",
        "accessed_splits": ["train"],
        "outer_validation_accessed": False,
        "test_file_opened": False,
        "validation_or_test_used_for_selection": False,
        "post_evolution_cv_reranking": False,
        "post_evolution_refit": False,
        "external_incumbent_allowed": False,
        "initial_population_all_generated_in_current_run": True,
        "completed_generations": 10,
        "legacy_numeric_r9_probe_executed": False,
        "candidate_evaluations": 110,
        "v16_source_sha256": {"f.py": "abc"},
        "selection_order": [
            "paper_training_fitness",
            "raw_training_r2_on_exact_fitness_ties",
            "negative_training_rmse_on_remaining_ties",
        ],
        "population_update": V16_CONTRACT.population_update,
        "strict_joint_pass_role": "diagnostic_only_not_a_hard_gate",
        "paper_fitness_definition_changed": False,
        "selected_enhanced_physics": _valid_formal_physics_payload(),
        "training_metrics": {"macro_nonnegative_r2": 0.8},
        "training_fitness": 1.8,
        "prompt_audit": {
            "successful_prompt_calls": 10,
            "mutation_prompt_calls": 9,
            "single_physical_contract_per_prompt": True,
            "single_output_schema_per_prompt": True,
            "output_schema_is_last_in_every_prompt": True,
            "mutation_has_parent": True,
            "training_metric_feedback_absent": True,
            "finite_nonnegative_low_demand_present": True,
            "exact_zero_flow_absent": True,
            "numeric_r9_threshold_absent": True,
            "approach_supervision_disclosed": True,
            "llm_does_not_fit_coefficients": True,
        },
        "llm_runtime": {
            "content_archive_audit": {
                "status": "pass",
                "instantiated_prompt_content_archived": True,
                "normalized_assistant_response_content_archived": True,
                "prompt_and_response_sha256_verified": True,
                "api_key_or_authorization_fields_present": False,
                "model_identifier_present_in_every_successful_record": True,
                "exact_snapshot_inference_from_alias_permitted": False,
                "requested_model_aliases": ["gpt-4.1-mini"],
                "returned_model_identifiers": ["gpt-4.1-mini"],
            }
        },
        "optimizer_restart_composition_audit": {
            "status": "pass",
            "restarts_per_approach_block": 10,
            "parent_warm_start_replaces_one_local_slot": True,
            "external_warm_start_allowed": False,
        },
        "epsilon_parsimony_diagnostic": {
            "diagnostic_only_not_used_for_evolution": True,
        },
        "expression": "a1*Cycle_Time*flow_lane/GR_phase",
        "lane_parameters": _valid_formal_parameters(),
    }
    for mutate, expected_text in (
        (lambda item: item.update(training_fitness=1.7), "training_fitness_contract"),
        (
            lambda item: item["prompt_audit"].update(
                output_schema_is_last_in_every_prompt=False
            ),
            "prompt_audit",
        ),
        (
            lambda item: item["llm_runtime"]["content_archive_audit"].update(
                status="fail"
            ),
            "content_archive_audit",
        ),
    ):
        candidate = json.loads(json.dumps(base_result))
        mutate(candidate)
        try:
            validate_formal_result(candidate, protocol, intersection=1)
        except RuntimeError as exc:
            assert expected_text in str(exc)
        else:
            raise AssertionError(f"formal validator accepted drift: {expected_text}")


def test_formal_integrity_survives_json_roundtrip_and_rejects_physics_tampering():
    protocol = {
        "policy": V16_POLICY.to_dict(),
        "method_contract": V16_CONTRACT.to_dict(),
        "source_sha256": {"f.py": "abc"},
        "training_sha256": {"1": {"sha256": "train-hash"}},
    }
    result = {
        "status": "completed_v16_p10g10_training_only_search",
        "not_a_formal_result": False,
        "formal_protocol_complete": True,
        "method_id": V16_POLICY.method_id,
        "intersection_id": 1,
        "population": 10,
        "generations": 10,
        "policy": V16_POLICY.to_dict(),
        "method_contract": V16_CONTRACT.to_dict(),
        "train_file_sha256": "train-hash",
        "selection_source": "full_training_evolution_only",
        "selection_order": [
            "paper_training_fitness",
            "raw_training_r2_on_exact_fitness_ties",
            "negative_training_rmse_on_remaining_ties",
        ],
        "population_update": V16_CONTRACT.population_update,
        "strict_joint_pass_role": "diagnostic_only_not_a_hard_gate",
        "paper_fitness_definition_changed": False,
        "accessed_splits": ["train"],
        "outer_validation_accessed": False,
        "test_file_opened": False,
        "validation_or_test_used_for_selection": False,
        "post_evolution_cv_reranking": False,
        "post_evolution_refit": False,
        "external_incumbent_allowed": False,
        "initial_population_all_generated_in_current_run": True,
        "completed_generations": 10,
        "legacy_numeric_r9_probe_executed": False,
        "candidate_evaluations": 110,
        "v16_source_sha256": {"f.py": "abc"},
        "selected_enhanced_physics": _valid_formal_physics_payload(),
        "training_metrics": {"macro_nonnegative_r2": 0.8},
        "training_fitness": 1.8,
        "prompt_audit": {
            "successful_prompt_calls": 10,
            "mutation_prompt_calls": 9,
            "single_physical_contract_per_prompt": True,
            "single_output_schema_per_prompt": True,
            "output_schema_is_last_in_every_prompt": True,
            "mutation_has_parent": True,
            "training_metric_feedback_absent": True,
            "finite_nonnegative_low_demand_present": True,
            "exact_zero_flow_absent": True,
            "numeric_r9_threshold_absent": True,
            "approach_supervision_disclosed": True,
            "llm_does_not_fit_coefficients": True,
        },
        "llm_runtime": {
            "content_archive_audit": {
                "status": "pass",
                "instantiated_prompt_content_archived": True,
                "normalized_assistant_response_content_archived": True,
                "prompt_and_response_sha256_verified": True,
                "api_key_or_authorization_fields_present": False,
                "model_identifier_present_in_every_successful_record": True,
                "exact_snapshot_inference_from_alias_permitted": False,
                "requested_model_aliases": ["gpt-4.1-mini"],
                "returned_model_identifiers": ["gpt-4.1-mini"],
            }
        },
        "optimizer_restart_composition_audit": {
            "status": "pass",
            "restarts_per_approach_block": 10,
            "parent_warm_start_replaces_one_local_slot": True,
            "external_warm_start_allowed": False,
        },
        "epsilon_parsimony_diagnostic": {
            "diagnostic_only_not_used_for_evolution": True,
        },
        "expression": "a1*Cycle_Time*flow_lane/GR_phase",
        "lane_parameters": _valid_formal_parameters(),
    }
    protocol = json.loads(json.dumps(protocol))
    result = json.loads(json.dumps(result))
    validate_formal_result(result, protocol, intersection=1)

    tampered = json.loads(json.dumps(result))
    tampered["selected_enhanced_physics"]["lane_rule_scores"]["S_T"][
        "R6_nonnegative_delay"
    ] = 0.5
    try:
        validate_formal_result(tampered, protocol, intersection=1)
    except RuntimeError as exc:
        assert "selected_enhanced_physics.recomputation" in str(exc)
    else:
        raise AssertionError("tampered movement physical cell was accepted")

    tampered = json.loads(json.dumps(result))
    tampered["lane_parameters"]["S_T"]["a1"] = 0.0
    try:
        validate_formal_result(tampered, protocol, intersection=1)
    except RuntimeError as exc:
        assert "nonpositive or nonfinite fitted coefficient" in str(exc)
    else:
        raise AssertionError("nonpositive fitted coefficient was accepted")

    tampered = json.loads(json.dumps(result))
    tampered["lane_parameters"].pop("W_R")
    tampered["selected_enhanced_physics"]["lane_rule_scores"].pop("W_R")
    try:
        validate_formal_result(tampered, protocol, intersection=1)
    except RuntimeError as exc:
        assert "intersection movement layout mismatch" in str(exc)
    else:
        raise AssertionError("jointly omitted movement was accepted")


def test_frozen_contract_matches_executable_policy_and_hidden_limits():
    validate_v16_contract()
    assert V16_CONTRACT.population == V16_CONTRACT.generations == 10
    assert V16_CONTRACT.optimizer_restarts == 10
    assert V16_CONTRACT.optimizer_maxcor == 10
    assert V16_CONTRACT.optimizer_ftol == 2.220446049250313e-9
    assert V16_CONTRACT.optimizer_gtol == 1e-5
    assert V16_CONTRACT.optimizer_maxls == 20
    assert V16_CONTRACT.parent_warm_start_replaces_a_restart_not_added
    assert not V16_CONTRACT.external_warm_start_allowed
    assert V16_CONTRACT.mutation_receives_compact_parent_seven_rule_feedback
    assert not V16_CONTRACT.legacy_eight_rule_targeted_feedback_augmenter_enabled
    assert not V16_POLICY.targeted_physical_feedback
    assert V16_CONTRACT.maximum_coefficients == 8
    assert V16_CONTRACT.maximum_symbolic_nodes == 90
    assert V16_CONTRACT.maximum_expression_characters == 2000
    assert V16_CONTRACT.invalid_outputs_per_generation_batch == 5
    assert V16_CONTRACT.generation_batches_per_population_slot == 30
    assert V16_CONTRACT.transport_attempts_per_output_attempt == 5
    assert V16_CONTRACT.maximum_duplicate_examples_in_prompt == 3
    assert V16_CONTRACT.schema_version == 10
    assert V16_POLICY.execution_contract_version.endswith("_v10")
    assert V16_POLICY.prompt_contract_version == PROMPT_CONTRACT_ID
    assert V16_CONTRACT.coefficient_identifiers_consecutive_from_a1
    assert not V16_CONTRACT.prefit_sampled_screen_is_global_domain_proof
    assert not V16_CONTRACT.mixed_nonlinear_coefficient_roles_allowed
    assert V16_CONTRACT.symbolic_limit_operation == (
        "sympy.limit_in_isolated_process"
    )
    assert V16_CONTRACT.physical_score_reused_as_enhanced_audit
    assert V16_CONTRACT.temperature is None
    assert V16_CONTRACT.top_p is None
    assert V16_CONTRACT.maximum_output_tokens is None
    assert not V16_CONTRACT.provider_seed_support_verified
    assert V16_MAX_PROMPT_EXCLUSIONS == 3


def test_parameter_boundary_audit_uses_log_optimizer_coordinates():
    from .diagnostics import audit_parameter_quality_log_coordinates

    observed = audit_parameter_quality_log_coordinates(
        "Cycle_Time * (a1 + a2 * flow_lane / GR_phase)",
        {"S_T": {"a1": 0.1, "a2": 1000.0}},
        {
            "scale": (0.001, 1000.0),
            "power_exponent": (0.05, 5.0),
            "exp_coefficient": (0.0001, 1.0),
        },
    )
    records = {item["coefficient"]: item for item in observed["boundary_records"]}
    # 0.1 is close in a raw linear interval but one third through this log span.
    assert "a1" not in records
    assert records["a2"]["at_upper_bound"]
    assert observed["coordinate_system"] == (
        "normalized_natural_log_bound_interval"
    )
    assert observed["diagnostic_only_not_used_for_selection"]


def test_v16_fitter_freezes_former_scipy_default_options():
    from . import fitter as v16_fitter

    captured = {}

    class Result:
        pass

    def fake_minimize(*args, **kwargs):
        del args
        captured.update(kwargs["options"])
        return Result()

    previous = v16_fitter._ORIGINAL_MINIMIZE
    v16_fitter._ORIGINAL_MINIMIZE = fake_minimize
    try:
        result = v16_fitter._minimize_with_frozen_lbfgsb_defaults(
            lambda value: value,
            [0.0],
            method="L-BFGS-B",
            options={"maxiter": 200, "maxfun": 20_000},
        )
    finally:
        v16_fitter._ORIGINAL_MINIMIZE = previous
    assert isinstance(result, Result)
    assert captured["maxcor"] == 10
    assert captured["ftol"] == 2.220446049250313e-9
    assert captured["gtol"] == 1e-5
    assert captured["maxls"] == 20


def test_restart_composition_audit_accepts_declared_initial_and_parent_starts():
    history = []
    initial = ["official_local_replay"] * 9 + ["sobol_role_wide"]
    mutation = ["parent_warm"] + ["official_local_replay"] * 8 + [
        "sobol_role_wide"
    ]
    candidate_id = 0
    for generation in range(11):
        for _ in range(10):
            candidate_id += 1
            kinds = initial if generation == 0 else mutation
            history.append(
                {
                    "event": "evaluated",
                    "candidate_id": candidate_id,
                    "generation": generation,
                    "evaluation_details": {
                        "fit": {
                            "n_restarts": 10,
                            "parent_warm_start_available": generation > 0,
                            "approaches": [
                                {
                                    "approach": "S",
                                    "restarts": [
                                        {"start_kind": kind} for kind in kinds
                                    ],
                                }
                            ],
                        }
                    },
                }
            )
    observed = _restart_composition_audit(history)
    assert observed["status"] == "pass"
    assert observed["fitted_candidates"] == 110
    assert observed["initialization_blocks"] == 10
    assert observed["warm_mutation_blocks"] == 100


def test_restart_composition_audit_supports_frozen_p3g2_sensitivity_budget():
    history = []
    initial = ["official_local_replay"] * 9 + ["sobol_role_wide"]
    mutation = ["parent_warm"] + ["official_local_replay"] * 8 + [
        "sobol_role_wide"
    ]
    for candidate in range(9):
        generation = candidate // 3
        kinds = initial if generation == 0 else mutation
        history.append(
            {
                "event": "evaluated",
                "candidate_id": candidate + 1,
                "generation": generation,
                "evaluation_details": {
                    "fit": {
                        "n_restarts": 10,
                        "parent_warm_start_available": generation > 0,
                        "approaches": [
                            {
                                "approach": "S",
                                "restarts": [
                                    {"start_kind": kind} for kind in kinds
                                ],
                            }
                        ],
                    }
                },
            }
        )
    observed = _restart_composition_audit(history, population=3, generations=2)
    assert observed["status"] == "pass"
    assert observed["population"] == 3
    assert observed["generations"] == 2
    assert observed["fitted_candidates"] == 9


def test_restart_composition_audit_rejects_an_eleventh_start():
    initial = ["official_local_replay"] * 9 + ["sobol_role_wide"]
    history = []
    for candidate in range(110):
        generation = candidate // 10
        kinds = (
            initial
            if generation == 0
            else ["parent_warm"]
            + ["official_local_replay"] * 8
            + ["sobol_role_wide"]
        )
        if candidate == 109:
            kinds = kinds + ["sobol_role_wide"]
        history.append(
            {
                "event": "evaluated",
                "candidate_id": candidate + 1,
                "generation": generation,
                "evaluation_details": {
                    "fit": {
                        "n_restarts": len(kinds),
                        "parent_warm_start_available": generation > 0,
                        "approaches": [
                            {
                                "approach": "S",
                                "restarts": [
                                    {"start_kind": kind} for kind in kinds
                                ],
                            }
                        ],
                    }
                },
            }
        )
    try:
        _restart_composition_audit(history)
    except RuntimeError as exc:
        assert "restart composition drift" in str(exc)
    else:
        raise AssertionError("an eleventh optimizer start must be rejected")


def test_llm_content_archive_requires_matching_prompt_and_response_hashes():
    import hashlib

    prompt = "instantiated prompt"
    response = "raw provider response"
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    record = {
        "status": "success",
        "model": "gpt-4.1-mini",
        "response_model": "gpt-4.1-mini",
        "prompt": prompt,
        "prompt_sha256": digest(prompt),
        "response": response,
        "response_sha256": digest(response),
    }
    observed = _audit_content_integrity([record], secret="secret-key")
    assert observed["status"] == "pass"
    assert observed["successful_records"] == 1
    assert observed["requested_model_aliases"] == ["gpt-4.1-mini"]
    assert observed["returned_model_identifiers"] == ["gpt-4.1-mini"]

    broken = dict(record, response_sha256="0" * 64)
    try:
        _audit_content_integrity([broken])
    except RuntimeError as exc:
        assert "incomplete or inconsistent" in str(exc)
    else:
        raise AssertionError("a mismatched response hash must fail closed")

    missing_returned_model = dict(record)
    missing_returned_model.pop("response_model")
    try:
        _audit_content_integrity([missing_returned_model])
    except RuntimeError as exc:
        assert "missing_response_model" in str(exc)
    else:
        raise AssertionError("missing returned model metadata must fail closed")

    leaked = dict(record, response="secret-key")
    leaked["response_sha256"] = digest(leaked["response"])
    try:
        _audit_content_integrity([leaked], secret="secret-key")
    except RuntimeError as exc:
        assert "unredacted_api_key_content" in str(exc)
    else:
        raise AssertionError("unredacted API-key content must fail closed")


def test_pilot_wrapper_records_principlewise_mode_and_disables_old_feedback():
    captured = {}

    def original(*args, **kwargs):
        captured.update(kwargs)
        return args, kwargs

    _force_principlewise_evolution(original)(
        "data",
        score_mode="binary",
        targeted_physical_feedback=True,
    )
    assert captured["score_mode"] == "principlewise"
    assert not captured["targeted_physical_feedback"]


def test_offline_smoke_profiles_have_declared_physical_outcomes():
    assert set(EXPRESSION_PROFILES) == {
        "positive_low_demand",
        "finite_zero_green",
    }
    assert EXPRESSION_PROFILES["positive_low_demand"]["expected_failed_rules"] == ()
    assert EXPRESSION_PROFILES["finite_zero_green"]["expected_failed_rules"] == (
        "R7_zero_green_limit",
    )
