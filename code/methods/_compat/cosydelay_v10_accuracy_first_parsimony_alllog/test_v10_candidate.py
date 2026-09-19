from __future__ import annotations

from unittest.mock import patch

from expression_rules import parse_symbolic_expression
from methods.cosydelay_lbfgsb_r10_parallel4_v3.physics_audit import (
    _fixed_inverse_green_certificate,
    audit_fitted_physics,
    audit_search_physics,
)
from methods.prospective_optimizer_conditioning_v1.complexity_prompt import (
    COMPLEXITY_PROMPT_NOTE,
)

from .final_prompt import (
    FINAL_PROMPT_CONTRACT_ID,
    MAX_NOVELTY_EXAMPLES_IN_PROMPT,
    build_final_init_prompt,
    build_final_mutation_prompt,
    format_parent_training_feedback,
)
from .policy import V10_POLICY


def test_v10_policy_retains_paper_budget_and_changes_only_declared_axes() -> None:
    assert V10_POLICY.population == 10
    assert V10_POLICY.generations == 10
    assert V10_POLICY.optimizer_restarts == 10
    assert V10_POLICY.optimizer_maxiter == 200
    assert V10_POLICY.optimizer_maxfun == 20_000
    assert V10_POLICY.approach_workers_cap == 4
    assert V10_POLICY.structural_diversity_mode == "family_unique"
    assert V10_POLICY.validation_or_test_used_for_selection is False
    assert V10_POLICY.external_incumbent_allowed is False
    assert V10_POLICY.fixed_r9_guard is False
    assert V10_POLICY.reuse_standard_fitted_physics_audit is True
    assert V10_POLICY.reuse_prefit_symbolic_endpoint_audit is True
    assert V10_POLICY.reject_fitted_structural_families is True


def test_fixed_inverse_green_fast_certificate_is_conservative() -> None:
    certified = (
        "Cycle_Time*flow_lane/GR_phase*"
        "(a1+a2*flow_lane**a3+a4*exp(-a5*GR_phase))"
    )
    parsed, symbols = parse_symbolic_expression(certified)
    passed, endpoint = _fixed_inverse_green_certificate(parsed, symbols)
    assert passed is True
    assert endpoint

    extra_singularity = (
        "Cycle_Time*flow_lane/GR_phase*"
        "(a1+a2*log(1+a3/GR_phase))"
    )
    parsed, symbols = parse_symbolic_expression(extra_singularity)
    passed, endpoint = _fixed_inverse_green_certificate(parsed, symbols)
    assert passed is False
    assert endpoint is None


def test_fitted_audit_reuses_standard_and_prefit_symbolic_decisions() -> None:
    expression = "Cycle_Time*a1*flow_lane/GR_phase"
    symbolic = audit_search_physics(expression, standard_joint_pass=True)
    rule_scores = {
        name: 1.0
        for name in (
            "R1_required_variables",
            "R2_nondecreasing_flow",
            "R3_nonincreasing_green",
            "R4_time_dimension",
            "R6_nonnegative_delay",
            "R7_operational_responsiveness",
            "R8_zero_flow_boundary",
            "R9_zero_green_limit",
        )
    }
    with patch(
        "methods.cosydelay_lbfgsb_r10_parallel4_v3.physics_audit."
        "score_fitted_lanes_principlewise",
        side_effect=AssertionError("standard audit must not be repeated"),
    ):
        result = audit_fitted_physics(
            expression,
            {"E_T": {"a1": 1.0}},
            ["E_T"],
            precomputed_standard={
                "joint_pass": True,
                "rule_scores": rule_scores,
            },
            precomputed_symbolic=symbolic,
        )
    assert result["joint_pass"] is True
    assert result["standard_audit_reused"] is True
    assert result["symbolic_endpoint_audit_reused"] is True
    assert result["symbolic_r9_method"] == (
        "fixed_inverse_green_finite_positive_h_certificate"
    )


def test_parsimony_prompt_has_no_required_coefficient_count() -> None:
    lowered = COMPLEXITY_PROMPT_NOTE.lower()
    assert "do not target a predetermined coefficient count" in lowered
    assert "a1 through a8" in lowered
    assert "large mutation" in lowered
    assert "automatically appending" in lowered


def test_final_initialization_prompt_is_single_clean_contract() -> None:
    prompt = build_final_init_prompt(
        {}, ["flow_lane", "GR_phase", "Cycle_Time"], intersection_id=1
    )
    lowered = prompt.lower()
    assert FINAL_PROMPT_CONTRACT_ID == V10_POLICY.prompt_contract_version
    assert "parent expression" not in lowered
    assert "do not target a predetermined coefficient count" in lowered
    assert "3--6" not in prompt
    assert "3-6" not in prompt
    assert prompt.count("PHYSICAL AND STRUCTURAL REQUIREMENTS:") == 1
    assert prompt.count("### Expression") == 1
    assert "y(0, GR_phase, Cycle_Time)=0 exactly" in prompt
    assert "exp(-a1*GR_phase)" in prompt
    assert "H must not introduce another zero-green singularity" in prompt
    assert "sharper growth toward high demand" in prompt


def test_final_mutation_prompt_contains_parent_and_training_feedback_once() -> None:
    feedback = format_parent_training_feedback(
        {
            "fitness": 1.72,
            "metrics": {
                "macro_nonnegative_r2": 0.72,
                "pooled_rmse": 5.4,
                "pooled_mae": 3.8,
            },
            "enhanced_physics": {"joint_pass": True},
        },
        current_best_fitness=1.80,
    )
    prompt = build_final_mutation_prompt(
        "Cycle_Time*a1*flow_lane/GR_phase",
        "",
        "",
        (True, "All physical rules passed"),
        "large",
        ["flow_lane", "GR_phase", "Cycle_Time"],
        search_feedback=feedback,
    )
    assert MAX_NOVELTY_EXAMPLES_IN_PROMPT == 10
    assert prompt.count("PARENT EXPRESSION:") == 1
    assert prompt.count("PARENT TRAINING EVALUATION (Training only):") == 1
    assert "Paper Fitness: 1.720000" in prompt
    assert "Pooled RMSE: 5.400000" in prompt
    assert "Current best Training Fitness" in prompt
    assert "Parent Fitness gap from current best: 0.080000" in prompt
    assert "Validation and Test metrics are not available" in prompt
    assert prompt.count("PHYSICAL AND STRUCTURAL REQUIREMENTS:") == 1
    assert prompt.count("### Expression") == 1
