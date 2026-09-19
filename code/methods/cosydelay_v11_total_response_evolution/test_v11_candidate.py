"""Focused tests for the V11 search repair."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .final_prompt import (
    build_init_prompt,
    build_mutation_prompt,
    format_parent_training_feedback,
    install_prompt_contract,
)
from .policy import V11_POLICY


def test_policy_restores_canonical_evolution_without_family_blacklist():
    assert V11_POLICY.structural_diversity_mode == "canonical"
    assert not V11_POLICY.reject_fitted_structural_families
    assert V11_POLICY.optimizer_restarts == 10
    assert V11_POLICY.external_incumbent_allowed is False
    assert V11_POLICY.validation_or_test_used_for_selection is False


def test_prompt_applies_monotonicity_to_complete_response():
    prompt = build_init_prompt({}, [], intersection_id=1)
    assert "Apply monotonicity to the COMPLETE delay response" in prompt
    assert "flow_lane*H must be nondecreasing" in prompt
    assert "H/GR_phase must be nonincreasing" in prompt
    assert "H itself may increase or decrease" in prompt
    assert "1/(1+a1*flow_lane)" in prompt
    assert "1/(1+a1/GR_phase)" in prompt
    assert "Validation" not in prompt
    assert "Test metrics" not in prompt


def test_compensated_saturation_is_monotone_as_a_complete_response():
    flow = np.linspace(0.001, 1.2, 500)
    green = np.linspace(0.04, 0.85, 500)
    a1, a2, a3, a4, a5 = 0.4, 1.7, 2.3, 0.8, 0.2
    demand = flow * (a1 + a2 / (1.0 + a3 * flow))
    green_response = a1 / green + a4 / (green + a5)
    assert np.all(np.diff(demand) >= -1e-12)
    assert np.all(np.diff(green_response) <= 1e-12)


def test_training_feedback_reports_each_approach_and_weakest():
    record = {
        "fitness": 1.75,
        "metrics": {
            "macro_nonnegative_r2": 0.75,
            "pooled_rmse": 5.0,
            "pooled_mae": 3.5,
            "by_approach": {
                "S": {"r2": 0.9, "rmse": 3.0, "mae": 2.0},
                "E": {"r2": 0.5, "rmse": 7.0, "mae": 5.0},
                "N": {"r2": 0.88, "rmse": 3.2, "mae": 2.2},
                "W": {"r2": 0.72, "rmse": 6.0, "mae": 4.8},
            },
        },
        "enhanced_physics": {"joint_pass": True, "errors": []},
    }
    feedback = format_parent_training_feedback(
        record, current_best_fitness=1.80
    )
    for approach in ("S", "E", "N", "W"):
        assert f"Approach {approach} Training" in feedback
    assert "Weakest Training approach: E" in feedback
    assert "Validation and Test metrics are not available" in feedback

    mutation = build_mutation_prompt(
        "Cycle_Time*flow_lane/GR_phase*a1",
        "",
        "",
        (True, "all rules pass"),
        "small",
        [],
        search_feedback=feedback,
    )
    assert "PARENT EXPRESSION:" in mutation
    assert "Weakest Training approach: E" in mutation
    assert mutation.count("PHYSICAL AND STRUCTURAL REQUIREMENTS:") == 1


def test_prompt_installation_is_scoped_and_reversible():
    original_init = lambda *args, **kwargs: "old init"
    original_mutation = lambda *args, **kwargs: "old mutation"
    module = SimpleNamespace(
        build_universal_lane_init_prompt=original_init,
        build_universal_lane_mutation_prompt=original_mutation,
    )
    with install_prompt_contract(adaptation_module=module):
        assert module.build_universal_lane_init_prompt is build_init_prompt
        assert module.build_universal_lane_mutation_prompt is build_mutation_prompt
    assert module.build_universal_lane_init_prompt is original_init
    assert module.build_universal_lane_mutation_prompt is original_mutation
