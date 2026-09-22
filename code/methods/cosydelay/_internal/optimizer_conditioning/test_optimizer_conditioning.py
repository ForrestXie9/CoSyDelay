"""Focused tests for the prospective optimizer-conditioning components."""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from methods.cosydelay._internal.optimizer_conditioning import all_log_fitter
from methods.cosydelay._internal.optimizer_conditioning.all_log_fitter import (
    _from_log_space,
    _mse_and_all_log_gradient,
    _to_log_space,
)
from methods.cosydelay._internal.optimizer_conditioning.role_policy import (
    CONDITIONED_ROLE_BOUNDS,
    coefficient_bounds_for_expression,
    nonlinear_role_conflicts,
)
from methods.cosydelay._internal.optimizer_conditioning.coordinate_portfolio_fitter import (
    restart_parameterizations,
)
from methods.cosydelay._internal.optimizer_conditioning.complexity_prompt import (
    COMPLEXITY_PROMPT_NOTE,
    expression_complexity,
    install_accuracy_first_parsimony_contract,
    select_epsilon_parsimonious,
)


def test_positive_log_roundtrip() -> None:
    values = np.asarray([0.0001, 0.1, 1.0, 1000.0])
    np.testing.assert_allclose(_from_log_space(_to_log_space(values)), values)


def test_log_space_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        _to_log_space(np.asarray([1.0, 0.0]))


def test_all_log_gradient_uses_chain_rule(monkeypatch) -> None:
    raw = np.asarray([0.25, 4.0])
    raw_gradient = np.asarray([2.0, -3.0])

    def fake_raw_objective(values, payload):
        np.testing.assert_allclose(values, raw)
        assert payload == {"marker": True}
        return 7.5, raw_gradient

    monkeypatch.setattr(
        all_log_fitter, "_mse_and_gradient_quiet", fake_raw_objective
    )
    objective, gradient = _mse_and_all_log_gradient(
        np.log(raw), {"marker": True}
    )
    assert objective == 7.5
    np.testing.assert_allclose(gradient, raw_gradient * raw)


def test_frozen_coordinate_portfolio_has_three_raw_and_seven_log() -> None:
    parameterizations = restart_parameterizations(10)
    assert [i for i, value in enumerate(parameterizations) if value == "raw"] == [
        0,
        3,
        6,
    ]
    assert parameterizations.count("all_positive_log") == 7
    with pytest.raises(ValueError, match="exactly ten"):
        restart_parameterizations(9)


def test_complexity_audit_does_not_prescribe_a_coefficient_count() -> None:
    simple = expression_complexity("Cycle_Time*flow_lane/GR_phase*a1")
    larger = expression_complexity(
        "Cycle_Time*flow_lane/GR_phase*(a1+a2*flow_lane+a3*GR_phase)"
    )
    assert simple["coefficient_count"] == 1
    assert larger["coefficient_count"] == 3
    assert larger["sympy_operations"] > simple["sympy_operations"]


def test_parsimony_contract_is_scoped_and_conflict_is_diagnostic_only() -> None:
    module = SimpleNamespace(
        PHYSICAL_REQUIREMENTS_STANDARD="standard",
        PHYSICAL_REQUIREMENTS_COMPACT="compact",
        INTERPRETABILITY_AND_PARSIMONY_RULES="parsimony",
        validate_candidate_expression=lambda expression: (True, "upstream pass"),
    )
    conflicted = "a1+a2+a3+a4+flow_lane**a4+exp(a4*GR_phase)"
    with install_accuracy_first_parsimony_contract(
        adaptation_module=module
    ) as audit:
        assert COMPLEXITY_PROMPT_NOTE in module.PHYSICAL_REQUIREMENTS_STANDARD
        passed, _ = module.validate_candidate_expression(conflicted)
        assert passed is True
        assert audit[-1]["nonlinear_role_conflicts_diagnostic_only"] == {
            "a4": ["exp_coefficient", "power_exponent"]
        }
        passed, reason = module.validate_candidate_expression("a1+a2+a3")
        assert passed is True
        assert reason == "upstream pass"
    assert module.PHYSICAL_REQUIREMENTS_STANDARD == "standard"
    assert module.validate_candidate_expression("a1")[0] is True


def test_epsilon_parsimony_is_diagnostic_and_accuracy_bounded() -> None:
    evaluations = {
        "complex": {
            "expression": "Cycle_Time*flow_lane/GR_phase*(a1+a2*flow_lane)",
            "fitness": 1.8000,
            "enhanced_physics": {"joint_pass": True},
        },
        "simple": {
            "expression": "Cycle_Time*flow_lane/GR_phase*a1",
            "fitness": 1.7990,
            "enhanced_physics": {"joint_pass": True},
        },
        "too_far": {
            "expression": "Cycle_Time*flow_lane/GR_phase*a1",
            "fitness": 1.7900,
            "enhanced_physics": {"joint_pass": True},
        },
    }
    selected = select_epsilon_parsimonious(evaluations, fitness_tolerance=0.002)
    assert selected["selected_expression"].endswith("*a1")
    assert selected["fitness_gap_from_best"] == pytest.approx(0.001)
    assert selected["diagnostic_only_not_used_for_evolution"] is True


def test_mixed_nonlinear_role_is_detected_and_rejected() -> None:
    expression = (
        "Cycle_Time*flow_lane/GR_phase*"
        "(a1*(1+flow_lane**a2)+a3*exp(a2*GR_phase))"
    )
    assert nonlinear_role_conflicts(expression) == {
        "a2": ["exp_coefficient", "power_exponent"]
    }
    with pytest.raises(ValueError, match="mixed nonlinear"):
        coefficient_bounds_for_expression(
            expression, CONDITIONED_ROLE_BOUNDS
        )


def test_conditioned_role_bounds_are_assigned() -> None:
    expression = (
        "Cycle_Time*flow_lane/GR_phase*"
        "(a1*(1+flow_lane**a2)+a3*exp(a4*flow_lane))"
    )
    bounds = coefficient_bounds_for_expression(
        expression, CONDITIONED_ROLE_BOUNDS
    )
    assert bounds["a1"] == CONDITIONED_ROLE_BOUNDS["scale"]
    assert bounds["a2"] == CONDITIONED_ROLE_BOUNDS["power_exponent"]
    assert bounds["a4"] == CONDITIONED_ROLE_BOUNDS["exp_coefficient"]
