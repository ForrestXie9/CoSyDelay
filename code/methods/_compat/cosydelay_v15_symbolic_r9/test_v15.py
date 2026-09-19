from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from methods.prospective_optimizer_conditioning_v1 import all_log_fitter

from .integration import _symbolic_r9_standard_wrapper
from .source_manifest import V15_SOURCE_FILES, validate_v15_source_manifest


def test_symbolic_r9_wrapper_changes_only_r9_gate():
    result = SimpleNamespace(
        rule_scores={"R2_nondecreasing_flow": 1.0, "R9_zero_green_limit": 0.0},
        lane_rule_scores={"S_T": {"R2_nondecreasing_flow": 1.0, "R9_zero_green_limit": 0.0}},
        lane_errors={"S_T": ["R9: below numerical threshold"]},
        score=0.5,
        joint_pass=False,
        diagnostics={},
    )
    wrapped = _symbolic_r9_standard_wrapper(lambda *args, **kwargs: result)
    observed = wrapped("expr", {}, [], [])
    assert observed.rule_scores["R2_nondecreasing_flow"] == 1.0
    assert observed.rule_scores["R9_zero_green_limit"] == 1.0
    assert observed.joint_pass
    assert observed.lane_errors == {}
    assert observed.diagnostics["numerical_near_zero_probe_role"] == "diagnostic_only"


def test_symbolic_r9_wrapper_preserves_non_r9_failure():
    result = SimpleNamespace(
        rule_scores={"R2_nondecreasing_flow": 0.0, "R9_zero_green_limit": 0.0},
        lane_rule_scores={
            "S_T": {"R2_nondecreasing_flow": 0.0, "R9_zero_green_limit": 0.0}
        },
        lane_errors={
            "S_T": [
                "R2: delay decreases as flow increases",
                "R9: below numerical threshold",
            ]
        },
        score=0.0,
        joint_pass=False,
        diagnostics={},
    )
    wrapped = _symbolic_r9_standard_wrapper(lambda *args, **kwargs: result)
    observed = wrapped("expr", {}, [], [])
    assert observed.rule_scores["R2_nondecreasing_flow"] == 0.0
    assert observed.rule_scores["R9_zero_green_limit"] == 1.0
    assert not observed.joint_pass
    assert observed.lane_errors == {
        "S_T": ["R2: delay decreases as flow increases"]
    }


def _two_restart_payload(*, constrain_r9: bool):
    return {
        "initials": np.asarray([[1.0], [2.0]], dtype=float),
        "bounds": [(0.1, 10.0)],
        "restart_kinds": ["objective_best", "numerical_r9_feasible"],
        "maxiter": 1,
        "maxfun": 2,
        "approach": "S",
        "block_lanes": ["S_T"],
        "coefficient_names": ["a1"],
        "fitted_rows": 1,
        "zero_flow_rows_excluded": 0,
        "zero_flow_policy": "none",
        "warm_values_inherited": 0,
        "r9_constrained_restart_selection": constrain_r9,
    }


def _fake_minimizer_with_objectives(objectives):
    calls = []

    def fake_minimize(fun, x0, *args, **kwargs):
        index = len(calls)
        calls.append(index)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            success=True,
            status=0,
            message="synthetic test result",
            fun=float(objectives[index]),
            nit=1,
            nfev=1,
            njev=1,
            jac=np.asarray([0.0]),
        )

    return fake_minimize


def test_v15_unconstrained_restart_selects_minimum_training_mse():
    with patch.object(
        all_log_fitter,
        "minimize",
        side_effect=_fake_minimizer_with_objectives([1.0, 2.0]),
    ), patch.object(
        all_log_fitter,
        "_r9_probe_restart",
        side_effect=[(False, {"S_T": 100.0}), (True, {"S_T": 20_000.0})],
    ):
        observed = all_log_fitter._solve_approach_all_log(
            _two_restart_payload(constrain_r9=False)
        )
    assert observed["chosen_restart"] == 0
    assert observed["objective"] == 1.0
    assert not observed["restarts"][0]["r9_probe_pass"]
    assert observed["r9_feasible_solution_available"]


def test_legacy_constrained_selector_would_choose_numerical_r9_restart():
    with patch.object(
        all_log_fitter,
        "minimize",
        side_effect=_fake_minimizer_with_objectives([1.0, 2.0]),
    ), patch.object(
        all_log_fitter,
        "_r9_probe_restart",
        side_effect=[(False, {"S_T": 100.0}), (True, {"S_T": 20_000.0})],
    ):
        observed = all_log_fitter._solve_approach_all_log(
            _two_restart_payload(constrain_r9=True)
        )
    assert observed["chosen_restart"] == 1
    assert observed["objective"] == 2.0


def test_v15_source_manifest_is_complete_and_unique():
    validate_v15_source_manifest()
    resolved = [path.resolve() for path in V15_SOURCE_FILES]
    assert len(resolved) == len(set(resolved))
