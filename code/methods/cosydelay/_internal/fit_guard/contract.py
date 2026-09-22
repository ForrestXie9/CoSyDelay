"""Machine-checkable contract for V18's bounded fit and R7 pre-fit gate."""

from __future__ import annotations

from dataclasses import dataclass

from methods.cosydelay._internal.retry_protocol.contract import (
    V17_CONTRACT,
    V17MethodContract,
    validate_v17_contract,
)
from methods.cosydelay._internal.retry_protocol.policy import V17_POLICY

from .policy import V18_POLICY


@dataclass(frozen=True)
class V18MethodContract(V17MethodContract):
    schema_version: int = 13
    method_id: str = V18_POLICY.method_id
    prefit_filter_scope: str = (
        "parser_grammar_sampled_numerical_legality_unambiguous_coefficient_"
        "roles_and_proven_finite_positive_coefficient_R7_endpoint_rejection"
    )
    prefit_physical_principle_rejection_enabled: bool = True
    prefit_r7_gate_enabled: bool = True
    prefit_r7_coefficient_assumption: str = (
        "all_fitted_coefficients_are_arbitrary_positive_finite_values"
    )
    prefit_r7_reject_verdict: str = (
        "complete_response_proven_finite_as_GR_phase_approaches_zero"
    )
    prefit_r7_unknown_action: str = (
        "allow_to_bounded_fit_and_existing_fitted_seven_rule_score"
    )
    prefit_r7_uses_training_values: bool = False
    prefit_r7_uses_targets: bool = False
    prefit_r7_uses_validation_or_test: bool = False
    prefit_r7_changes_fitness_definition: bool = False
    candidate_fit_wall_timeout_seconds: float = 100.0
    candidate_fit_startup_timeout_seconds: float = 30.0
    candidate_fit_close_timeout_seconds: float = 5.0
    candidate_fit_isolated_supervisor_process: bool = True
    candidate_fit_descendant_tree_terminated_on_timeout: bool = True
    candidate_fit_supervisor_restarted_after_failure: bool = True
    timed_out_candidate_counts_toward_budget: bool = False
    failed_candidate_counts_toward_budget: bool = False
    successful_candidate_budget: int = 110
    rejected_candidate_action: str = (
        "ban_canonical_expression_and_regenerate_same_population_slot"
    )
    rejected_candidate_rng_policy: str = (
        "commit_optimizer_rng_state_only_after_successful_fit"
    )
    timeout_used_as_fitness_component: bool = False
    timeout_feedback_contains_accuracy_metrics: bool = False
    timeout_feedback_contains_validation_or_test_data: bool = False
    persistent_approach_pool_retained_inside_supervisor: bool = True
    intersection_process_wall_timeout_seconds: float = 10_800.0
    test_evaluator_wall_timeout_seconds: float = 1_800.0
    full_training_launcher_wall_timeout_seconds: float = 68_400.0
    active_process_waits_are_bounded: bool = True
    generation_budget_changed_from_v17: bool = False
    optimizer_contract_changed_from_v17: bool = False
    fitness_contract_changed_from_v17: bool = False
    physical_contract_changed_from_v16: bool = True
    physical_contract_changed_from_v17: bool = True
    physical_scoring_contract_changed_from_v17: bool = False
    prompt_contract_changed_from_v17: bool = False


V18_CONTRACT = V18MethodContract()


def validate_v18_contract() -> None:
    """Fail closed unless V18 matches its declared admissibility repairs."""
    validate_v17_contract()
    base_policy = V17_POLICY.to_dict()
    policy = V18_POLICY.to_dict()
    allowed_policy_changes = {
        "method_id",
        "method_status",
        "execution_contract_version",
        "coefficient_robust_r9_prefit_gate",
    }
    policy_drift = {
        key: {"v17": value, "v18": policy.get(key)}
        for key, value in base_policy.items()
        if key not in allowed_policy_changes and policy.get(key) != value
    }
    base_contract = V17_CONTRACT.to_dict()
    contract = V18_CONTRACT.to_dict()
    contract_drift = {
        key: {"v17": value, "v18": contract.get(key)}
        for key, value in base_contract.items()
        if key
        not in {
            "schema_version",
            "method_id",
            "prefit_filter_scope",
            "prefit_physical_principle_rejection_enabled",
            "physical_contract_changed_from_v16",
        }
        and contract.get(key) != value
    }
    expected_policy = {
        "method_id": "cosydelay_v18_fit_timeout_guard",
        "execution_contract_version": (
            "training_only_manuscript_principlewise_v13_bounded_fit_r7_prefit"
        ),
        "prompt_contract_version": V17_POLICY.prompt_contract_version,
        "coefficient_robust_r9_prefit_gate": True,
    }
    id_drift = {
        key: {"expected": expected, "observed": policy.get(key)}
        for key, expected in expected_policy.items()
        if policy.get(key) != expected
    }
    required_flags = {
        "candidate_fit_isolated_supervisor_process": True,
        "candidate_fit_descendant_tree_terminated_on_timeout": True,
        "candidate_fit_supervisor_restarted_after_failure": True,
        "timed_out_candidate_counts_toward_budget": False,
        "failed_candidate_counts_toward_budget": False,
        "successful_candidate_budget": 110,
        "timeout_used_as_fitness_component": False,
        "timeout_feedback_contains_accuracy_metrics": False,
        "timeout_feedback_contains_validation_or_test_data": False,
        "persistent_approach_pool_retained_inside_supervisor": True,
        "active_process_waits_are_bounded": True,
        "generation_budget_changed_from_v17": False,
        "optimizer_contract_changed_from_v17": False,
        "fitness_contract_changed_from_v17": False,
        "physical_contract_changed_from_v16": True,
        "physical_contract_changed_from_v17": True,
        "physical_scoring_contract_changed_from_v17": False,
        "prompt_contract_changed_from_v17": False,
        "prefit_physical_principle_rejection_enabled": True,
        "prefit_r7_gate_enabled": True,
        "prefit_r7_uses_training_values": False,
        "prefit_r7_uses_targets": False,
        "prefit_r7_uses_validation_or_test": False,
        "prefit_r7_changes_fitness_definition": False,
    }
    flag_drift = {
        key: {"expected": expected, "observed": getattr(V18_CONTRACT, key)}
        for key, expected in required_flags.items()
        if getattr(V18_CONTRACT, key) != expected
    }
    positive_timeouts = {
        name: getattr(V18_CONTRACT, name)
        for name in (
            "candidate_fit_wall_timeout_seconds",
            "candidate_fit_startup_timeout_seconds",
            "candidate_fit_close_timeout_seconds",
            "intersection_process_wall_timeout_seconds",
            "test_evaluator_wall_timeout_seconds",
            "full_training_launcher_wall_timeout_seconds",
        )
    }
    invalid_timeouts = {
        name: value for name, value in positive_timeouts.items() if value <= 0.0
    }
    if (
        V18_CONTRACT.candidate_fit_wall_timeout_seconds != 100.0
        or V18_CONTRACT.successful_candidate_budget
        != V18_CONTRACT.population * (V18_CONTRACT.generations + 1)
    ):
        flag_drift["fixed_budget_or_timeout"] = {
            "timeout": V18_CONTRACT.candidate_fit_wall_timeout_seconds,
            "successful_budget": V18_CONTRACT.successful_candidate_budget,
        }
    if policy_drift or contract_drift or id_drift or flag_drift or invalid_timeouts:
        raise RuntimeError(
            "V18 differs from V17 outside the declared R7/fit-bound repairs: "
            f"policy={policy_drift}, contract={contract_drift}, ids={id_drift}, "
            f"flags={flag_drift}, invalid_timeouts={invalid_timeouts}"
        )
