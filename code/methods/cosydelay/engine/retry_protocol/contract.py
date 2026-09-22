"""Machine-checkable contract for V17's isolated retry repair."""

from __future__ import annotations

from dataclasses import dataclass

from methods.cosydelay.engine.training_protocol.contract import (
    V16_CONTRACT,
    V16MethodContract,
    validate_v16_contract,
)
from methods.cosydelay.engine.training_protocol.policy import V16_POLICY

from .policy import V17_POLICY, V17_PROMPT_CONTRACT_ID


@dataclass(frozen=True)
class V17MethodContract(V16MethodContract):
    schema_version: int = 11
    method_id: str = V17_POLICY.method_id
    cross_batch_legality_failure_memory: bool = True
    cross_batch_feedback_scope: str = (
        "last_failed_generation_batch_reason_and_one_truncated_expression_only"
    )
    cross_batch_feedback_cleared_after_success: bool = True
    cross_batch_retry_sequence_in_prompt: bool = True
    cross_batch_feedback_used_for_selection: bool = False
    cross_batch_feedback_contains_accuracy_metrics: bool = False
    cross_batch_feedback_contains_validation_or_test_data: bool = False
    generation_budget_changed_from_v16: bool = False
    optimizer_contract_changed_from_v16: bool = False
    fitness_contract_changed_from_v16: bool = False
    physical_contract_changed_from_v16: bool = False


V17_CONTRACT = V17MethodContract()


def validate_v17_contract() -> None:
    """Fail closed unless V17 differs from V16 only in retry bookkeeping."""
    validate_v16_contract()
    base_policy = V16_POLICY.to_dict()
    policy = V17_POLICY.to_dict()
    allowed_policy_changes = {
        "method_id",
        "method_status",
        "execution_contract_version",
        "prompt_contract_version",
    }
    policy_drift = {
        key: {"v16": value, "v17": policy.get(key)}
        for key, value in base_policy.items()
        if key not in allowed_policy_changes and policy.get(key) != value
    }
    base_contract = V16_CONTRACT.to_dict()
    contract = V17_CONTRACT.to_dict()
    contract_drift = {
        key: {"v16": value, "v17": contract.get(key)}
        for key, value in base_contract.items()
        if key not in {"schema_version", "method_id"} and contract.get(key) != value
    }
    expected_ids = {
        "method_id": "cosydelay_v17_cross_batch_retry",
        "execution_contract_version": (
            "training_only_manuscript_principlewise_v11_cross_batch_retry"
        ),
        "prompt_contract_version": V17_PROMPT_CONTRACT_ID,
    }
    id_drift = {
        key: {"expected": value, "observed": policy.get(key)}
        for key, value in expected_ids.items()
        if policy.get(key) != value
    }
    repair_flags = {
        "cross_batch_legality_failure_memory": True,
        "cross_batch_feedback_cleared_after_success": True,
        "cross_batch_retry_sequence_in_prompt": True,
        "cross_batch_feedback_used_for_selection": False,
        "cross_batch_feedback_contains_accuracy_metrics": False,
        "cross_batch_feedback_contains_validation_or_test_data": False,
        "generation_budget_changed_from_v16": False,
        "optimizer_contract_changed_from_v16": False,
        "fitness_contract_changed_from_v16": False,
        "physical_contract_changed_from_v16": False,
    }
    flag_drift = {
        key: {"expected": value, "observed": getattr(V17_CONTRACT, key)}
        for key, value in repair_flags.items()
        if getattr(V17_CONTRACT, key) != value
    }
    expected_scope = (
        "last_failed_generation_batch_reason_and_one_truncated_expression_only"
    )
    if V17_CONTRACT.cross_batch_feedback_scope != expected_scope:
        flag_drift["cross_batch_feedback_scope"] = {
            "expected": expected_scope,
            "observed": V17_CONTRACT.cross_batch_feedback_scope,
        }
    if policy_drift or contract_drift or id_drift or flag_drift:
        raise RuntimeError(
            "V17 differs from V16 outside the declared retry repair: "
            f"policy={policy_drift}, contract={contract_drift}, "
            f"ids={id_drift}, flags={flag_drift}"
        )
