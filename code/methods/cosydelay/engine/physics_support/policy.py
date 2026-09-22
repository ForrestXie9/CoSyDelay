"""Frozen policy for the repaired V14 candidate."""

from dataclasses import replace

from methods.cosydelay.engine.prompt_support.policy import V13_POLICY


V14_POLICY = replace(
    V13_POLICY,
    method_id="cosydelay_v14_physical_feedback_repair",
    method_status="prospective_physical_feedback_repair_pending_pilots",
    execution_contract_version=(
        "physical_only_retry_feedback_explicit_training_tie_rank_v1"
    ),
    prompt_contract_version="structure_neutral_physics_only_retry_feedback_v1",
)
