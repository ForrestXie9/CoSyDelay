"""Frozen policy for the V17 cross-batch retry repair."""

from dataclasses import replace

from methods.cosydelay_v16_manuscript_principlewise.policy import V16_POLICY


V17_PROMPT_CONTRACT_ID = (
    "manuscript_seven_rule_physics_feedback_v5_cross_batch_legality_repair"
)

V17_POLICY = replace(
    V16_POLICY,
    method_id="cosydelay_v17_cross_batch_retry",
    method_status="cross_batch_legality_repair_frozen_p10g10_pending",
    execution_contract_version=(
        "training_only_manuscript_principlewise_v11_cross_batch_retry"
    ),
    prompt_contract_version=V17_PROMPT_CONTRACT_ID,
)
