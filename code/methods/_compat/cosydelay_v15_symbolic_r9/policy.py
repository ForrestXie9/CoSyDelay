"""Frozen policy for V15 symbolic R9."""

from dataclasses import replace

from methods.cosydelay_v14_physical_feedback_repair.policy import V14_POLICY


V15_POLICY = replace(
    V14_POLICY,
    method_id="cosydelay_v15_symbolic_r9",
    method_status="prospective_symbolic_r9_pending_pilots",
    execution_contract_version="paper_aligned_symbolic_r9_unconstrained_mse_restarts_v1",
    prompt_contract_version="structure_neutral_physics_only_retry_feedback_v1",
)
