"""Isolated policy for the V13 clean-search repair."""

from dataclasses import replace

from methods.cosydelay_v11_total_response_evolution.policy import V11_POLICY


V13_POLICY = replace(
    V11_POLICY,
    method_id="cosydelay_v13_unbiased_prompt_tie_break",
    method_status="prospective_training_only_prompt_and_tie_repair",
    execution_contract_version="structure_neutral_prompt_internal_training_tie_break_v2",
    prompt_contract_version="structure_neutral_physics_no_training_feedback_v2",
)
