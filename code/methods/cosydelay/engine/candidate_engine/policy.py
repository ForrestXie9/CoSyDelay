"""Frozen prospective policy layered on the clean V9 protocol."""

from __future__ import annotations

from dataclasses import replace

from methods.cosydelay.engine.data_protocol.policy import CLEAN_POLICY


V10_POLICY = replace(
    CLEAN_POLICY,
    method_id="cosydelay_v10_accuracy_first_parsimony_alllog",
    method_status=(
        "prospective_candidate_prompt_v6_efficiency_v1_confirmed_p3g2_i1_i2_"
        "pending_formal_p10g10"
    ),
    execution_contract_version="v6_equivalent_audit_reuse_family_cache_v1",
    optimizer="L-BFGS-B_all_positive_log_coordinates",
    prompt_contract_version="paper_structured_clean_training_feedback_v6",
    structural_diversity_mode="family_unique",
    reuse_standard_fitted_physics_audit=True,
    reuse_prefit_symbolic_endpoint_audit=True,
    reject_fitted_structural_families=True,
)
