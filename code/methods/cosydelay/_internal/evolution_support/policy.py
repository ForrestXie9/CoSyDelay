"""Isolated policy for the prospective V11 repair candidate."""

from __future__ import annotations

from dataclasses import replace

from methods.cosydelay._internal.data_protocol.policy import CLEAN_POLICY


V11_POLICY = replace(
    CLEAN_POLICY,
    method_id="cosydelay_v11_total_response_evolution",
    method_status="prospective_training_only_repair_pending_p3g2",
    execution_contract_version=(
        "total_response_physics_canonical_evolution_audit_reuse_v1"
    ),
    optimizer="L-BFGS-B_all_positive_log_coordinates",
    prompt_contract_version="total_response_training_feedback_v1",
    # Canonical mode rejects algebraically identical expressions but permits
    # useful variants within a strong structural family to keep evolving.
    structural_diversity_mode="canonical",
    reuse_standard_fitted_physics_audit=True,
    reuse_prefit_symbolic_endpoint_audit=True,
    # A failed fitted member no longer blacklists its entire structural family.
    reject_fitted_structural_families=False,
)
