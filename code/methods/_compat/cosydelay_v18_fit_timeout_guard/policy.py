"""Policy identifiers for V18 before its first formal P10/G10 run."""

from dataclasses import replace

from methods.cosydelay_v17_cross_batch_retry.policy import V17_POLICY


V18_POLICY = replace(
    V17_POLICY,
    method_id="cosydelay_v18_fit_timeout_guard",
    method_status="bounded_fit_and_conservative_r7_prefit_p10g10_pending",
    execution_contract_version=(
        "training_only_manuscript_principlewise_v13_bounded_fit_r7_prefit"
    ),
    coefficient_robust_r9_prefit_gate=True,
)
