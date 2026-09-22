"""Frozen prospective policy for the manuscript principle-wise candidate."""

from dataclasses import replace

from methods.cosydelay._internal.numeric_fitting.policy import V15_POLICY


V16_POLICY = replace(
    V15_POLICY,
    method_id="cosydelay_v16_manuscript_principlewise",
    method_status="p3g2_api_pilots_complete_p10g10_confirmation_pending",
    execution_contract_version="training_only_manuscript_principlewise_v10",
    prompt_contract_version=(
        "manuscript_seven_rule_physics_only_feedback_v4_output_schema_last"
    ),
    fitness_definition=(
        "training_mean_nonnegative_approach_r2_plus_equal_mean_over_"
        "movement_by_seven_rule_physical_scores"
    ),
    coefficient_robust_r9_prefit_gate=False,
    fitted_enhanced_physics_is_hard_gate=False,
    # The generic flag controls the superseded eight-rule feedback augmenter.
    # V16 supplies its own compact R1--R7 vector through validation_result.
    targeted_physical_feedback=False,
)
