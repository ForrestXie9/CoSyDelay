"""Runtime audit for the concise V20 paper-style prompt contract."""

from __future__ import annotations

from methods.cosydelay.engine.training_protocol.prompt import (
    output_schema_is_last,
)


def audit_v20_prompts(llm_attempts: list[dict]) -> dict:
    prompts = [
        str(item["prompt"])
        for item in llm_attempts
        if item.get("prompt") and item.get("status") == "success"
    ]
    regeneration = [item for item in prompts if "REGENERATION TASK:" in item]
    initialization = [
        item
        for item in prompts
        if "INITIALIZATION TASK:" in item
    ]
    audit = {
        "successful_prompt_calls": len(prompts),
        "initialization_prompt_calls": len(initialization),
        "regeneration_prompt_calls": len(regeneration),
        "single_physical_contract_per_prompt": all(
            item.count("The expression must satisfy:") == 1 for item in prompts
        ),
        "single_output_schema_per_prompt": all(
            item.count("### Expression") == 1 for item in prompts
        ),
        "output_schema_is_last_in_every_prompt": all(
            output_schema_is_last(item) for item in prompts
        ),
        "regeneration_has_one_parent": all(
            item.count("Parent expression:") == 1 for item in regeneration
        ),
        "regeneration_exploration_is_open": all(
            "Replace or fundamentally reorganize the parent's main demand-green "
            "interaction" in item
            and "algebraically equivalent rewrite is insufficient" in item
            and "No functional family is prescribed" in item
            for item in regeneration
        ),
        "training_validation_test_feedback_absent": all(
            "PARENT TRAINING EVALUATION" not in item
            and "Paper Fitness:" not in item
            and "Pooled RMSE:" not in item
            and "Pooled MAE:" not in item
            and "Validation R2" not in item
            and "Test R2" not in item
            for item in prompts
        ),
        "seven_principles_present": all(
            "1. Use flow_lane, GR_phase, and Cycle_Time." in item
            and "7. For positive flow_lane and Cycle_Time" in item
            for item in prompts
        ),
        "numeric_operating_domain_absent": all(
            "0.001<=flow_lane" not in item
            and "0.02<=GR_phase" not in item
            and "30<=Cycle_Time" not in item
            for item in prompts
        ),
        "a8_limit_absent": all("a8" not in item for item in prompts),
        "python_power_operator_disclosed": all(
            "Use only +, -, *, /, **, exp, and log; never use ^." in item
            for item in prompts
        ),
        "exact_time_dimension_required": all(
            "homogeneous of degree one in Cycle_Time" in item
            and "Cycle_Time supplies the only time dimension" in item
            for item in prompts
        ),
        "concise_traffic_interpretability_required": all(
            "clear traffic-physical interpretation" in item
            and "compact and sufficiently expressive" in item
            for item in prompts
        ),
        "historical_formula_list_absent": all(
            "NOVELTY REQUIREMENT:" not in item for item in prompts
        ),
        "approach_supervision_disclosed": all(
            "jointly calibrated within each approach" in item
            and "flow-weighted approach-average delay predictions" in item
            for item in prompts
        ),
        "llm_does_not_fit_coefficients": all(
            "do not provide numerical coefficient values" in item.replace("\n", " ")
            for item in prompts
        ),
    }
    required = tuple(audit)[3:]
    if (
        not prompts
        or not initialization
        or not regeneration
        or not all(audit[name] for name in required)
    ):
        raise RuntimeError(f"V20 prompt audit failed: {audit}")
    return audit
