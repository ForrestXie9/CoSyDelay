"""Prospective P3/G2 Training-only V16 API pilot entry point."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from methods.cosydelay_v10_accuracy_first_parsimony_alllog import (
    run_final_prompt_pilot as base,
)

from .integration import install_v16_candidate
from .contract import V16_CONTRACT, validate_v16_contract
from .policy import V16_POLICY
from .prompt import output_schema_is_last
from .source_manifest import validate_v16_source_manifest


def _output_argument() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError("missing --output argument") from exc


def _prompt_audit(llm_attempts: list[dict]) -> dict:
    prompts = [
        str(item["prompt"])
        for item in llm_attempts
        if item.get("prompt") and item.get("status") == "success"
    ]
    mutation = [item for item in prompts if "MUTATION TASK:" in item]
    audit = {
        "successful_prompt_calls": len(prompts),
        "mutation_prompt_calls": len(mutation),
        "single_physical_contract_per_prompt": all(
            item.count("PHYSICAL AND STRUCTURAL REQUIREMENTS:") == 1
            for item in prompts
        ),
        "single_output_schema_per_prompt": all(
            item.count("### Expression") == 1 for item in prompts
        ),
        "output_schema_is_last_in_every_prompt": all(
            output_schema_is_last(item) for item in prompts
        ),
        "mutation_has_parent": all(
            item.count("PARENT EXPRESSION:") == 1 for item in mutation
        ),
        "training_metric_feedback_absent": all(
            "PARENT TRAINING EVALUATION" not in item
            and "Paper Fitness:" not in item
            and "Pooled RMSE:" not in item
            and "Pooled MAE:" not in item
            for item in mutation
        ),
        "finite_nonnegative_low_demand_present": all(
            "finite,\n   nonnegative limit" in item
            and "may be positive" in item
            for item in prompts
        ),
        "exact_zero_flow_absent": all(
            "y(0, GR_phase, Cycle_Time)=0" not in item
            and "factor flow_lane from the complete expression" not in item
            for item in prompts
        ),
        "numeric_r9_threshold_absent": all(
            "10000" not in item and "10,000" not in item for item in prompts
        ),
        "approach_supervision_disclosed": all(
            "available observed targets are approach-average delays" in item
            and "jointly calibrated through" in item
            for item in prompts
        ),
        "llm_does_not_fit_coefficients": all(
            "never supplies fitted numerical coefficient values" in item
            for item in prompts
        ),
    }
    required = tuple(audit)[2:]
    if not prompts or not mutation or not all(audit[name] for name in required):
        raise RuntimeError(f"V16 prompt audit failed: {audit}")
    return audit


def _history_principle_summary(history_path: Path) -> dict:
    history = json.loads(history_path.read_text(encoding="utf-8"))
    evaluated = [item for item in history if item.get("event") == "evaluated"]
    joint = [
        item for item in evaluated if bool(item.get("physical_joint_pass", False))
    ]
    details = [
        item.get("evaluation_details", {})
        for item in evaluated
        if isinstance(item.get("evaluation_details"), dict)
    ]
    v16_walls = [
        float(item.get("v16_physics_wall_seconds", 0.0))
        for item in details
    ]
    bypass_walls = [
        float(item.get("compatibility_bypass_physical_wall_seconds", 0.0))
        + float(
            item.get(
                "compatibility_bypass_enhanced_physics_wall_seconds", 0.0
            )
        )
        for item in details
    ]
    return {
        "principlewise_evaluations": len(evaluated),
        "strict_joint_passing_evaluations": len(joint),
        "strict_joint_pass_rate": len(joint) / len(evaluated) if evaluated else None,
        "v16_physics_timing": {
            "evaluations": len(v16_walls),
            "wall_seconds_sum": float(sum(v16_walls)),
            "wall_seconds_mean": (
                float(sum(v16_walls) / len(v16_walls)) if v16_walls else None
            ),
            "compatibility_bypass_wall_seconds_sum": float(
                sum(bypass_walls)
            ),
            "score_reused_as_enhanced_audit": True,
        },
    }


def _force_principlewise_evolution(original):
    """Override legacy runner labels and disable its eight-rule feedback map."""
    def wrapped(*args, **kwargs):
        kwargs["score_mode"] = "principlewise"
        kwargs["targeted_physical_feedback"] = False
        return original(*args, **kwargs)

    return wrapped


def _diagnostic_epsilon_parsimony(evaluations, original):
    """Keep the inherited complexity report strictly diagnostic.

    The retained reporter only defines its epsilon set among strict joint
    passes and raises when that set is empty.  In V16, however, strict joint
    pass is explicitly diagnostic and the seven-component score is the
    physical Fitness term.  An empty strict-pass subset therefore makes this
    optional report unavailable; it must not invalidate a completed search.
    """
    strict_passes = sum(
        bool((item.get("enhanced_physics") or {}).get("joint_pass"))
        for item in evaluations.values()
    )
    if strict_passes:
        return original(evaluations)
    return {
        "status": "unavailable_no_strict_joint_pass",
        "diagnostic_only_not_used_for_evolution": True,
        "strict_joint_pass_role": "diagnostic_only",
        "evaluated_candidates": len(evaluations),
        "strict_joint_passing_candidates": 0,
        "selected_expression": None,
        "fitness_tolerance": None,
    }


def main() -> int:
    output = _output_argument()
    validate_v16_contract()
    previous = (
        base.install_v10_candidate,
        base.V10_POLICY,
        base._prompt_audit,
        base.validate_source_manifest,
        base.evolve_universal_lane_expression,
        base.select_epsilon_parsimonious,
    )
    base.install_v10_candidate = install_v16_candidate
    base.V10_POLICY = V16_POLICY
    base._prompt_audit = _prompt_audit
    base.validate_source_manifest = validate_v16_source_manifest
    base.evolve_universal_lane_expression = _force_principlewise_evolution(
        base.evolve_universal_lane_expression
    )
    original_epsilon_diagnostic = base.select_epsilon_parsimonious
    base.select_epsilon_parsimonious = lambda evaluations: (
        _diagnostic_epsilon_parsimony(
            evaluations, original_epsilon_diagnostic
        )
    )
    try:
        code = base.main()
    finally:
        (
            base.install_v10_candidate,
            base.V10_POLICY,
            base._prompt_audit,
            base.validate_source_manifest,
            base.evolve_universal_lane_expression,
            base.select_epsilon_parsimonious,
        ) = previous

    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "status": "completed_v16_p3g2_training_only_pilot",
            "method_id": V16_POLICY.method_id,
            "method_status": V16_POLICY.method_status,
            "evolution_selection": (
                "Training mean nonnegative approach R2 plus the equal mean "
                "over movement-by-seven-rule physical scores"
            ),
            "accuracy_supervision_level": "approach",
            "movement_delay_labels_available": False,
            "coefficient_parameterization_level": "movement",
            "paper_fitness_changed": False,
            "fitness_changed_from_v15_binary_implementation": True,
            "fitness_contract_alignment": (
                "recovered_manuscript_R2_plus_principlewise_physical_score"
            ),
            "strict_joint_pass_role": "diagnostic_only_not_a_hard_gate",
            "validation_or_test_used_for_selection": False,
            "method_contract": V16_CONTRACT.to_dict(),
        }
    )
    principle_summary = _history_principle_summary(output / "history.json")
    result.update(principle_summary)
    v16_timing = principle_summary["v16_physics_timing"]
    result.setdefault("efficiency_timing", {}).update(
        {
            "enhanced_physics_wall_seconds_sum": v16_timing[
                "wall_seconds_sum"
            ],
            "v16_principlewise_physics_wall_seconds_sum": v16_timing[
                "wall_seconds_sum"
            ],
            "v16_principlewise_physics_wall_seconds_mean": v16_timing[
                "wall_seconds_mean"
            ],
            "compatibility_bypass_wall_seconds_sum": v16_timing[
                "compatibility_bypass_wall_seconds_sum"
            ],
            "v16_score_reused_as_enhanced_audit": True,
        }
    )
    result.pop("physically_passing_evaluations", None)
    result.pop("postfit_pass_rate", None)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
