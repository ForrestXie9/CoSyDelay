"""P3/G2 Training-only V13 pilot."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from methods.cosydelay_v10_accuracy_first_parsimony_alllog import run_final_prompt_pilot as base

from .integration import install_v13_candidate
from .policy import V13_POLICY


def _v13_prompt_audit(llm_attempts: list[dict]) -> dict:
    prompts = [
        str(item["prompt"])
        for item in llm_attempts
        if item.get("prompt") and item.get("status") == "success"
    ]
    mutation = [item for item in prompts if "MUTATION TASK:" in item]
    audit = {
            "successful_prompt_calls": len(prompts),
            "mutation_prompt_calls": len(mutation),
            "maximum_prompt_characters": max(map(len, prompts), default=0),
            "single_physical_contract_per_prompt": all(
                item.count("PHYSICAL AND STRUCTURAL REQUIREMENTS:") == 1
                for item in prompts
            ),
            "single_output_schema_per_prompt": all(
                item.count("### Expression") == 1 for item in prompts
            ),
            "mutation_has_parent": all(
                item.count("PARENT EXPRESSION:") == 1 for item in mutation
            ),
            "mutation_has_physical_audit": all(
                item.count("PARENT PHYSICAL AUDIT:") == 1 for item in mutation
            ),
            "mutation_training_feedback_absent": all(
                "PARENT TRAINING EVALUATION" not in item
                and "Paper Fitness:" not in item
                and "Weakest Training approach:" not in item
                and "Pooled RMSE:" not in item
                and "Pooled MAE:" not in item
                for item in mutation
            ),
            "fixed_global_template_absent": all(
                "Use the global skeleton" not in item
                and "fixed first-power inverse-GR_phase" not in item
                and "explicit uncancellable /GR_phase factor" not in item
                for item in prompts
            ),
            "fixed_amplitude_prohibited": all(
                "not as an unavoidable delay amplitude" in item for item in prompts
            ),
            "complete_response_monotonicity_present": all(
                "Apply these tests to the complete expression" in item
                for item in prompts
            ),
        }
    required = (
        "single_physical_contract_per_prompt",
        "single_output_schema_per_prompt",
        "mutation_has_parent",
        "mutation_has_physical_audit",
        "mutation_training_feedback_absent",
        "fixed_global_template_absent",
        "fixed_amplitude_prohibited",
        "complete_response_monotonicity_present",
    )
    if not prompts or not mutation or not all(audit[name] for name in required):
        raise RuntimeError(f"V13 prompt audit failed: {audit}")
    return audit


def _output_argument() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError("missing --output argument") from exc


def main() -> int:
    output = _output_argument()
    previous = (base.install_v10_candidate, base.V10_POLICY, base._prompt_audit)
    base.install_v10_candidate = install_v13_candidate
    base.V10_POLICY = V13_POLICY
    base._prompt_audit = _v13_prompt_audit
    try:
        code = base.main()
    finally:
        base.install_v10_candidate, base.V10_POLICY, base._prompt_audit = previous

    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "status": "completed_v13_p3g2_training_pilot",
            "method_id": V13_POLICY.method_id,
            "method_status": V13_POLICY.method_status,
            "evolution_order": [
                "paper_fitness_descending",
                "training_macro_raw_r2_descending_on_exact_fitness_tie",
                "training_pooled_rmse_ascending_on_remaining_tie",
            ],
            "paper_fitness_changed": False,
            "validation_or_test_used_for_selection": False,
        }
    )
    base.write_json(result_path, result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
