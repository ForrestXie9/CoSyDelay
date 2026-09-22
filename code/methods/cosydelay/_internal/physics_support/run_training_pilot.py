"""P3/G2 Training-only V14 repair pilot."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from importlib import metadata

from methods.cosydelay._internal.candidate_engine import (
    run_final_prompt_pilot as base,
)

from .integration import install_v14_candidate
from .policy import V14_POLICY
from .source_manifest import GMINI, V14_SOURCE_FILES, validate_v14_source_manifest
from methods.cosydelay._internal.data_protocol.run_formal_training_search import sha256_file


_PROHIBITED_METRICS = re.compile(
    r"PARENT TRAINING EVALUATION|Paper Fitness:|Pooled RMSE:|Pooled MAE:|"
    r"Weakest Training approach:|Current best Training Fitness:|Parent Fitness gap:"
)


def _v14_prompt_audit(llm_attempts: list[dict]) -> dict:
    prompts = [
        str(item["prompt"])
        for item in llm_attempts
        if item.get("prompt") and item.get("status") == "success"
    ]
    mutation = [item for item in prompts if "MUTATION TASK:" in item]
    repaired = [
        item
        for item in mutation
        if "POST-FIT PHYSICAL REPAIR FROM THE REJECTED CANDIDATE:" in item
    ]
    audit = {
        "successful_prompt_calls": len(prompts),
        "mutation_prompt_calls": len(mutation),
        "physical_repair_prompt_calls": len(repaired),
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
            _PROHIBITED_METRICS.search(item) is None for item in mutation
        ),
        "repair_is_rule_only": all(
            any(f"- {rule}:" in item for rule in ("R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "Domain"))
            and "Training" not in item
            and "RMSE" not in item
            and "MAE" not in item
            for item in repaired
        ),
        "fixed_global_template_absent": all(
            "Use the global skeleton" not in item
            and "fixed first-power inverse-GR_phase" not in item
            and "explicit uncancellable /GR_phase factor" not in item
            for item in prompts
        ),
    }
    required = (
        "single_physical_contract_per_prompt",
        "single_output_schema_per_prompt",
        "mutation_has_parent",
        "mutation_has_physical_audit",
        "mutation_training_feedback_absent",
        "repair_is_rule_only",
        "fixed_global_template_absent",
    )
    if not prompts or not mutation or not all(audit[name] for name in required):
        raise RuntimeError(f"V14 prompt audit failed: {audit}")
    return audit


def _output_argument() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError("missing --output argument") from exc


def main() -> int:
    output = _output_argument()
    validate_v14_source_manifest()
    previous = (base.install_v10_candidate, base.V10_POLICY, base._prompt_audit)
    base.install_v10_candidate = install_v14_candidate
    base.V10_POLICY = V14_POLICY
    base._prompt_audit = _v14_prompt_audit
    try:
        code = base.main()
    finally:
        base.install_v10_candidate, base.V10_POLICY, base._prompt_audit = previous

    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "status": "completed_v14_p3g2_training_pilot",
            "method_id": V14_POLICY.method_id,
            "method_status": V14_POLICY.method_status,
            "evolution_order": [
                "paper_fitness_descending",
                "training_macro_raw_r2_descending_on_exact_fitness_tie",
                "training_pooled_rmse_ascending_on_remaining_tie",
            ],
            "physical_only_retry_feedback": True,
            "llm_training_metric_feedback_exposed": False,
            "paper_fitness_changed": False,
            "validation_or_test_used_for_selection": False,
            "source_manifest_sha256": {
                str(path.relative_to(GMINI)).replace("\\", "/"): sha256_file(path)
                for path in V14_SOURCE_FILES
            },
            "runtime_package_versions": {
                name: metadata.version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn", "sympy", "numexpr")
            },
        }
    )
    base.write_json(result_path, result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
