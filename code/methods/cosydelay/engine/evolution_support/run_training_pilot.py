"""P3/G2 Training-only runner for the isolated V11 repair candidate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from methods.cosydelay.engine.candidate_engine import (
    run_final_prompt_pilot as base,
)

from .integration import install_v11_candidate
from .policy import V11_POLICY


_BASE_INSTALLER = base.install_v10_candidate
_BASE_POLICY = base.V10_POLICY
_BASE_PROMPT_AUDIT = base._prompt_audit


def _v11_prompt_audit(llm_attempts: list[dict]) -> dict:
    audit = _BASE_PROMPT_AUDIT(llm_attempts)
    prompts = [
        str(item["prompt"])
        for item in llm_attempts
        if item.get("prompt") and item.get("status") == "success"
    ]
    mutation = [item for item in prompts if "MUTATION TASK:" in item]
    audit.update(
        {
            "complete_response_monotonicity_present": all(
                "Apply monotonicity to the COMPLETE delay response" in item
                for item in prompts
            ),
            "isolated_h_monotonicity_absent": all(
                "H must be finite, strictly positive" not in item
                or "nondecreasing in flow_lane" not in item
                for item in prompts
            ),
            "per_approach_training_feedback_present": all(
                "Weakest Training approach:" in item for item in mutation
            ),
        }
    )
    required = (
        "complete_response_monotonicity_present",
        "isolated_h_monotonicity_absent",
        "per_approach_training_feedback_present",
    )
    if not all(audit[name] for name in required):
        raise RuntimeError(f"V11 prompt audit failed: {audit}")
    return audit


def _output_argument() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError("missing --output argument") from exc


def main() -> int:
    output = _output_argument()
    previous = (base.install_v10_candidate, base.V10_POLICY, base._prompt_audit)
    base.install_v10_candidate = install_v11_candidate
    base.V10_POLICY = V11_POLICY
    base._prompt_audit = _v11_prompt_audit
    try:
        code = base.main()
    finally:
        base.install_v10_candidate, base.V10_POLICY, base._prompt_audit = previous

    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "status": "completed_v11_total_response_p3g2_training_pilot",
            "method_id": V11_POLICY.method_id,
            "method_status": V11_POLICY.method_status,
            "v11_changes": {
                "physics_applied_to_complete_response": True,
                "isolated_h_monotonicity_required": False,
                "structural_diversity_mode": "canonical",
                "fitted_family_blacklist": False,
                "per_approach_training_feedback": True,
            },
        }
    )
    base.write_json(result_path, result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
