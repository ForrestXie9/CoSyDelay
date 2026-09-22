"""Run one frozen V17 P10/G10 Training-only search from scratch."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from methods.cosydelay.engine.training_protocol import (
    run_p10g10_training as base,
)
from methods.cosydelay.engine.training_protocol.prompt import OUTPUT_SCHEMA

from .contract import V17_CONTRACT, validate_v17_contract
from .integration import get_last_retry_audit, install_v17_candidate
from .policy import V17_POLICY
from .prompt import correction_contains_supervision_metrics
from .source_manifest import (
    GMINI,
    V17_SOURCE_FILES,
    validate_v17_source_manifest,
)


STATUS = "completed_v17_p10g10_training_only_search"
_BASE_PROMPT_AUDIT = base._prompt_audit


def _source_hashes() -> dict[str, str]:
    return {
        (
            str(path.relative_to(GMINI)).replace("\\", "/")
            if path.is_relative_to(GMINI)
            else str(path)
        ): base.sha256_file(path)
        for path in V17_SOURCE_FILES
    }


def _v17_prompt_audit(llm_attempts: list[dict]) -> dict:
    audit = _BASE_PROMPT_AUDIT(llm_attempts)
    prompts = [
        str(item["prompt"])
        for item in llm_attempts
        if item.get("prompt") and item.get("status") == "success"
    ]
    marker = "CROSS-BATCH LEGALITY CORRECTION:"
    corrected = [prompt for prompt in prompts if marker in prompt]
    sections = [
        prompt.split(marker, 1)[1].rsplit(OUTPUT_SCHEMA, 1)[0]
        for prompt in corrected
    ]
    extension = {
        "cross_batch_correction_prompt_calls": len(corrected),
        "at_most_one_cross_batch_correction_per_prompt": all(
            prompt.count(marker) == 1 for prompt in corrected
        ),
        "cross_batch_correction_has_sequence": all(
            "Correction sequence:" in section for section in sections
        ),
        "cross_batch_correction_has_last_reason": all(
            "Last legality failure:" in section for section in sections
        ),
        "cross_batch_correction_has_no_accuracy_or_split_supervision": all(
            not correction_contains_supervision_metrics(section)
            for section in sections
        ),
    }
    if not all(
        value is True
        for key, value in extension.items()
        if key != "cross_batch_correction_prompt_calls"
    ):
        raise RuntimeError(f"V17 cross-batch prompt audit failed: {extension}")
    audit.update(extension)
    return audit


def validate_retry_audit(records: list[dict]) -> dict:
    """Prove that correction memory is one-batch, legality-only state."""
    errors: list[dict] = []
    pending = False
    failed = 0
    succeeded = 0
    injected = 0
    recovered = 0
    for expected_sequence, item in enumerate(records, 1):
        event = item.get("event")
        observed_sequence = item.get("sequence")
        observed_injected = bool(item.get("correction_injected"))
        if observed_sequence != expected_sequence:
            errors.append(
                {
                    "sequence": expected_sequence,
                    "reason": "nonconsecutive audit sequence",
                    "observed": observed_sequence,
                }
            )
        if observed_injected != pending:
            errors.append(
                {
                    "sequence": expected_sequence,
                    "reason": "correction state did not equal prior failure state",
                    "observed": observed_injected,
                    "expected": pending,
                }
            )
        if observed_injected:
            injected += 1
        if event == "generation_batch_failed":
            failed += 1
            reason = str(item.get("last_reason", ""))
            if not reason:
                errors.append(
                    {
                        "sequence": expected_sequence,
                        "reason": "failed batch has no compact reason",
                    }
                )
            elif correction_contains_supervision_metrics(reason):
                errors.append(
                    {
                        "sequence": expected_sequence,
                        "reason": "failed-batch memory contains supervision",
                    }
                )
            pending = True
        elif event == "generation_batch_succeeded":
            succeeded += 1
            if observed_injected:
                recovered += 1
            pending = False
        else:
            errors.append(
                {
                    "sequence": expected_sequence,
                    "reason": "unknown retry-audit event",
                    "observed": event,
                }
            )
    if errors:
        raise RuntimeError(f"V17 retry state-machine audit failed: {errors}")
    return {
        "status": "pass",
        "generation_batch_calls": len(records),
        "failed_generation_batches": failed,
        "successful_generation_batches": succeeded,
        "correction_injections": injected,
        "successful_correction_recoveries": recovered,
        "trailing_failure_pending": pending,
        "one_failed_batch_memory_only": True,
        "memory_cleared_after_success": True,
        "accuracy_or_split_supervision_present": False,
        "selection_contract_changed": False,
        "candidate_budget_changed": False,
    }


def main() -> int:
    validate_v17_contract()
    validate_v17_source_manifest()
    previous = (
        base.V16_CONTRACT,
        base.V16_POLICY,
        base.install_v16_candidate,
        base.validate_v16_contract,
        base.validate_v16_source_manifest,
        base.V16_SOURCE_FILES,
        base._prompt_audit,
    )
    try:
        base.V16_CONTRACT = V17_CONTRACT
        base.V16_POLICY = V17_POLICY
        base.install_v16_candidate = install_v17_candidate
        base.validate_v16_contract = validate_v17_contract
        base.validate_v16_source_manifest = validate_v17_source_manifest
        base.V16_SOURCE_FILES = V17_SOURCE_FILES
        base._prompt_audit = _v17_prompt_audit
        code = base.main()
    finally:
        (
            base.V16_CONTRACT,
            base.V16_POLICY,
            base.install_v16_candidate,
            base.validate_v16_contract,
            base.validate_v16_source_manifest,
            base.V16_SOURCE_FILES,
            base._prompt_audit,
        ) = previous
    if code:
        return int(code)

    args = base.arguments()
    result_path = Path(args.output) / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    retry_audit = get_last_retry_audit()
    retry_summary = validate_retry_audit(retry_audit)
    if retry_summary["trailing_failure_pending"]:
        raise RuntimeError("V17 completed with uncleared failed-batch state")
    expected_candidates = V17_CONTRACT.population * (V17_CONTRACT.generations + 1)
    if retry_summary["successful_generation_batches"] < expected_candidates:
        raise RuntimeError(
            "V17 retry audit has fewer successful generation batches than "
            "the required candidate budget"
        )
    recorded_source = result.pop("v16_source_sha256", None)
    current_source = _source_hashes()
    if recorded_source != current_source:
        raise RuntimeError("V17 source changed during the child search")
    if int(result.get("candidate_evaluations", -1)) != expected_candidates:
        raise RuntimeError("V17 did not complete the exact 110-candidate budget")
    result.update(
        {
            "status": STATUS,
            "method_id": V17_POLICY.method_id,
            "method_status": "v17_cross_batch_retry_p10g10_i1_i6_confirmation",
            "policy": V17_POLICY.to_dict(),
            "method_contract": V17_CONTRACT.to_dict(),
            "cross_batch_retry_audit": retry_audit,
            "cross_batch_retry_summary": retry_summary,
            "cross_batch_retry_used_for_selection": False,
            "cross_batch_retry_contains_accuracy_metrics": False,
            "cross_batch_retry_contains_validation_or_test_data": False,
            "v17_source_sha256": current_source,
        }
    )
    base.engine.write_json(result_path, result)
    metrics = result["selected_training_metrics"]
    print(
        f"I{args.intersection} V17 P10/G10 complete: "
        f"R2={metrics['macro_raw_r2']:.6f}, "
        f"failed_batches={retry_summary['failed_generation_batches']}, "
        f"recovered={retry_summary['successful_correction_recoveries']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
