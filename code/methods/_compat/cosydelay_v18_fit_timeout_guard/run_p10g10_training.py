"""Run one frozen V18 P10/G10 Training-only search from scratch."""

from __future__ import annotations

import json
from pathlib import Path

from methods.cosydelay_v17_cross_batch_retry import (
    run_p10g10_training as base,
)

from .contract import V18_CONTRACT, validate_v18_contract
from .integration import get_last_fit_guard_audit, install_v18_candidate
from .policy import V18_POLICY
from .source_manifest import (
    GMINI,
    V18_SOURCE_FILES,
    validate_v18_source_manifest,
)


STATUS = "completed_v18_p10g10_training_only_search"


def _source_hashes() -> dict[str, str]:
    return {
        (
            str(path.relative_to(GMINI)).replace("\\", "/")
            if path.is_relative_to(GMINI)
            else str(path)
        ): base.base.sha256_file(path)
        for path in V18_SOURCE_FILES
    }


def _cleanup_still_alive(event: dict) -> list[int]:
    cleanup = event.get("cleanup") or {}
    tree = cleanup.get("tree_cleanup") or {}
    return [int(pid) for pid in tree.get("still_alive_pids", [])]


def validate_fit_guard_audit(
    audit: dict, *, expected_candidates: int = V18_CONTRACT.successful_candidate_budget
) -> dict:
    """Recompute the transactional fit and successful-budget invariants."""
    fit_events = audit.get("fit_events")
    candidate_events = audit.get("candidate_events")
    prefit_r7_events = audit.get("prefit_r7_events")
    errors: list[dict] = []
    if not isinstance(fit_events, list):
        raise RuntimeError("V18 fit guard audit is missing fit_events")
    if not isinstance(candidate_events, list):
        raise RuntimeError("V18 fit guard audit is missing candidate_events")
    if not isinstance(prefit_r7_events, list):
        raise RuntimeError("V18 fit guard audit is missing prefit_r7_events")
    terminal_fit_events = [
        event
        for event in fit_events
        if event.get("event")
        in {"fit_succeeded", "fit_timeout", "fit_worker_error"}
    ]
    observed_request_ids = [
        int(event.get("request_id", -1)) for event in terminal_fit_events
    ]
    required_request_ids = list(range(1, len(terminal_fit_events) + 1))
    if observed_request_ids != required_request_ids:
        errors.append(
            {
                "reason": "fit request ids are not consecutive",
                "observed": observed_request_ids,
                "required": required_request_ids,
            }
        )
    fit_successes = sum(
        event.get("event") == "fit_succeeded" for event in terminal_fit_events
    )
    fit_timeouts = sum(
        event.get("event") == "fit_timeout" for event in terminal_fit_events
    )
    fit_worker_errors = sum(
        event.get("event") == "fit_worker_error"
        for event in terminal_fit_events
    )
    starts = [event for event in fit_events if event.get("event") == "supervisor_started"]
    detachments = [
        event for event in fit_events if event.get("event") == "supervisor_detached"
    ]
    if not starts:
        errors.append({"reason": "no fitting supervisor startup was recorded"})
    if not any(event.get("reason") == "normal_close" for event in detachments):
        errors.append({"reason": "bounded normal supervisor close was not recorded"})
    for event in terminal_fit_events:
        if len(str(event.get("expression_sha256", ""))) != 64:
            errors.append(
                {
                    "reason": "fit event lacks expression hash",
                    "request_id": event.get("request_id"),
                }
            )
        if event.get("event") in {"fit_timeout", "fit_worker_error"} and event.get(
            "optimizer_rng_state_committed"
        ) is not False:
            errors.append(
                {
                    "reason": "failed fit committed optimizer RNG state",
                    "request_id": event.get("request_id"),
                }
            )
    for event in detachments:
        alive = _cleanup_still_alive(event)
        if alive:
            errors.append(
                {
                    "reason": "supervisor cleanup left live descendants",
                    "pids": alive,
                    "generation": event.get("supervisor_generation"),
                }
            )

    accepted = [
        event
        for event in candidate_events
        if event.get("event") == "candidate_accepted"
    ]
    rejected = [
        event
        for event in candidate_events
        if event.get("event") == "candidate_rejected"
    ]
    if len(accepted) != int(expected_candidates):
        errors.append(
            {
                "reason": "successful candidate budget mismatch",
                "observed": len(accepted),
                "required": int(expected_candidates),
            }
        )
    rejected_hashes = {event.get("canonical_sha256") for event in rejected}
    accepted_hashes = {event.get("canonical_sha256") for event in accepted}
    overlap = sorted(str(value) for value in rejected_hashes & accepted_hashes)
    if overlap:
        errors.append(
            {
                "reason": "a rejected canonical expression later entered the budget",
                "canonical_sha256": overlap,
            }
        )
    if len(rejected) < fit_timeouts + fit_worker_errors:
        errors.append(
            {
                "reason": "fit failure lacks candidate rejection",
                "candidate_rejections": len(rejected),
                "fit_timeouts": fit_timeouts,
                "fit_worker_errors": fit_worker_errors,
            }
        )
    for event in candidate_events:
        if event.get("accuracy_metrics_present") is not False:
            errors.append({"reason": "runtime rejection audit contains accuracy"})
        if event.get("validation_or_test_data_present") is not False:
            errors.append(
                {"reason": "runtime rejection audit contains Validation/Test data"}
            )
        expected_counted = event.get("event") == "candidate_accepted"
        if event.get("counts_toward_successful_candidate_budget") is not expected_counted:
            errors.append(
                {
                    "reason": "candidate budget flag mismatch",
                    "event": event.get("event"),
                }
            )
    for event in prefit_r7_events:
        if event.get("event") != "prefit_r7_decision":
            errors.append({"reason": "unknown prefit R7 audit event"})
        if event.get("verdict") not in {"proven_finite", "unknown_allow"}:
            errors.append(
                {
                    "reason": "invalid prefit R7 verdict",
                    "verdict": event.get("verdict"),
                }
            )
        expected_pass = event.get("verdict") != "proven_finite"
        if event.get("passed") is not expected_pass:
            errors.append(
                {
                    "reason": "prefit R7 verdict/pass mismatch",
                    "verdict": event.get("verdict"),
                }
            )
        if event.get("accuracy_metrics_present") is not False:
            errors.append({"reason": "prefit R7 audit contains accuracy"})
        if event.get("validation_or_test_data_present") is not False:
            errors.append(
                {"reason": "prefit R7 audit contains Validation/Test data"}
            )
        if event.get("counts_toward_successful_candidate_budget") is not False:
            errors.append({"reason": "prefit R7 event entered candidate budget"})
    accepted_expression_hashes = {
        event.get("expression_sha256") for event in accepted
    }
    prefit_rejected_expression_hashes = {
        event.get("expression_sha256")
        for event in prefit_r7_events
        if event.get("verdict") == "proven_finite"
    }
    prefit_allowed_expression_hashes = {
        event.get("expression_sha256")
        for event in prefit_r7_events
        if event.get("verdict") == "unknown_allow"
    }
    prefit_budget_overlap = sorted(
        str(value)
        for value in (
            accepted_expression_hashes & prefit_rejected_expression_hashes
        )
    )
    if prefit_budget_overlap:
        errors.append(
            {
                "reason": "a proved-finite R7 expression entered the budget",
                "expression_sha256": prefit_budget_overlap,
            }
        )
    missing_prefit_allowance = sorted(
        str(value)
        for value in (
            accepted_expression_hashes - prefit_allowed_expression_hashes
        )
    )
    if missing_prefit_allowance:
        errors.append(
            {
                "reason": "an accepted candidate bypassed the R7 pre-fit gate",
                "expression_sha256": missing_prefit_allowance,
            }
        )
    if errors:
        raise RuntimeError(f"V18 fit guard audit failed: {errors}")
    return {
        "status": "pass",
        "fit_attempts": len(terminal_fit_events),
        "successful_fits": fit_successes,
        "fit_timeouts": fit_timeouts,
        "fit_worker_errors": fit_worker_errors,
        "candidate_evaluation_rejections": len(rejected),
        "prefit_r7_checks": len(prefit_r7_events),
        "prefit_r7_proven_finite_rejections": sum(
            event.get("verdict") == "proven_finite"
            for event in prefit_r7_events
        ),
        "prefit_r7_unknown_allowed": sum(
            event.get("verdict") == "unknown_allow"
            for event in prefit_r7_events
        ),
        "successful_candidates_counted": len(accepted),
        "required_successful_candidates": int(expected_candidates),
        "supervisor_starts": len(starts),
        "supervisor_detachments": len(detachments),
        "remaining_descendant_pids": [],
        "failed_candidate_rng_commits": 0,
        "rejected_candidate_entered_budget": False,
        "prefit_r7_rejection_entered_budget": False,
        "timeout_used_for_fitness": False,
        "accuracy_or_split_supervision_present": False,
    }


def main() -> int:
    validate_v18_contract()
    validate_v18_source_manifest()
    previous = (
        base.V17_CONTRACT,
        base.V17_POLICY,
        base.install_v17_candidate,
        base.validate_v17_contract,
        base.validate_v17_source_manifest,
        base.V17_SOURCE_FILES,
        base.STATUS,
    )
    try:
        base.V17_CONTRACT = V18_CONTRACT
        base.V17_POLICY = V18_POLICY
        base.install_v17_candidate = install_v18_candidate
        base.validate_v17_contract = validate_v18_contract
        base.validate_v17_source_manifest = validate_v18_source_manifest
        base.V17_SOURCE_FILES = V18_SOURCE_FILES
        base.STATUS = STATUS
        code = base.main()
    finally:
        (
            base.V17_CONTRACT,
            base.V17_POLICY,
            base.install_v17_candidate,
            base.validate_v17_contract,
            base.validate_v17_source_manifest,
            base.V17_SOURCE_FILES,
            base.STATUS,
        ) = previous
    if code:
        return int(code)

    args = base.base.arguments()
    result_path = Path(args.output) / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    recorded_source = result.pop("v17_source_sha256", None)
    current_source = _source_hashes()
    if recorded_source != current_source:
        raise RuntimeError("V18 source changed during the child search")
    fit_guard_audit = get_last_fit_guard_audit()
    fit_guard_summary = validate_fit_guard_audit(fit_guard_audit)
    if int(result.get("candidate_evaluations", -1)) != V18_CONTRACT.successful_candidate_budget:
        raise RuntimeError("V18 did not complete the exact 110-success candidate budget")
    result.update(
        {
            "status": STATUS,
            "method_id": V18_POLICY.method_id,
            "method_status": (
                "v18_bounded_fit_conservative_r7_prefit_"
                "p10g10_i1_i6_confirmation"
            ),
            "policy": V18_POLICY.to_dict(),
            "method_contract": V18_CONTRACT.to_dict(),
            "fit_guard_audit": fit_guard_audit,
            "fit_guard_summary": fit_guard_summary,
            "candidate_fit_timeout_used_for_fitness": False,
            "timed_out_or_failed_candidates_counted": False,
            "candidate_fit_timeout_contains_accuracy_metrics": False,
            "candidate_fit_timeout_contains_validation_or_test_data": False,
            "v18_source_sha256": current_source,
        }
    )
    base.base.engine.write_json(result_path, result)
    metrics = result["selected_training_metrics"]
    print(
        f"I{args.intersection} V18 P10/G10 complete: "
        f"R2={metrics['macro_raw_r2']:.6f}, "
        f"fit_timeouts={fit_guard_summary['fit_timeouts']}, "
        f"fit_worker_errors={fit_guard_summary['fit_worker_errors']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
