"""Reclassify fixed fitted V14 candidates under alternative R9 policies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping


ARMS = {
    "numeric_10000_seconds": 10_000.0,
    "numeric_1000_seconds": 1_000.0,
    "symbolic_positive_infinity_only": None,
}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def probe_delays(record: Mapping[str, Any]) -> dict[str, float]:
    diagnostics = record.get("details", {}).get("verifier", {}).get("diagnostics", {})
    numerical = diagnostics.get("zero_green_symbolic_proof", {}).get(
        "numerical_probe", {}
    )
    lanes = numerical.get("delays_by_lane", {}) or {}
    values = {}
    for lane, item in lanes.items():
        try:
            value = float(item["low_green_delay_seconds"])
        except (KeyError, TypeError, ValueError):
            value = math.nan
        values[str(lane)] = value
    return values


def non_r9_pass(enhanced: Mapping[str, Any]) -> bool:
    scores = enhanced.get("standard_rule_scores", {}) or {}
    other_standard = all(
        float(value) >= 1.0 - 1e-12
        for name, value in scores.items()
        if str(name) != "R9_zero_green_limit"
    )
    return bool(
        other_standard
        and enhanced.get("symbolic_r8_exact")
        and enhanced.get("symbolic_r9_positive_infinity")
        and enhanced.get("dense_finite")
        and enhanced.get("dense_nonnegative")
        and enhanced.get("dense_flow_nondecreasing")
        and enhanced.get("dense_green_nonincreasing")
    )


def arm_pass(record: Mapping[str, Any], threshold: float | None) -> bool:
    enhanced = record.get("enhanced_physics", {}) or {}
    if not non_r9_pass(enhanced):
        return False
    if threshold is None:
        return True
    values = probe_delays(record)
    return bool(
        values
        and all(
            math.isinf(value) or (math.isfinite(value) and value > threshold)
            for value in values.values()
        )
    )


def candidate_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", {}) or {}
    values = probe_delays(record)
    finite = [value for value in values.values() if math.isfinite(value)]
    return {
        "expression": record.get("expression"),
        "macro_raw_r2": metrics.get("macro_raw_r2"),
        "macro_nonnegative_r2": metrics.get("macro_nonnegative_r2"),
        "pooled_rmse": metrics.get("pooled_rmse"),
        "pooled_mae": metrics.get("pooled_mae"),
        "symbolic_r9_positive_infinity": bool(
            record.get("enhanced_physics", {}).get("symbolic_r9_positive_infinity")
        ),
        "non_r9_physics_pass": non_r9_pass(record.get("enhanced_physics", {})),
        "minimum_finite_probe_delay_seconds": min(finite) if finite else None,
        "probe_lane_count": len(values),
        "passes": {
            name: arm_pass(record, threshold) for name, threshold in ARMS.items()
        },
    }


def analyze_run(run: Path) -> dict[str, Any]:
    result = read(run / "result.json")
    rejected = read(run / "fitted_rejections.json")
    winner = {
        "expression": result["selected_expression"],
        "parameters": result["selected_parameters"],
        "metrics": result["selected_training_metrics"],
        "enhanced_physics": result["selected_enhanced_physics"],
        "details": {
            "verifier": {
                "diagnostics": result["selected_enhanced_physics"].get(
                    "standard_diagnostics", {}
                )
            }
        },
    }
    records = [candidate_summary(item) for item in rejected]
    arms = {}
    for name in ARMS:
        eligible = [item for item in records if item["passes"][name]]
        best = max(
            eligible,
            key=lambda item: (
                float(item.get("macro_nonnegative_r2") or -math.inf),
                float(item.get("macro_raw_r2") or -math.inf),
                -float(item.get("pooled_rmse") or math.inf),
            ),
            default=None,
        )
        arms[name] = {
            "eligible_rejected_candidates": len(eligible),
            "rescued_from_current_10000_arm": sum(
                item["passes"][name]
                and not item["passes"]["numeric_10000_seconds"]
                for item in records
            ),
            "best_eligible_rejected_candidate": best,
        }
    r9_only = [item for item in records if item["non_r9_physics_pass"]]
    return {
        "intersection_id": result["intersection_id"],
        "run": str(run.resolve()),
        "selected_training_metrics": result["selected_training_metrics"],
        "selected_expression": result["selected_expression"],
        "fixed_fitted_rejection_records": len(records),
        "rejections_passing_every_non_r9_check_and_symbolic_r9": len(r9_only),
        "arms": arms,
        "candidate_records": records,
        "limitation": (
            "Only fitted rejection records retain every candidate's fitted parameters. "
            "This reclassifies those fixed candidates without refitting and does not "
            "reconstruct unselected surviving-population candidates."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyses = [analyze_run(path.resolve()) for path in args.run]
    aggregate = {}
    for name in ARMS:
        aggregate[name] = {
            "eligible_rejected_candidates": sum(
                item["arms"][name]["eligible_rejected_candidates"] for item in analyses
            ),
            "rescued_from_current_10000_arm": sum(
                item["arms"][name]["rescued_from_current_10000_arm"] for item in analyses
            ),
        }
    output = {
        "schema_version": 1,
        "status": "completed_fixed_fitted_candidate_r9_ablation",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "data_scope": "V14 Training-only fitted candidates; no Validation/Test access",
        "arms": {
            name: {
                "symbolic_positive_infinity_required": True,
                "numerical_probe_threshold_seconds": threshold,
                "probe_green_ratio": 1e-6 if threshold is not None else None,
            }
            for name, threshold in ARMS.items()
        },
        "aggregate": aggregate,
        "runs": analyses,
    }
    write(args.output.resolve(), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
