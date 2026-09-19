"""Summarize matched-fold Training-only P4/G2 evidence for v2 versus v3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE / "experiments"
OUTPUT = EXPERIMENTS / "whole_search_p4g2_paired_i1_i4_i6_2seeds_summary.json"
SEEDS = (20260803, 20260805)
INTERSECTIONS = (1, 4, 6)


def _v3_result_path(intersection: int, seed: int) -> Path:
    suffix = "incumbent_v1" if intersection == 4 else "finalfast_v1"
    return (
        EXPERIMENTS
        / f"whole_search_p4g2_i{intersection}_seed{seed}_{suffix}"
        / "result.json"
    )


def _baseline_path(intersection: int, seed: int) -> Path:
    if intersection == 4:
        return (
            EXPERIMENTS
            / f"baseline_oof_i4_archived_run03_reference_seed{seed}.json"
        )
    return EXPERIMENTS / f"baseline_oof_i{intersection}_seed{seed}.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _mean(rows: list[dict], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def main() -> int:
    pairs = []
    for intersection in INTERSECTIONS:
        for seed in SEEDS:
            baseline_path = _baseline_path(intersection, seed)
            v3_path = _v3_result_path(intersection, seed)
            baseline = _read(baseline_path)
            v3 = _read(v3_path)
            source = _read(Path(baseline["source_result"]))
            if baseline["test_file_opened"] or v3["test_file_opened"]:
                raise RuntimeError("Test access reported by a paired artifact")
            if v3["outer_validation_accessed"]:
                raise RuntimeError("outer Validation access reported by v3")
            old = baseline["oof_metrics"]
            new = v3["selection_report"]["selected_oof_metrics"]
            v2_pipeline_wall = float(source["wall_seconds"]) + float(
                baseline["wall_seconds"]
            )
            pairs.append(
                {
                    "intersection_id": intersection,
                    "seed": seed,
                    "fold_assignments_source": str(v3_path.resolve()),
                    "v2_source_result": baseline["source_result"],
                    "v2_expression": baseline["expression"],
                    "v3_expression": v3["expression"],
                    "v3_selected_candidate_source": v3["selection_report"].get(
                        "selected_candidate_source", "generated_v3"
                    ),
                    "v3_incumbent_cv_slot_reserved": bool(
                        v3.get("incumbent_cv_slot_reserved", False)
                    ),
                    "v2_oof_r2": float(old["macro_r2"]),
                    "v3_oof_r2": float(new["macro_r2"]),
                    "delta_r2": float(new["macro_r2"] - old["macro_r2"]),
                    "v2_oof_rmse": float(old["macro_rmse"]),
                    "v3_oof_rmse": float(new["macro_rmse"]),
                    "delta_rmse": float(new["macro_rmse"] - old["macro_rmse"]),
                    "relative_rmse_delta_percent": float(
                        100.0
                        * (new["macro_rmse"] - old["macro_rmse"])
                        / old["macro_rmse"]
                    ),
                    "v2_oof_mae": float(old["macro_mae"]),
                    "v3_oof_mae": float(new["macro_mae"]),
                    "delta_mae": float(new["macro_mae"] - old["macro_mae"]),
                    "relative_mae_delta_percent": float(
                        100.0
                        * (new["macro_mae"] - old["macro_mae"])
                        / old["macro_mae"]
                    ),
                    "v2_all_folds_enhanced_physics": bool(
                        baseline["all_folds_enhanced_physics"]
                    ),
                    "v3_final_enhanced_physics": bool(
                        v3["selection_report"]["selected_enhanced_physics"][
                            "joint_pass"
                        ]
                    ),
                    "v2_search_wall_seconds": float(source["wall_seconds"]),
                    "v2_reference_oof_wall_seconds": float(
                        baseline["wall_seconds"]
                    ),
                    "v2_comparable_pipeline_wall_seconds": v2_pipeline_wall,
                    "v3_pipeline_wall_seconds": float(v3["wall_seconds"]),
                    "relative_pipeline_wall_delta_percent": float(
                        100.0
                        * (float(v3["wall_seconds"]) - v2_pipeline_wall)
                        / v2_pipeline_wall
                    ),
                }
            )

    aggregate = {
        "pairs": len(pairs),
        "mean_v2_oof_r2": _mean(pairs, "v2_oof_r2"),
        "mean_v3_oof_r2": _mean(pairs, "v3_oof_r2"),
        "mean_delta_r2": _mean(pairs, "delta_r2"),
        "mean_v2_oof_rmse": _mean(pairs, "v2_oof_rmse"),
        "mean_v3_oof_rmse": _mean(pairs, "v3_oof_rmse"),
        "mean_delta_rmse": _mean(pairs, "delta_rmse"),
        "mean_relative_rmse_delta_percent": _mean(
            pairs, "relative_rmse_delta_percent"
        ),
        "mean_v2_oof_mae": _mean(pairs, "v2_oof_mae"),
        "mean_v3_oof_mae": _mean(pairs, "v3_oof_mae"),
        "mean_delta_mae": _mean(pairs, "delta_mae"),
        "mean_relative_mae_delta_percent": _mean(
            pairs, "relative_mae_delta_percent"
        ),
        "v2_enhanced_physics_pairs": sum(
            row["v2_all_folds_enhanced_physics"] for row in pairs
        ),
        "v3_enhanced_physics_pairs": sum(
            row["v3_final_enhanced_physics"] for row in pairs
        ),
        "mean_v2_comparable_pipeline_wall_seconds": _mean(
            pairs, "v2_comparable_pipeline_wall_seconds"
        ),
        "mean_v3_pipeline_wall_seconds": _mean(
            pairs, "v3_pipeline_wall_seconds"
        ),
        "pipeline_wall_ratio_of_means_percent": float(
            100.0
            * _mean(pairs, "v3_pipeline_wall_seconds")
            / _mean(pairs, "v2_comparable_pipeline_wall_seconds")
        ),
        "relative_pipeline_wall_delta_ratio_of_means_percent": float(
            100.0
            * (
                _mean(pairs, "v3_pipeline_wall_seconds")
                / _mean(pairs, "v2_comparable_pipeline_wall_seconds")
                - 1.0
            )
        ),
        "mean_relative_pipeline_wall_delta_percent": _mean(
            pairs, "relative_pipeline_wall_delta_percent"
        ),
    }
    by_intersection = {}
    for intersection in INTERSECTIONS:
        rows = [row for row in pairs if row["intersection_id"] == intersection]
        by_intersection[str(intersection)] = {
            "runs": len(rows),
            "mean_delta_r2": _mean(rows, "delta_r2"),
            "mean_relative_rmse_delta_percent": _mean(
                rows, "relative_rmse_delta_percent"
            ),
            "mean_relative_mae_delta_percent": _mean(
                rows, "relative_mae_delta_percent"
            ),
            "v2_enhanced_physics_runs": sum(
                row["v2_all_folds_enhanced_physics"] for row in rows
            ),
            "v3_enhanced_physics_runs": sum(
                row["v3_final_enhanced_physics"] for row in rows
            ),
        }
    descriptive_checks = {
        "mean_r2_improves": aggregate["mean_delta_r2"] > 0.0,
        "mean_rmse_improves": aggregate["mean_delta_rmse"] < 0.0,
        "mean_mae_improves": aggregate["mean_delta_mae"] < 0.0,
        "no_pair_rmse_degrades": max(
            row["relative_rmse_delta_percent"] for row in pairs
        ) <= 0.0,
        "all_v3_final_physics_pass": (
            aggregate["v3_enhanced_physics_pairs"] == len(pairs)
        ),
    }
    result = {
        "experiment": "training_only_p4g2_paired_reference_oof",
        "method_status": "prospective_not_official",
        "comparison_note": (
            "Both methods use Training-only structure selection. Archived v2 "
            "expressions are refit on the exact v3 fold assignments; these are "
            "development OOF results, not untouched-holdout claims."
        ),
        "test_file_opened": False,
        "outer_validation_accessed_by_v3": False,
        "pairs": pairs,
        "aggregate": aggregate,
        "by_intersection": by_intersection,
        "descriptive_checks_not_preregistered": descriptive_checks,
        "all_descriptive_checks_pass": all(descriptive_checks.values()),
    }
    _write(OUTPUT, result)
    print(json.dumps(aggregate, ensure_ascii=False))
    print(f"all_descriptive_checks_pass={result['all_descriptive_checks_pass']}")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
