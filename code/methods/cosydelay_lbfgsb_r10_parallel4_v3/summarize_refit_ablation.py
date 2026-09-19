"""Aggregate the predeclared four-fold, Training-only refit ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE / "experiments"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        default="training_only_refit_ablation_i1_i4_i6_v7_r9feasible_fold",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENTS
        / "training_only_refit_ablation_i1_i4_i6_v7_r9feasible_cv4_summary.json",
    )
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _arguments()
    cases = []
    source_summaries = []
    for fold in range(4):
        source = (
            EXPERIMENTS
            / f"{args.prefix}{fold}"
            / "summary.json"
        )
        summary = json.loads(source.read_text(encoding="utf-8"))
        if summary["test_file_opened"] or summary["outer_validation_accessed"]:
            raise RuntimeError(f"non-Training data access reported by {source}")
        source_summaries.append(str(source.resolve()))
        for row in summary["rows"]:
            delta = row["delta_v3_minus_baseline"]
            baseline_rmse = float(row["baseline"]["metrics"]["macro_rmse"])
            cases.append(
                {
                    "fold": fold,
                    "intersection_id": int(row["intersection_id"]),
                    "delta_r2": float(delta["macro_r2"]),
                    "delta_rmse": float(delta["macro_rmse"]),
                    "delta_mae": float(delta["macro_mae"]),
                    "relative_rmse_delta_percent": (
                        100.0 * float(delta["macro_rmse"]) / baseline_rmse
                    ),
                    "delta_wall_seconds": float(delta["wall_seconds"]),
                    "enhanced_physics_pass": bool(
                        row["v3"]["enhanced_physics"]["joint_pass"]
                    ),
                }
            )

    def mean(field: str, rows=cases) -> float:
        return float(np.mean([float(row[field]) for row in rows]))

    by_intersection = {}
    for intersection_id in sorted({row["intersection_id"] for row in cases}):
        rows = [row for row in cases if row["intersection_id"] == intersection_id]
        by_intersection[str(intersection_id)] = {
            "cases": len(rows),
            "mean_delta_r2": mean("delta_r2", rows),
            "mean_delta_rmse": mean("delta_rmse", rows),
            "mean_delta_mae": mean("delta_mae", rows),
            "mean_delta_wall_seconds": mean("delta_wall_seconds", rows),
            "worst_relative_rmse_delta_percent": float(
                max(row["relative_rmse_delta_percent"] for row in rows)
            ),
            "physics_passes": sum(row["enhanced_physics_pass"] for row in rows),
        }

    aggregate = {
        "cases": len(cases),
        "mean_delta_r2": mean("delta_r2"),
        "mean_delta_rmse": mean("delta_rmse"),
        "mean_delta_mae": mean("delta_mae"),
        "mean_delta_wall_seconds": mean("delta_wall_seconds"),
        "worst_relative_rmse_delta_percent": float(
            max(row["relative_rmse_delta_percent"] for row in cases)
        ),
        "enhanced_physics_passes": sum(
            row["enhanced_physics_pass"] for row in cases
        ),
    }
    checks = {
        "mean_r2_nondecreasing": aggregate["mean_delta_r2"] >= 0.0,
        "mean_rmse_nonincreasing": aggregate["mean_delta_rmse"] <= 0.0,
        "mean_mae_nonincreasing": aggregate["mean_delta_mae"] <= 0.0,
        "all_enhanced_physics_pass": (
            aggregate["enhanced_physics_passes"] == aggregate["cases"]
        ),
        "no_case_over_0_5pct_relative_rmse_degradation": (
            aggregate["worst_relative_rmse_delta_percent"] <= 0.5
        ),
    }
    result = {
        "experiment": "training_only_fixed_expression_refit_ablation_cv4",
        "method_status": "prospective_not_official",
        "test_file_opened": False,
        "outer_validation_accessed": False,
        "source_summaries": source_summaries,
        "cases": cases,
        "aggregate": aggregate,
        "by_intersection": by_intersection,
        "predeclared_gate_checks": checks,
        "coefficient_component_gate": "pass" if all(checks.values()) else "fail",
        "next_gate": "whole_search_training_only_paired_runs",
    }
    _write_json(args.output, result)
    print(json.dumps(result["aggregate"], ensure_ascii=False))
    print(f"coefficient_component_gate={result['coefficient_component_gate']}")
    print(args.output)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
