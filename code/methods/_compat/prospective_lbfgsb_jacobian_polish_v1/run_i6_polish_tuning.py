"""Tune deterministic Jacobian polish budgets on I6 validation only."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]
REVIEW_ROOT = GMINI / "reviewer_revision_experiments"
for path in (GMINI, REVIEW_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evolution_matrix import runner  # noqa: E402
from evolution_matrix.tune_i6_validation_only import (  # noqa: E402
    SOURCE,
    build_train_bundle,
    evaluate_split,
    flatten_validation_metrics,
    load_selected_source,
    physical_errors,
    read_json,
)
from methods.prospective_lbfgsb_jacobian_polish_v1 import (  # noqa: E402
    ParallelApproachPolisher,
)
from optimization_lane import prepare_optimization_context  # noqa: E402


RUN_IDS = tuple(range(1, 11))
POLISH_BUDGETS = (50, 100, 200)
BASE_RESULTS = (
    GMINI
    / "methods"
    / "prospective_lbfgsb_jacobian_v1"
    / "experiments"
    / "i6_paired_full10"
)
OUTPUT = HERE / "experiments" / "i6_polish_tuning_v1"
METRICS = (
    "validation_macro_r2",
    "validation_macro_rmse",
    "validation_macro_mae",
    "validation_macro_mape",
)
RMSE_TIE = 0.001
MAX_RUN_RMSE_RELATIVE_DEGRADATION = 0.005


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_test(
    baseline: Sequence[float],
    candidate: Sequence[float],
) -> dict[str, Any]:
    difference = np.asarray(candidate) - np.asarray(baseline)
    t_result = ttest_rel(candidate, baseline)
    wilcoxon_p = (
        1.0
        if np.allclose(difference, 0.0)
        else float(wilcoxon(candidate, baseline).pvalue)
    )
    return {
        "ttest_p": float(t_result.pvalue),
        "wilcoxon_p": wilcoxon_p,
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = read_json(SOURCE / "plan_manifest.json")
    core = runner._import_core()
    bundle = build_train_bundle(core, plan)
    prepared = prepare_optimization_context(
        bundle["fit"],
        bundle["lanes"],
        bundle["lane_to_approach"],
        6,
    )
    rows = []
    base_records: dict[int, Mapping[str, Any]] = {}
    for run_id in RUN_IDS:
        base_records[run_id] = read_json(
            BASE_RESULTS / f"run_{run_id:03d}.json"
        )

    with ParallelApproachPolisher() as polisher:
        for position, run_id in enumerate(RUN_IDS):
            source = load_selected_source(run_id)
            base_record = base_records[run_id]
            base = base_record["jacobian"]
            base_metrics = flatten_validation_metrics(
                base["validation_metrics"]
            )
            rotation = position % len(POLISH_BUDGETS)
            ordered_budgets = (
                POLISH_BUDGETS[rotation:] + POLISH_BUDGETS[:rotation]
            )
            for budget in ordered_budgets:
                diagnostics: dict[str, Any] = {}
                started = time.perf_counter()
                parameters = polisher(
                    polish_maxiter=budget,
                    universal_expr=source["expression"],
                    df=bundle["fit"],
                    lanes=bundle["lanes"],
                    lane_to_approach=bundle["lane_to_approach"],
                    approach_targets=bundle["fit_targets"],
                    intersection_id=6,
                    lane_parameters=base["parameters"],
                    prepared_context=prepared,
                    diagnostics=diagnostics,
                )
                polish_wall = float(time.perf_counter() - started)
                validation_metrics, _ = evaluate_split(
                    core,
                    bundle,
                    source["expression"],
                    parameters,
                    "validation",
                    run_id,
                    f"cosydelay_jacobian_polish_{budget}",
                )
                current_metrics = flatten_validation_metrics(
                    validation_metrics
                )
                errors = physical_errors(
                    source["expression"],
                    parameters,
                    bundle["lanes"],
                )
                base_wall = float(base["fit_wall_seconds"])
                official_wall = float(
                    base_record["baseline"]["fit_wall_seconds"]
                )
                row = {
                    "run_id": run_id,
                    "polish_maxiter": budget,
                    "physical_joint_pass": not bool(errors),
                    "base_objective_sum": base[
                        "optimizer_diagnostics"
                    ]["objective_sum"],
                    "polish_start_objective_sum": diagnostics[
                        "objective_sum_before"
                    ],
                    "polished_objective_sum": diagnostics[
                        "objective_sum_after"
                    ],
                    "objective_improvement": diagnostics[
                        "objective_improvement"
                    ],
                    "accepted_approaches": diagnostics[
                        "accepted_approaches"
                    ],
                    "polish_iterations": diagnostics[
                        "total_iterations"
                    ],
                    "polish_function_evaluations": diagnostics[
                        "total_function_evaluations"
                    ],
                    "polish_wall_seconds": polish_wall,
                    "base_wall_seconds": base_wall,
                    "estimated_total_wall_seconds": base_wall + polish_wall,
                    "official_wall_seconds": official_wall,
                    "estimated_speedup_vs_official": (
                        official_wall / (base_wall + polish_wall)
                    ),
                    "rng_values_consumed": diagnostics[
                        "rng_values_consumed"
                    ],
                }
                for metric in METRICS:
                    row[f"base_{metric}"] = base_metrics[metric]
                    row[f"polished_{metric}"] = current_metrics[metric]
                    row[f"delta_{metric}"] = (
                        current_metrics[metric] - base_metrics[metric]
                    )
                row["relative_delta_validation_macro_rmse"] = (
                    row["delta_validation_macro_rmse"]
                    / row["base_validation_macro_rmse"]
                )
                rows.append(row)
                result = {
                    "run_id": run_id,
                    "polish_maxiter": budget,
                    "source_config_hash": source["source_config_hash"],
                    "expression": source["expression"],
                    "parameters": parameters,
                    "optimizer_diagnostics": diagnostics,
                    "validation_metrics": validation_metrics,
                    "physical_errors": errors,
                    "comparison": row,
                    "test_file_opened": False,
                    "test_used_for_selection": False,
                }
                result_dir = (
                    OUTPUT
                    / f"run_{run_id:03d}"
                    / f"polish_{budget:03d}"
                )
                _write_json(result_dir / "validation_result.json", result)
                _write_csv(OUTPUT / "per_run.csv", rows)
                print(
                    f"run {run_id:02d} p{budget:03d}: "
                    f"dRMSE={row['delta_validation_macro_rmse']:+.6f}, "
                    f"objective_gain={row['objective_improvement']:.6g}, "
                    f"added_wall={polish_wall:.2f}s",
                    flush=True,
                )

    candidate_summaries = []
    for budget in POLISH_BUDGETS:
        selected = [
            row for row in rows if row["polish_maxiter"] == budget
        ]
        base_rmse = np.asarray(
            [row["base_validation_macro_rmse"] for row in selected]
        )
        polished_rmse = np.asarray(
            [row["polished_validation_macro_rmse"] for row in selected]
        )
        estimated_total = float(
            np.mean(
                [row["estimated_total_wall_seconds"] for row in selected]
            )
        )
        official_wall = float(
            np.mean([row["official_wall_seconds"] for row in selected])
        )
        summary = {
            "polish_maxiter": budget,
            "runs": len(selected),
            "physical_passes": int(
                sum(row["physical_joint_pass"] for row in selected)
            ),
            "rng_values_consumed": int(
                sum(row["rng_values_consumed"] for row in selected)
            ),
            "objective_improvement_mean": float(
                np.mean(
                    [row["objective_improvement"] for row in selected]
                )
            ),
            "objective_improvement_runs": int(
                sum(row["objective_improvement"] > 0 for row in selected)
            ),
            "accepted_approaches_total": int(
                sum(row["accepted_approaches"] for row in selected)
            ),
            "polish_wall_seconds_mean": float(
                np.mean([row["polish_wall_seconds"] for row in selected])
            ),
            "estimated_total_wall_seconds_mean": estimated_total,
            "official_wall_seconds_mean": official_wall,
            "estimated_speedup_vs_official": (
                official_wall / estimated_total
            ),
            "base_validation_macro_rmse_mean": float(
                np.mean(base_rmse)
            ),
            "polished_validation_macro_rmse_mean": float(
                np.mean(polished_rmse)
            ),
            "delta_validation_macro_rmse_mean": float(
                np.mean(polished_rmse - base_rmse)
            ),
            "max_run_relative_macro_rmse_degradation": float(
                max(
                    row["relative_delta_validation_macro_rmse"]
                    for row in selected
                )
            ),
        }
        for metric in METRICS:
            base_values = [
                row[f"base_{metric}"] for row in selected
            ]
            polished_values = [
                row[f"polished_{metric}"] for row in selected
            ]
            summary[f"base_{metric}_mean"] = float(
                np.mean(base_values)
            )
            summary[f"polished_{metric}_mean"] = float(
                np.mean(polished_values)
            )
            summary[f"delta_{metric}_mean"] = float(
                np.mean(np.asarray(polished_values) - base_values)
            )
        summary["eligible"] = bool(
            summary["physical_passes"] == len(RUN_IDS)
            and summary["delta_validation_macro_rmse_mean"] < 0.0
            and summary["max_run_relative_macro_rmse_degradation"]
            <= MAX_RUN_RMSE_RELATIVE_DEGRADATION
            and estimated_total <= official_wall
            and summary["rng_values_consumed"] == 0
        )
        candidate_summaries.append(summary)

    eligible = [
        item for item in candidate_summaries if item["eligible"]
    ]
    selected_candidate = None
    if eligible:
        best_rmse = min(
            item["polished_validation_macro_rmse_mean"]
            for item in eligible
        )
        tied = [
            item
            for item in eligible
            if item["polished_validation_macro_rmse_mean"]
            <= best_rmse + RMSE_TIE
        ]
        selected_candidate = min(
            tied,
            key=lambda item: item["polish_wall_seconds_mean"],
        )

    statistical_tests = {}
    if selected_candidate is not None:
        budget = selected_candidate["polish_maxiter"]
        selected_rows = [
            row for row in rows if row["polish_maxiter"] == budget
        ]
        for metric in METRICS:
            statistical_tests[metric] = _paired_test(
                [row[f"base_{metric}"] for row in selected_rows],
                [row[f"polished_{metric}"] for row in selected_rows],
            )
    summary = {
        "schema_version": 1,
        "experiment": "i6_deterministic_best_restart_polish_tuning",
        "candidates": candidate_summaries,
        "selection_rule": (
            "among eligible candidates, lowest mean validation macro RMSE; "
            "within 0.001 choose lower added wall"
        ),
        "selected_candidate": selected_candidate,
        "decision": (
            "retain_selected_polish"
            if selected_candidate is not None
            else "reject_polish"
        ),
        "selected_candidate_paired_tests": statistical_tests,
        "base_results": str(BASE_RESULTS.resolve()),
        "code_sha256": {
            "polish_fitter.py": _sha256(HERE / "polish_fitter.py"),
            "run_i6_polish_tuning.py": _sha256(
                Path(__file__).resolve()
            ),
            "method_config.json": _sha256(HERE / "method_config.json"),
            "protocol.md": _sha256(HERE / "PROTOCOL.md"),
        },
        "test_file_opened": False,
        "test_used_for_selection": False,
        "eligible_as_new_locked_test_evidence": False,
    }
    _write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
