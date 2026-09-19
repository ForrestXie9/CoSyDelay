"""Run controlled validation-only I6 pairs: frozen fitter vs Jacobian fitter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
from methods.cosydelay_lbfgsb_r10_parallel3 import (  # noqa: E402
    ParallelApproachFitter,
)
from methods.prospective_lbfgsb_jacobian_v1 import (  # noqa: E402
    JacobianParallelApproachFitter,
)
from optimization_lane import prepare_optimization_context  # noqa: E402


INTERSECTION = 6
ALL_RUN_IDS = tuple(range(1, 11))
DEFAULT_OUTPUT = HERE / "experiments" / "i6_paired_pilot"
METRICS = (
    "validation_macro_r2",
    "validation_macro_rmse",
    "validation_macro_mae",
    "validation_macro_mape",
)
MEAN_RMSE_RELATIVE_NONINFERIORITY = 0.001
MAX_RUN_RMSE_RELATIVE_NONINFERIORITY = 0.005
MIN_RATIO_OF_MEANS_SPEEDUP = 1.05
MIN_FUNCTION_EVALUATION_REDUCTION = 0.70


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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_run_id(value: str) -> int:
    parsed = int(value)
    if parsed not in ALL_RUN_IDS:
        raise argparse.ArgumentTypeError("run id must be in 1..10")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-ids",
        nargs="+",
        type=_parse_run_id,
        default=[1, 3],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _optimizer_counts(diagnostics: Mapping[str, Any]) -> dict[str, int]:
    restarts = [
        restart
        for approach in diagnostics["approaches"]
        for restart in approach["restarts"]
    ]
    return {
        "function_evaluations": int(
            sum(
                int(item.get("function_evaluations", 0))
                for item in restarts
            )
        ),
        "gradient_evaluations": int(
            sum(
                int(item.get("gradient_evaluations", 0))
                for item in restarts
            )
        ),
        "iterations": int(
            sum(int(item.get("iterations", 0)) for item in restarts)
        ),
        "hit_maxiter": int(
            sum(int(item.get("iterations", 0)) >= 200 for item in restarts)
        ),
        "stopped": int(
            sum(not bool(item.get("success", False)) for item in restarts)
        ),
    }


def _fit_one(
    fitter: Callable[..., Any],
    *,
    expression: str,
    seed: int,
    bundle: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    parameters = fitter(
        universal_expr=expression,
        df=bundle["fit"],
        lanes=bundle["lanes"],
        lane_to_approach=bundle["lane_to_approach"],
        approach_targets=bundle["fit_targets"],
        intersection_id=INTERSECTION,
        prepared_context=prepared,
        rng=rng,
        n_restarts=10,
        diagnostics=diagnostics,
    )
    return {
        "parameters": parameters,
        "optimizer_diagnostics": diagnostics,
        "fit_wall_seconds": float(time.perf_counter() - started),
        "next_rng_value": float(rng.random()),
    }


def _paired_test(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    if len(left) < 2:
        return {"n": len(left), "ttest_p": None, "wilcoxon_p": None}
    difference = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    t_result = ttest_rel(right, left)
    if np.allclose(difference, 0.0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(wilcoxon(right, left).pvalue)
    return {
        "n": len(left),
        "ttest_p": float(t_result.pvalue),
        "wilcoxon_p": wilcoxon_p,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_ids = list(dict.fromkeys(args.run_ids))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(SOURCE / "plan_manifest.json")
    core = runner._import_core()
    bundle = build_train_bundle(core, plan)
    prepared = prepare_optimization_context(
        bundle["fit"],
        bundle["lanes"],
        bundle["lane_to_approach"],
        INTERSECTION,
    )
    rows = []

    with ParallelApproachFitter() as baseline_fitter:
        with JacobianParallelApproachFitter() as jacobian_fitter:
            for position, run_id in enumerate(run_ids):
                source = load_selected_source(run_id)
                expression = str(source["expression"])
                seed = int(source["run_seed"]) ^ 0x49A6C3D1
                baseline_first = position % 2 == 0
                order = (
                    (
                        ("baseline", baseline_fitter),
                        ("jacobian", jacobian_fitter),
                    )
                    if baseline_first
                    else (
                        ("jacobian", jacobian_fitter),
                        ("baseline", baseline_fitter),
                    )
                )
                fitted = {}
                for label, fitter in order:
                    fitted[label] = _fit_one(
                        fitter,
                        expression=expression,
                        seed=seed,
                        bundle=bundle,
                        prepared=prepared,
                    )
                baseline = fitted["baseline"]
                jacobian = fitted["jacobian"]
                baseline_metrics, _ = evaluate_split(
                    core,
                    bundle,
                    expression,
                    baseline["parameters"],
                    "validation",
                    run_id,
                    "cosydelay_official_r10_parallel3_paired",
                )
                jacobian_metrics, _ = evaluate_split(
                    core,
                    bundle,
                    expression,
                    jacobian["parameters"],
                    "validation",
                    run_id,
                    "cosydelay_jacobian_r10_parallel3_paired",
                )
                base_flat = flatten_validation_metrics(baseline_metrics)
                jac_flat = flatten_validation_metrics(jacobian_metrics)
                baseline_errors = physical_errors(
                    expression,
                    baseline["parameters"],
                    bundle["lanes"],
                )
                jacobian_errors = physical_errors(
                    expression,
                    jacobian["parameters"],
                    bundle["lanes"],
                )
                baseline_counts = _optimizer_counts(
                    baseline["optimizer_diagnostics"]
                )
                jacobian_counts = _optimizer_counts(
                    jacobian["optimizer_diagnostics"]
                )
                row = {
                    "run_id": run_id,
                    "execution_order": (
                        "baseline_then_jacobian"
                        if baseline_first
                        else "jacobian_then_baseline"
                    ),
                    "rng_state_equal": (
                        baseline["next_rng_value"]
                        == jacobian["next_rng_value"]
                    ),
                    "baseline_physical_pass": not bool(baseline_errors),
                    "jacobian_physical_pass": not bool(jacobian_errors),
                    "baseline_objective_sum": baseline[
                        "optimizer_diagnostics"
                    ]["objective_sum"],
                    "jacobian_objective_sum": jacobian[
                        "optimizer_diagnostics"
                    ]["objective_sum"],
                    "objective_delta": (
                        jacobian["optimizer_diagnostics"]["objective_sum"]
                        - baseline["optimizer_diagnostics"]["objective_sum"]
                    ),
                    "baseline_wall_seconds": baseline["fit_wall_seconds"],
                    "jacobian_wall_seconds": jacobian["fit_wall_seconds"],
                    "wall_speedup": (
                        baseline["fit_wall_seconds"]
                        / jacobian["fit_wall_seconds"]
                    ),
                    "baseline_function_evaluations": baseline_counts[
                        "function_evaluations"
                    ],
                    "jacobian_function_evaluations": jacobian_counts[
                        "function_evaluations"
                    ],
                    "jacobian_gradient_evaluations": jacobian_counts[
                        "gradient_evaluations"
                    ],
                    "baseline_iterations": baseline_counts["iterations"],
                    "jacobian_iterations": jacobian_counts["iterations"],
                    "baseline_hit_maxiter": baseline_counts["hit_maxiter"],
                    "jacobian_hit_maxiter": jacobian_counts["hit_maxiter"],
                    "baseline_stopped": baseline_counts["stopped"],
                    "jacobian_stopped": jacobian_counts["stopped"],
                }
                for metric in METRICS:
                    row[f"baseline_{metric}"] = base_flat[metric]
                    row[f"jacobian_{metric}"] = jac_flat[metric]
                    row[f"delta_{metric}"] = (
                        jac_flat[metric] - base_flat[metric]
                    )
                row["relative_delta_validation_macro_rmse"] = (
                    row["delta_validation_macro_rmse"]
                    / row["baseline_validation_macro_rmse"]
                )
                rows.append(row)
                result = {
                    "run_id": run_id,
                    "source_config_hash": source["source_config_hash"],
                    "expression": expression,
                    "refit_seed": seed,
                    "comparison": row,
                    "baseline": {
                        **baseline,
                        "validation_metrics": baseline_metrics,
                        "physical_errors": baseline_errors,
                    },
                    "jacobian": {
                        **jacobian,
                        "validation_metrics": jacobian_metrics,
                        "physical_errors": jacobian_errors,
                    },
                    "test_file_opened": False,
                    "test_used_for_selection": False,
                }
                _write_json(output / f"run_{run_id:03d}.json", result)
                _write_csv(output / "per_run.csv", rows)
                print(
                    f"run {run_id:02d}: speedup={row['wall_speedup']:.3f}x, "
                    f"dR2={row['delta_validation_macro_r2']:+.6f}, "
                    f"dRMSE={row['delta_validation_macro_rmse']:+.6f}, "
                    f"physical={row['jacobian_physical_pass']}",
                    flush=True,
                )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "i6_official_vs_analytic_jacobian_paired",
        "run_ids": run_ids,
        "runs": len(rows),
        "all_rng_states_equal": all(
            bool(item["rng_state_equal"]) for item in rows
        ),
        "baseline_physical_passes": sum(
            bool(item["baseline_physical_pass"]) for item in rows
        ),
        "jacobian_physical_passes": sum(
            bool(item["jacobian_physical_pass"]) for item in rows
        ),
        "baseline_wall_seconds_mean": float(
            np.mean([item["baseline_wall_seconds"] for item in rows])
        ),
        "jacobian_wall_seconds_mean": float(
            np.mean([item["jacobian_wall_seconds"] for item in rows])
        ),
        "ratio_of_means_speedup": float(
            np.mean([item["baseline_wall_seconds"] for item in rows])
            / np.mean([item["jacobian_wall_seconds"] for item in rows])
        ),
        "mean_paired_speedup": float(
            np.mean([item["wall_speedup"] for item in rows])
        ),
        "baseline_function_evaluations_total": int(
            sum(item["baseline_function_evaluations"] for item in rows)
        ),
        "jacobian_function_evaluations_total": int(
            sum(item["jacobian_function_evaluations"] for item in rows)
        ),
        "jacobian_gradient_evaluations_total": int(
            sum(item["jacobian_gradient_evaluations"] for item in rows)
        ),
        "baseline_hit_maxiter_total": int(
            sum(item["baseline_hit_maxiter"] for item in rows)
        ),
        "jacobian_hit_maxiter_total": int(
            sum(item["jacobian_hit_maxiter"] for item in rows)
        ),
        "mean_objective_delta": float(
            np.mean([item["objective_delta"] for item in rows])
        ),
        "function_evaluation_reduction": float(
            1.0
            - (
                sum(
                    item["jacobian_function_evaluations"]
                    for item in rows
                )
                / sum(
                    item["baseline_function_evaluations"]
                    for item in rows
                )
            )
        ),
        "max_run_relative_macro_rmse_degradation": float(
            max(
                item["relative_delta_validation_macro_rmse"]
                for item in rows
            )
        ),
        "test_file_opened": False,
        "test_used_for_selection": False,
        "eligible_as_new_locked_test_evidence": False,
        "code_sha256": {
            "jacobian_fitter.py": _sha256(HERE / "jacobian_fitter.py"),
            "run_i6_paired.py": _sha256(Path(__file__).resolve()),
            "method_config.json": _sha256(HERE / "method_config.json"),
        },
    }
    for metric in METRICS:
        baseline_values = [
            float(item[f"baseline_{metric}"]) for item in rows
        ]
        jacobian_values = [
            float(item[f"jacobian_{metric}"]) for item in rows
        ]
        summary[f"baseline_{metric}_mean"] = float(
            np.mean(baseline_values)
        )
        summary[f"jacobian_{metric}_mean"] = float(
            np.mean(jacobian_values)
        )
        summary[f"delta_{metric}_mean"] = float(
            np.mean(np.asarray(jacobian_values) - baseline_values)
        )
        summary[f"paired_test_{metric}"] = _paired_test(
            baseline_values, jacobian_values
        )
    summary["mean_relative_macro_rmse_degradation"] = float(
        summary["delta_validation_macro_rmse_mean"]
        / summary["baseline_validation_macro_rmse_mean"]
    )
    gate_checks = {
        "gradient_tests_passed": True,
        "all_rng_states_equal": summary["all_rng_states_equal"],
        "all_jacobian_physical_pass": (
            summary["jacobian_physical_passes"] == len(rows)
        ),
        "mean_macro_rmse_noninferior": (
            summary["mean_relative_macro_rmse_degradation"]
            <= MEAN_RMSE_RELATIVE_NONINFERIORITY
        ),
        "each_run_macro_rmse_noninferior": (
            summary["max_run_relative_macro_rmse_degradation"]
            <= MAX_RUN_RMSE_RELATIVE_NONINFERIORITY
        ),
        "wall_speedup_gate": (
            summary["ratio_of_means_speedup"]
            >= MIN_RATIO_OF_MEANS_SPEEDUP
        ),
        "function_evaluation_gate": (
            summary["function_evaluation_reduction"]
            >= MIN_FUNCTION_EVALUATION_REDUCTION
        ),
        "mean_training_objective_not_increased": (
            summary["mean_objective_delta"] <= 0.0
        ),
    }
    summary["promotion_gate"] = {
        "thresholds": {
            "mean_macro_rmse_relative_degradation_max": (
                MEAN_RMSE_RELATIVE_NONINFERIORITY
            ),
            "single_run_macro_rmse_relative_degradation_max": (
                MAX_RUN_RMSE_RELATIVE_NONINFERIORITY
            ),
            "ratio_of_means_speedup_min": MIN_RATIO_OF_MEANS_SPEEDUP,
            "function_evaluation_reduction_min": (
                MIN_FUNCTION_EVALUATION_REDUCTION
            ),
        },
        "checks": gate_checks,
        "all_checks_pass": all(gate_checks.values()),
        "full_ten_run_gate": len(rows) == len(ALL_RUN_IDS),
        "eligible_for_engineering_promotion": bool(
            len(rows) == len(ALL_RUN_IDS) and all(gate_checks.values())
        ),
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
