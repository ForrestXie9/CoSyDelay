"""Verify official parallel3 against archived I6 random-uniform r10 fits.

Only the frozen fit/validation training file and archived validation results are
loaded. The test file is neither opened nor evaluated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


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
    OFFICIAL_MAXFUN,
    OFFICIAL_MAXITER,
    OFFICIAL_RESTARTS,
    ParallelApproachFitter,
)
from optimization_lane import prepare_optimization_context  # noqa: E402


INTERSECTION = 6
RUN_IDS = tuple(range(1, 11))
PRIOR_TUNING = (
    REVIEW_ROOT
    / "matrix_runs"
    / "i06_validation_only_tuning_v1"
)
DEFAULT_OUTPUT = HERE / "verification" / "i6_archived_r10_equivalence"
METRIC_NAMES = (
    "validation_macro_r2",
    "validation_macro_rmse",
    "validation_macro_mae",
    "validation_macro_mape",
    "validation_micro_r2",
    "validation_micro_rmse",
    "validation_micro_mae",
)


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


def _max_parameter_delta(
    left: Mapping[str, Mapping[str, float]],
    right: Mapping[str, Mapping[str, float]],
) -> float:
    if set(left) != set(right):
        raise ValueError("Lane keys differ between archived and parallel fits")
    deltas = []
    for lane in sorted(left):
        if set(left[lane]) != set(right[lane]):
            raise ValueError(f"Coefficient keys differ for lane {lane}")
        deltas.extend(
            abs(float(left[lane][name]) - float(right[lane][name]))
            for name in sorted(left[lane])
        )
    return float(max(deltas, default=0.0))


def _restart_objective_delta(
    archived: Mapping[str, Any],
    current: Mapping[str, Any],
) -> tuple[float, bool]:
    old_by_approach = {
        str(item["approach"]): item
        for item in archived["optimizer_diagnostics"]["approaches"]
    }
    new_by_approach = {
        str(item["approach"]): item
        for item in current["approaches"]
    }
    if set(old_by_approach) != set(new_by_approach):
        raise ValueError("Approach keys differ between diagnostics")
    deltas = []
    chosen_equal = True
    for approach in sorted(old_by_approach):
        old = old_by_approach[approach]
        new = new_by_approach[approach]
        chosen_equal &= int(old["chosen_restart"]) == int(
            new["chosen_restart"]
        )
        old_restarts = old["restarts"]
        new_restarts = new["restarts"]
        if len(old_restarts) != len(new_restarts):
            raise ValueError(f"Restart count differs for approach {approach}")
        deltas.extend(
            abs(float(old_item["objective"]) - float(new_item["objective"]))
            for old_item, new_item in zip(old_restarts, new_restarts)
        )
    return float(max(deltas, default=0.0)), bool(chosen_equal)


def _reference_path(run_id: int) -> Path:
    return (
        PRIOR_TUNING
        / "restart_tuning"
        / "linear_r10"
        / f"run_{run_id:03d}"
        / "validation_result.json"
    )


def _positive_prefix(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > len(RUN_IDS):
        raise argparse.ArgumentTypeError(
            f"runs must be between 1 and {len(RUN_IDS)}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=_positive_prefix, default=len(RUN_IDS))
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not np.isfinite(args.tolerance) or args.tolerance < 0.0:
        raise ValueError("--tolerance must be finite and non-negative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_plan = read_json(SOURCE / "plan_manifest.json")
    core = runner._import_core()
    bundle = build_train_bundle(core, source_plan)
    prepared = prepare_optimization_context(
        bundle["fit"],
        bundle["lanes"],
        bundle["lane_to_approach"],
        INTERSECTION,
    )
    rows = []
    full_results = []
    selected_run_ids = RUN_IDS[: args.runs]

    with ParallelApproachFitter() as fitter:
        for run_id in selected_run_ids:
            archived_path = _reference_path(run_id)
            archived = read_json(archived_path)
            source = load_selected_source(run_id)
            expression = str(source["expression"])
            if expression != str(archived["expression"]):
                raise ValueError(f"Expression mismatch in run {run_id}")
            refit_seed = int(source["run_seed"]) ^ 0x49A6C3D1
            if refit_seed != int(archived["refit_seed"]):
                raise ValueError(f"Refit seed mismatch in run {run_id}")

            diagnostics: dict[str, Any] = {}
            started = time.perf_counter()
            parameters = fitter(
                universal_expr=expression,
                df=bundle["fit"],
                lanes=bundle["lanes"],
                lane_to_approach=bundle["lane_to_approach"],
                approach_targets=bundle["fit_targets"],
                intersection_id=INTERSECTION,
                prepared_context=prepared,
                rng=np.random.default_rng(refit_seed),
                n_restarts=OFFICIAL_RESTARTS,
                diagnostics=diagnostics,
            )
            fit_wall_seconds = float(time.perf_counter() - started)
            validation_metrics, _ = evaluate_split(
                core,
                bundle,
                expression,
                parameters,
                "validation",
                run_id,
                "cosydelay_lbfgsb_r10_parallel3_verification",
            )
            errors = physical_errors(
                expression, parameters, bundle["lanes"]
            )
            current_metrics = flatten_validation_metrics(validation_metrics)
            archived_metrics = flatten_validation_metrics(
                archived["validation_metrics"]
            )
            parameter_delta = _max_parameter_delta(
                archived["parameters"], parameters
            )
            objective_delta = abs(
                float(archived["optimizer_diagnostics"]["objective_sum"])
                - float(diagnostics["objective_sum"])
            )
            metric_delta = max(
                abs(current_metrics[name] - archived_metrics[name])
                for name in METRIC_NAMES
            )
            restart_delta, chosen_equal = _restart_objective_delta(
                archived, diagnostics
            )
            equivalent = bool(
                parameter_delta <= args.tolerance
                and objective_delta <= args.tolerance
                and metric_delta <= args.tolerance
                and restart_delta <= args.tolerance
                and chosen_equal
            )
            archived_wall = float(archived["fit_wall_seconds"])
            speedup = (
                archived_wall / fit_wall_seconds
                if fit_wall_seconds > 0.0
                else None
            )
            row = {
                "run_id": run_id,
                "equivalent_at_tolerance": equivalent,
                "max_abs_parameter_delta": parameter_delta,
                "objective_abs_delta": objective_delta,
                "max_validation_metric_abs_delta": metric_delta,
                "max_restart_objective_abs_delta": restart_delta,
                "chosen_restarts_equal": chosen_equal,
                "physical_joint_pass": not bool(errors),
                "archived_serial_wall_seconds": archived_wall,
                "parallel_wall_seconds": fit_wall_seconds,
                "historical_wall_speedup": speedup,
                "internal_parallel_efficiency_ratio": diagnostics[
                    "parallel_efficiency_ratio"
                ],
                **current_metrics,
            }
            rows.append(row)
            result = {
                "run_id": run_id,
                "source_config_hash": source["source_config_hash"],
                "archived_result": str(archived_path.resolve()),
                "archived_result_sha256": _sha256(archived_path),
                "refit_seed": refit_seed,
                "expression": expression,
                "parameters": parameters,
                "optimizer_diagnostics": diagnostics,
                "validation_metrics": validation_metrics,
                "physical_errors": errors,
                "comparison": row,
                "test_file_opened": False,
                "test_used_for_selection": False,
            }
            full_results.append(result)
            _write_json(output / f"run_{run_id:03d}.json", result)
            print(
                f"run {run_id:02d}: equivalent={equivalent}, "
                f"parameter_delta={parameter_delta:.3g}, "
                f"wall={fit_wall_seconds:.2f}s",
                flush=True,
            )

    parallel_walls = [float(item["parallel_wall_seconds"]) for item in rows]
    archived_walls = [
        float(item["archived_serial_wall_seconds"]) for item in rows
    ]
    paired_speedups = [
        float(item["historical_wall_speedup"])
        for item in rows
        if item["historical_wall_speedup"] is not None
    ]
    summary = {
        "schema_version": 1,
        "method_id": "cosydelay_lbfgsb_r10_parallel3_v1",
        "intersection_id": INTERSECTION,
        "run_ids": list(selected_run_ids),
        "solver": "L-BFGS-B",
        "optimizer_restarts": OFFICIAL_RESTARTS,
        "optimizer_maxiter": OFFICIAL_MAXITER,
        "optimizer_maxfun": OFFICIAL_MAXFUN,
        "parallel_workers": 3,
        "tolerance": float(args.tolerance),
        "runs_equivalent": sum(
            bool(item["equivalent_at_tolerance"]) for item in rows
        ),
        "all_runs_equivalent": all(
            bool(item["equivalent_at_tolerance"]) for item in rows
        ),
        "all_runs_physical_pass": all(
            bool(item["physical_joint_pass"]) for item in rows
        ),
        "max_abs_parameter_delta": max(
            float(item["max_abs_parameter_delta"]) for item in rows
        ),
        "max_objective_abs_delta": max(
            float(item["objective_abs_delta"]) for item in rows
        ),
        "max_validation_metric_abs_delta": max(
            float(item["max_validation_metric_abs_delta"])
            for item in rows
        ),
        "max_restart_objective_abs_delta": max(
            float(item["max_restart_objective_abs_delta"])
            for item in rows
        ),
        "archived_serial_wall_seconds_mean": float(
            np.mean(archived_walls)
        ),
        "parallel_wall_seconds_mean": float(np.mean(parallel_walls)),
        "historical_paired_speedup_mean": float(
            np.mean(paired_speedups)
        ),
        "historical_ratio_of_means_speedup": float(
            np.mean(archived_walls) / np.mean(parallel_walls)
        ),
        "wall_timing_note": (
            "The serial times are archived historical measurements, not "
            "simultaneous paired timings; use them as efficiency context only."
        ),
        "source_matrix": str(SOURCE.resolve()),
        "archived_reference_root": str(PRIOR_TUNING.resolve()),
        "code_sha256": {
            "parallel_fitter.py": _sha256(HERE / "parallel_fitter.py"),
            "verify_i6_equivalence.py": _sha256(Path(__file__).resolve()),
            "method_config.json": _sha256(HERE / "method_config.json"),
        },
        "test_file_opened": False,
        "test_used_for_selection": False,
        "eligible_as_new_locked_test_evidence": False,
    }
    _write_csv(output / "per_run.csv", rows)
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if summary["all_runs_equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
