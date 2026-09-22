"""Paired coefficient-fit ablation on a holdout drawn only from Training.

No LLM/API call is made and no Test path is constructed or opened.  The same
archived train-selected expression, rows, and ten-restart budget are used by
both arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

import numpy as np


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]
if str(GMINI) not in sys.path:
    sys.path.insert(0, str(GMINI))

from constants import INTERSECTION_CONFIGS  # noqa: E402
from data_processing import load_dataset_flexible, preprocess_data_flexible  # noqa: E402
from methods.cosydelay._internal.optimizer_parallel4.parallel_fitter import (  # noqa: E402
    ParallelApproachFitter,
)
from methods.cosydelay._internal.optimizer_parallel4_v3.fitter import (  # noqa: E402
    MixedRestartJacobianFitter,
    V3_RESTARTS,
)
from methods.cosydelay._internal.optimizer_parallel4_v3.physics_audit import (  # noqa: E402
    audit_fitted_physics,
)
from methods.cosydelay._internal.optimizer_parallel4_v3.training_selection import (  # noqa: E402
    balanced_regression_folds,
    score_predictions,
)
from optimization_lane import (  # noqa: E402
    calculate_approach_delays_from_universal,
    prepare_optimization_context,
)


LOCKED_SEARCH_ROOT = (
    GMINI
    / "methods"
    / "cosydelay_lbfgsb_r10_parallel4"
    / "experiments"
    / "locked_test_p4g2_all9_r3_20260803"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersections", nargs="+", type=int, default=[1, 4, 6])
    parser.add_argument("--source-run", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=GMINI.parent.parent / "Final_cosy_delay" / "jsonl_files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE
        / "experiments"
        / "training_only_refit_ablation_i1_i4_i6_v7_r9feasible",
    )
    return parser.parse_args()


def _lanes(config: Dict[str, Any]):
    lanes = []
    mapping = {}
    for approach in config["approaches"]:
        for movement in config["movements"][approach]:
            lane = f"{approach}_{movement}"
            lanes.append(lane)
            mapping[lane] = approach
    return lanes, mapping


def _targets(frame, approaches):
    return {
        approach: frame[f"Delay_{approach}"].reset_index(drop=True)
        for approach in approaches
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _evaluate(
    expression,
    parameters,
    validation,
    validation_targets,
    lanes,
    lane_to_approach,
    intersection_id,
):
    predictions = calculate_approach_delays_from_universal(
        df=validation,
        universal_expr=expression,
        lane_parameters=parameters,
        lanes=lanes,
        lane_to_approach=lane_to_approach,
        intersection_id=intersection_id,
        strict=True,
    )
    return score_predictions(validation_targets, predictions)


def main() -> int:
    args = _arguments()
    if not 0 <= args.fold < args.folds:
        raise ValueError("--fold must be inside [0, --folds)")
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for intersection_id in args.intersections:
        config = INTERSECTION_CONFIGS[intersection_id]
        approaches = list(config["approaches"])
        train_path = args.data_dir / f"Intersection_{intersection_id}_Train.jsonl"
        source_path = (
            LOCKED_SEARCH_ROOT
            / f"intersection_{intersection_id:02d}"
            / f"run_{args.source_run:02d}"
            / "search_result.json"
        )
        if not train_path.exists():
            raise FileNotFoundError(train_path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        expression = json.loads(source_path.read_text(encoding="utf-8"))["expression"]
        full_train = preprocess_data_flexible(
            load_dataset_flexible(str(train_path), intersection_id),
            intersection_id,
        ).reset_index(drop=True)
        full_targets = _targets(full_train, approaches)
        assignments = balanced_regression_folds(
            full_targets, n_folds=args.folds, seed=args.seed
        )
        fit_positions = np.flatnonzero(assignments != args.fold)
        validation_positions = np.flatnonzero(assignments == args.fold)
        fit = full_train.iloc[fit_positions].reset_index(drop=True)
        validation = full_train.iloc[validation_positions].reset_index(drop=True)
        fit_targets = _targets(fit, approaches)
        validation_targets = _targets(validation, approaches)
        lanes, lane_to_approach = _lanes(config)
        prepared = prepare_optimization_context(
            fit, lanes, lane_to_approach, intersection_id
        )
        seed = args.seed + intersection_id * 10_000 + args.fold

        baseline_diagnostics: Dict[str, Any] = {}
        baseline_started = time.perf_counter()
        with ParallelApproachFitter(parallel_workers=4) as baseline_fitter:
            baseline_parameters = baseline_fitter(
                universal_expr=expression,
                df=fit,
                lanes=lanes,
                lane_to_approach=lane_to_approach,
                approach_targets=fit_targets,
                intersection_id=intersection_id,
                prepared_context=prepared,
                rng=np.random.default_rng(seed),
                n_restarts=V3_RESTARTS,
                diagnostics=baseline_diagnostics,
            )
        baseline_wall = float(time.perf_counter() - baseline_started)
        baseline_metrics = _evaluate(
            expression,
            baseline_parameters,
            validation,
            validation_targets,
            lanes,
            lane_to_approach,
            intersection_id,
        )
        baseline_physics = audit_fitted_physics(
            expression, baseline_parameters, lanes
        )

        v3_diagnostics: Dict[str, Any] = {}
        v3_started = time.perf_counter()
        with MixedRestartJacobianFitter(parallel_workers=4) as v3_fitter:
            v3_parameters = v3_fitter.fit_final_with_polish(
                universal_expr=expression,
                df=fit,
                lanes=lanes,
                lane_to_approach=lane_to_approach,
                approach_targets=fit_targets,
                intersection_id=intersection_id,
                prepared_context=prepared,
                rng=np.random.default_rng(seed),
                n_restarts=V3_RESTARTS,
                diagnostics=v3_diagnostics,
            )
        v3_wall = float(time.perf_counter() - v3_started)
        v3_metrics = _evaluate(
            expression,
            v3_parameters,
            validation,
            validation_targets,
            lanes,
            lane_to_approach,
            intersection_id,
        )
        v3_physics = v3_diagnostics.get("selected_enhanced_physics")
        if not isinstance(v3_physics, dict):
            v3_physics = audit_fitted_physics(
                expression, v3_parameters, lanes
            )
        row = {
            "intersection_id": intersection_id,
            "source_expression_file": str(source_path.resolve()),
            "expression": expression,
            "fit_rows": len(fit),
            "training_inner_holdout_rows": len(validation),
            "fold": args.fold,
            "folds": args.folds,
            "seed": seed,
            "test_file_opened": False,
            "outer_validation_accessed": False,
            "baseline": {
                "method": "retained_random_uniform_finite_difference_r10",
                "metrics": baseline_metrics,
                "enhanced_physics": baseline_physics,
                "wall_seconds": baseline_wall,
                "optimizer": baseline_diagnostics,
            },
            "v3": {
                "method": "officiallocal9_raw_rolewide1_log_analytic_r10_r9feasible_polishrollback",
                "metrics": v3_metrics,
                "enhanced_physics": v3_physics,
                "wall_seconds": v3_wall,
                "optimizer": v3_diagnostics,
            },
            "delta_v3_minus_baseline": {
                "macro_r2": v3_metrics["macro_r2"] - baseline_metrics["macro_r2"],
                "macro_rmse": v3_metrics["macro_rmse"] - baseline_metrics["macro_rmse"],
                "macro_mae": v3_metrics["macro_mae"] - baseline_metrics["macro_mae"],
                "wall_seconds": v3_wall - baseline_wall,
            },
        }
        rows.append(row)
        _write_json(
            args.output / f"intersection_{intersection_id:02d}.json", row
        )
        _write_json(args.output / "partial_results.json", rows)
        print(
            f"I{intersection_id}: dR2={row['delta_v3_minus_baseline']['macro_r2']:+.6f}, "
            f"dRMSE={row['delta_v3_minus_baseline']['macro_rmse']:+.6f}, "
            f"dMAE={row['delta_v3_minus_baseline']['macro_mae']:+.6f}, "
            f"wall={baseline_wall:.1f}->{v3_wall:.1f}s",
            flush=True,
        )

    summary = {
        "experiment": "training_only_fixed_expression_refit_ablation",
        "intersections": args.intersections,
        "source_run": args.source_run,
        "fold": args.fold,
        "folds": args.folds,
        "test_file_opened": False,
        "outer_validation_accessed": False,
        "rows": rows,
        "mean_delta_v3_minus_baseline": {
            metric: float(
                np.mean([item["delta_v3_minus_baseline"][metric] for item in rows])
            )
            for metric in ("macro_r2", "macro_rmse", "macro_mae", "wall_seconds")
        },
        "all_v3_enhanced_physics_pass": all(
            item["v3"]["enhanced_physics"]["joint_pass"] for item in rows
        ),
    }
    _write_json(args.output / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
