"""Evaluate an archived Training-selected expression on a reference OOF split.

This utility opens only Intersection_*_Train.jsonl.  It is used to put the
retained v2 expression on the exact fold assignments reported by a v3 run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]
if str(GMINI) not in sys.path:
    sys.path.insert(0, str(GMINI))

from constants import INTERSECTION_CONFIGS  # noqa: E402
from data_processing import load_dataset_flexible, preprocess_data_flexible  # noqa: E402
from methods.cosydelay.engine.optimizer_parallel4.parallel_fitter import (  # noqa: E402
    ParallelApproachFitter,
)
from methods.cosydelay.engine.optimizer_parallel4_v3.physics_audit import (  # noqa: E402
    audit_fitted_physics,
)
from methods.cosydelay.engine.optimizer_parallel4_v3.training_selection import (  # noqa: E402
    _stable_fit_seed,
    score_predictions,
)
from optimization_lane import (  # noqa: E402
    calculate_approach_delays_from_universal,
    prepare_optimization_context,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", type=int, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--reference-v3-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=GMINI.parent.parent / "Final_cosy_delay" / "jsonl_files",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _lanes(config):
    lanes = []
    mapping = {}
    for approach in config["approaches"]:
        for movement in config["movements"][approach]:
            lane = f"{approach}_{movement}"
            lanes.append(lane)
            mapping[lane] = approach
    return lanes, mapping


def main() -> int:
    args = _arguments()
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_v3_result.read_text(encoding="utf-8"))
    if bool(source.get("test_file_opened", False)):
        raise RuntimeError("source result reports Test access")
    if bool(reference.get("test_file_opened", False)):
        raise RuntimeError("reference v3 result reports Test access")
    if bool(reference.get("outer_validation_accessed", False)):
        raise RuntimeError("reference v3 result reports outer Validation access")
    if int(source["intersection_id"]) != args.intersection:
        raise ValueError("source intersection does not match --intersection")
    if int(reference["intersection_id"]) != args.intersection:
        raise ValueError("reference intersection does not match --intersection")

    config = INTERSECTION_CONFIGS[args.intersection]
    approaches = list(config["approaches"])
    lanes, lane_to_approach = _lanes(config)
    train_path = args.data_dir / f"Intersection_{args.intersection}_Train.jsonl"
    train = preprocess_data_flexible(
        load_dataset_flexible(str(train_path), args.intersection),
        args.intersection,
    ).reset_index(drop=True)
    targets = {
        approach: train[f"Delay_{approach}"].reset_index(drop=True)
        for approach in approaches
    }
    assignments = np.asarray(
        reference["selection_report"]["fold_assignments"], dtype=int
    )
    if len(assignments) != len(train):
        raise ValueError("reference fold assignments do not match Training rows")
    folds = sorted(np.unique(assignments).tolist())
    expression = str(source["expression"])
    seed = int(reference["seed"])
    oof_predictions = {
        approach: np.full(len(train), np.nan, dtype=float)
        for approach in approaches
    }
    fold_records = []
    started = time.perf_counter()
    with ParallelApproachFitter(parallel_workers=4) as fitter:
        for fold in folds:
            fit_positions = np.flatnonzero(assignments != fold)
            selection_positions = np.flatnonzero(assignments == fold)
            fit = train.iloc[fit_positions].reset_index(drop=True)
            selection = train.iloc[selection_positions].reset_index(drop=True)
            fit_targets = {
                name: values.iloc[fit_positions].reset_index(drop=True)
                for name, values in targets.items()
            }
            selection_targets = {
                name: values.iloc[selection_positions].reset_index(drop=True)
                for name, values in targets.items()
            }
            diagnostics = {}
            fold_started = time.perf_counter()
            parameters = fitter(
                universal_expr=expression,
                df=fit,
                lanes=lanes,
                lane_to_approach=lane_to_approach,
                approach_targets=fit_targets,
                intersection_id=args.intersection,
                prepared_context=prepare_optimization_context(
                    fit, lanes, lane_to_approach, args.intersection
                ),
                rng=np.random.default_rng(
                    _stable_fit_seed(seed, expression, fold, "four_fold_cv")
                ),
                n_restarts=10,
                diagnostics=diagnostics,
            )
            predictions = calculate_approach_delays_from_universal(
                df=selection,
                universal_expr=expression,
                lane_parameters=parameters,
                lanes=lanes,
                lane_to_approach=lane_to_approach,
                intersection_id=args.intersection,
                strict=True,
            )
            for approach in approaches:
                oof_predictions[approach][selection_positions] = np.asarray(
                    predictions[approach], dtype=float
                )
            fold_records.append(
                {
                    "fold": int(fold),
                    "fit_rows": int(len(fit_positions)),
                    "selection_rows": int(len(selection_positions)),
                    "metrics": score_predictions(selection_targets, predictions),
                    "enhanced_physics": audit_fitted_physics(
                        expression, parameters, lanes
                    ),
                    "optimizer": diagnostics,
                    "wall_seconds": float(time.perf_counter() - fold_started),
                }
            )

    result = {
        "experiment": "archived_training_selected_expression_reference_oof",
        "method": "retained_v2_finite_difference_lbfgsb_r10_parallel4",
        "intersection_id": args.intersection,
        "source_result": str(args.source_result.resolve()),
        "reference_v3_result": str(args.reference_v3_result.resolve()),
        "train_file_name": train_path.name,
        "train_rows": len(train),
        "outer_validation_accessed": False,
        "test_file_opened": False,
        "expression": expression,
        "fold_assignments_sha256_source": "reference_v3_result",
        "folds": fold_records,
        "oof_metrics": score_predictions(targets, oof_predictions),
        "all_folds_enhanced_physics": all(
            row["enhanced_physics"]["joint_pass"] for row in fold_records
        ),
        "wall_seconds": float(time.perf_counter() - started),
    }
    _write_json(args.output, result)
    metrics = result["oof_metrics"]
    print(
        f"I{args.intersection} retained-v2 OOF: R2={metrics['macro_r2']:.5f}, "
        f"RMSE={metrics['macro_rmse']:.5f}, MAE={metrics['macro_mae']:.5f}, "
        f"physical={result['all_folds_enhanced_physics']}, "
        f"wall={result['wall_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
