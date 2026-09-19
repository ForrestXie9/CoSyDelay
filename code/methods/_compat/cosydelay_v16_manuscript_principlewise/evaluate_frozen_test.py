"""Prediction-only Test evaluation after all six V16 searches are frozen."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
from methods.cosydelay_v9_clean_from_scratch.metrics import score_predictions
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import (
    DEFAULT_DATA_DIR,
    lanes_for,
    sha256_file,
)
from optimization_lane import calculate_approach_delays_from_universal

from . import launch_p10g10_i1_i6 as launcher
from .policy import V16_POLICY
from .source_manifest import GMINI


HERE = Path(__file__).resolve().parent
INTERSECTIONS = tuple(range(1, 7))
METHOD_LABEL = "cosydelay_v16_p10g10"
FROZEN_FIXED17_SUMMARY_SHA256 = (
    "2263c47a80a263a11f71123578d51a24b45752aa03580894a00e8a7cc86b948a"
)
DEFAULT_BASELINE_DIR = (
    GMINI
    / "reviewer_revision_experiments"
    / "07_raw16_fixed_train_test_data_driven_comparison"
    / "outputs"
    / "formal_raw16_fixed_17methods_i1_i6_10runs_v1"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument(
        "--skip-baseline-ranking",
        action="store_true",
        help="Write frozen Test metrics without the fixed-17 comparison.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(value), ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_sample_ids(path: Path) -> list[str]:
    """Match the fixed raw-16 baseline's stable sample identifier."""
    identifiers = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            dialogue = json.loads(line)["dialogue"]
            digest = hashlib.sha256(dialogue.encode("utf-8")).hexdigest()
            identifiers.append(f"{digest[:20]}-{line_number:05d}")
    return identifiers


def pooled_metrics(
    targets: Mapping[str, Sequence[float]],
    predictions: Mapping[str, Sequence[float]],
) -> dict[str, float | int]:
    truth_parts = []
    prediction_parts = []
    for approach, target in targets.items():
        truth = np.asarray(target, dtype=float)
        predicted = np.asarray(predictions[approach], dtype=float)
        if truth.shape != predicted.shape:
            raise RuntimeError(
                f"{approach} prediction shape {predicted.shape} != {truth.shape}"
            )
        valid = np.isfinite(truth) & np.isfinite(predicted)
        truth_parts.append(truth[valid])
        prediction_parts.append(predicted[valid])
    truth = np.concatenate(truth_parts)
    predicted = np.concatenate(prediction_parts)
    if len(truth) < 2 or not np.isfinite(predicted).all():
        raise RuntimeError("insufficient finite pooled Test predictions")
    error = truth - predicted
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    if denominator <= 0.0:
        raise RuntimeError("pooled Test target has non-positive variance")
    return {
        "r2": float(1.0 - np.sum(error**2) / denominator),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "rows": int(len(truth)),
    }


def verify_frozen_search(
    search_root: Path, data_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify Training artifacts without opening or hashing any Test file."""
    protocol_path = search_root / "FROZEN_PROTOCOL.json"
    batch_path = search_root / "BATCH_COMPLETE.json"
    status_path = search_root / "RUN_STATUS.json"
    protocol = _read_json(protocol_path)
    batch = _read_json(batch_path)
    status = _read_json(status_path)
    if protocol.get("status") != "frozen_before_any_v16_formal_api_call":
        raise RuntimeError("search protocol was not frozen before V16 API calls")
    if protocol.get("test_sha256") is not None:
        raise RuntimeError("search protocol unexpectedly contains a Test hash")
    if protocol.get("existing_test_allowed_during_search") is not False:
        raise RuntimeError("search protocol did not prohibit Test access")
    if batch.get("status") != "complete" or batch.get("formal_training_runs") != 6:
        raise RuntimeError("V16 Training batch is not complete")
    if status.get("status") != "completed" or len(status.get("runs", [])) != 6:
        raise RuntimeError("V16 run status is not complete for all six intersections")
    if batch.get("frozen_protocol_sha256") != sha256_file(protocol_path):
        raise RuntimeError("frozen protocol hash differs from completion record")

    current_source = launcher._source_hashes()
    if current_source != protocol.get("source_sha256"):
        changed = sorted(
            key
            for key in set(current_source) | set(protocol.get("source_sha256", {}))
            if current_source.get(key) != protocol.get("source_sha256", {}).get(key)
        )
        raise RuntimeError(
            "frozen V16 source changed before Test evaluation: " + ", ".join(changed)
        )
    results = []
    for intersection in INTERSECTIONS:
        train_path = data_dir / f"Intersection_{intersection}_Train.jsonl"
        if not train_path.is_file():
            raise FileNotFoundError(train_path)
        expected_train_hash = protocol["training_sha256"][str(intersection)][
            "sha256"
        ]
        if sha256_file(train_path) != expected_train_hash:
            raise RuntimeError(f"I{intersection} Training data changed after freeze")
        result_path = (
            search_root
            / f"intersection_{intersection:02d}"
            / "run_01"
            / "result.json"
        )
        if sha256_file(result_path) != batch["result_sha256"][str(intersection)]:
            raise RuntimeError(f"I{intersection} result changed after completion")
        result = _read_json(result_path)
        launcher.validate_formal_result(
            result, protocol, intersection=intersection
        )
        results.append(result)
    return protocol, results


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "method",
        "run_id",
        "intersection_id",
        "approach",
        "sample_id",
        "split",
        "y_true",
        "y_pred",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validated_fixed17_summary(baseline_path: Path) -> pd.DataFrame:
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    baseline = pd.read_csv(baseline_path)
    required = {"method", "intersection_id", "r2", "rmse", "mae"}
    if not required.issubset(baseline.columns):
        raise RuntimeError(f"baseline summary lacks columns: {required - set(baseline)}")
    baseline = baseline.loc[:, sorted(required)].copy()
    if baseline["method"].nunique() != 17 or len(baseline) != 17 * 6:
        raise RuntimeError("fixed baseline summary is not 17 methods x 6 intersections")
    expected_intersections = set(INTERSECTIONS)
    if set(baseline["intersection_id"].astype(int)) != expected_intersections:
        raise RuntimeError("fixed baseline summary has unexpected intersections")
    metric_values = baseline.loc[:, ["r2", "rmse", "mae"]].to_numpy(dtype=float)
    if not np.isfinite(metric_values).all():
        raise RuntimeError("fixed baseline summary contains non-finite metrics")
    return baseline


def _rank_against_fixed17(
    *,
    baseline_dir: Path,
    evaluations: list[dict[str, Any]],
    output: Path,
    expected_baseline_sha256: str,
) -> dict[str, Any]:
    baseline_path = baseline_dir / "raw16_per_intersection_method_metrics.csv"
    if sha256_file(baseline_path) != expected_baseline_sha256:
        raise RuntimeError("fixed-17 baseline summary changed after Test preflight")
    baseline = _validated_fixed17_summary(baseline_path)
    cosy = pd.DataFrame(
        [
            {
                "method": METHOD_LABEL,
                "intersection_id": item["intersection_id"],
                "r2": item["pooled_test_metrics"]["r2"],
                "rmse": item["pooled_test_metrics"]["rmse"],
                "mae": item["pooled_test_metrics"]["mae"],
            }
            for item in evaluations
        ]
    )
    blocks = pd.concat([baseline, cosy], ignore_index=True)
    ranked_parts = []
    for intersection, group in blocks.groupby("intersection_id", sort=True):
        if len(group) != 18:
            raise RuntimeError(f"I{intersection} comparison has {len(group)} methods")
        group = group.copy()
        group["r2_rank"] = group["r2"].rank(method="average", ascending=False)
        group["rmse_rank"] = group["rmse"].rank(method="average", ascending=True)
        group["mae_rank"] = group["mae"].rank(method="average", ascending=True)
        group["composite_rank"] = group[
            ["r2_rank", "rmse_rank", "mae_rank"]
        ].mean(axis=1)
        ranked_parts.append(group)
    ranked = pd.concat(ranked_parts, ignore_index=True)
    average = (
        ranked.groupby("method", as_index=False)
        .agg(
            r2_rank=("r2_rank", "mean"),
            rmse_rank=("rmse_rank", "mean"),
            mae_rank=("mae_rank", "mean"),
            average_rank=("composite_rank", "mean"),
        )
        .sort_values(["average_rank", "method"], ignore_index=True)
    )
    average.insert(0, "overall_position", np.arange(1, len(average) + 1))
    ranked.to_csv(output / "rank_blocks_18methods.csv", index=False)
    average.to_csv(output / "average_ranks_18methods.csv", index=False)
    selected = average.loc[average["method"].eq(METHOD_LABEL)].iloc[0]
    intersection_rows = ranked.loc[ranked["method"].eq(METHOD_LABEL)]
    best_intersections = {
        metric: [
            int(value)
            for value in intersection_rows.loc[
                intersection_rows[f"{metric}_rank"].eq(1.0), "intersection_id"
            ]
        ]
        for metric in ("r2", "rmse", "mae")
    }
    best_intersections["composite"] = [
        int(value)
        for value in intersection_rows.loc[
            intersection_rows["composite_rank"].eq(1.0), "intersection_id"
        ]
    ]
    payload = {
        "comparison_scope": (
            "one frozen V16 run versus the fixed-parameter raw-16 baselines; "
            "stochastic baselines are ten-run means"
        ),
        "ranking_unit": "pooled valid approach samples within each intersection",
        "metrics": ["R2 descending", "RMSE ascending", "MAE ascending"],
        "fixed_baseline_summary_file": str(baseline_path.resolve()),
        "fixed_baseline_summary_sha256": sha256_file(baseline_path),
        "method_count": 18,
        "v16_overall_position": int(selected["overall_position"]),
        "v16_r2_average_rank": float(selected["r2_rank"]),
        "v16_rmse_average_rank": float(selected["rmse_rank"]),
        "v16_mae_average_rank": float(selected["mae_rank"]),
        "v16_three_metric_average_rank": float(selected["average_rank"]),
        "v16_best_intersections_by_metric": best_intersections,
        "v16_best_intersection_counts_by_metric": {
            key: len(value) for key, value in best_intersections.items()
        },
        "v16_intersection_composite_ranks": {
            str(int(row.intersection_id)): float(row.composite_rank)
            for row in intersection_rows.itertuples()
        },
        "selection_use": False,
        "existing_test_previously_seen_during_project_development": True,
        "fresh_unseen_test_still_required_for_strictly_blind_claim": True,
    }
    _write_json(output / "fixed17_comparison.json", payload)
    return payload


def _evaluate_one(
    *,
    intersection: int,
    result: dict[str, Any],
    search_root: Path,
    data_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    test_path = (data_dir / f"Intersection_{intersection}_Test.jsonl").resolve()
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    sample_ids = _raw_sample_ids(test_path)
    test_hash = sha256_file(test_path)
    test = preprocess_data_flexible(
        load_dataset_flexible(str(test_path), intersection), intersection
    ).reset_index(drop=True)
    if len(sample_ids) != len(test):
        raise RuntimeError(f"I{intersection} raw/Test row count mismatch")
    config = INTERSECTION_CONFIGS[intersection]
    approaches = list(config["approaches"])
    lanes, lane_to_approach = lanes_for(config)
    expression = str(result["expression"])
    parameters = result["lane_parameters"]
    predictions = calculate_approach_delays_from_universal(
        df=test,
        universal_expr=expression,
        lane_parameters=parameters,
        lanes=lanes,
        lane_to_approach=lane_to_approach,
        intersection_id=intersection,
        strict=True,
    )
    targets = {
        approach: test[f"Delay_{approach}"].to_numpy(dtype=float)
        for approach in approaches
    }
    approach_metrics = score_predictions(targets, predictions)
    pooled = pooled_metrics(targets, predictions)
    prediction_rows = []
    for approach in approaches:
        truth = np.asarray(targets[approach], dtype=float)
        predicted = np.asarray(predictions[approach], dtype=float)
        if not np.isfinite(predicted).all():
            raise RuntimeError(f"I{intersection}/{approach} has non-finite prediction")
        for sample_id, actual, estimate in zip(sample_ids, truth, predicted):
            prediction_rows.append(
                {
                    "method": METHOD_LABEL,
                    "run_id": 1,
                    "intersection_id": intersection,
                    "approach": approach,
                    "sample_id": sample_id,
                    "split": "test",
                    "y_true": float(actual),
                    "y_pred": float(estimate),
                }
            )
    result_path = (
        search_root / f"intersection_{intersection:02d}" / "run_01" / "result.json"
    )
    evaluation = {
        "intersection_id": intersection,
        "method_id": V16_POLICY.method_id,
        "method_label": METHOD_LABEL,
        "search_result_sha256": sha256_file(result_path),
        "test_file_name": test_path.name,
        "test_file_sha256": test_hash,
        "test_rows": len(test),
        "approaches": approaches,
        "expression": expression,
        "lane_parameters_sha256": hashlib.sha256(
            json.dumps(
                parameters, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "approach_macro_test_metrics": approach_metrics,
        "pooled_test_metrics": pooled,
        "formal_fixed17_comparison_metrics": {
            "pooled_r2": pooled["r2"],
            "pooled_rmse": pooled["rmse"],
            "pooled_mae": pooled["mae"],
        },
        "selected_training_fitness": result["training_fitness"],
        "selected_training_metrics": result["training_metrics"],
        "selected_physical_score": result["selected_enhanced_physics"]["score"],
        "selected_strict_joint_pass": bool(
            result["selected_enhanced_physics"].get("joint_pass")
        ),
        "prediction_only": True,
        "test_used_for_generation": False,
        "test_used_for_coefficient_fitting": False,
        "test_used_for_selection": False,
        "test_refit": False,
        "test_reselection": False,
        "evaluated_utc": utc_now(),
    }
    return evaluation, prediction_rows


def main() -> int:
    args = arguments()
    search_root = args.search_root.resolve()
    data_dir = args.data_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing Test output: {output}")

    search_protocol, results = verify_frozen_search(search_root, data_dir)
    baseline_path = (
        args.baseline_dir.resolve() / "raw16_per_intersection_method_metrics.csv"
    )
    baseline_sha256 = None
    if not args.skip_baseline_ranking:
        baseline_sha256 = sha256_file(baseline_path)
        if baseline_sha256 != FROZEN_FIXED17_SUMMARY_SHA256:
            raise RuntimeError(
                "fixed-17 baseline summary differs from the hash committed "
                "in the frozen V16 evaluator"
            )
        _validated_fixed17_summary(baseline_path)
    output.mkdir(parents=True, exist_ok=False)
    evaluator_path = Path(__file__).resolve()
    evaluation_protocol = {
        "schema_version": 2,
        "status": "frozen_before_this_process_opened_or_hashed_any_test_file",
        "frozen_utc": utc_now(),
        "search_root": str(search_root),
        "search_batch_completed_utc": _read_json(
            search_root / "BATCH_COMPLETE.json"
        )["completed_utc"],
        "search_protocol_sha256": sha256_file(
            search_root / "FROZEN_PROTOCOL.json"
        ),
        "search_source_sha256": search_protocol["source_sha256"],
        "evaluator_file": str(evaluator_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "fixed17_baseline_summary_file": (
            str(baseline_path) if baseline_sha256 is not None else None
        ),
        "fixed17_baseline_summary_sha256": baseline_sha256,
        "intersections": list(INTERSECTIONS),
        "prediction_only": True,
        "test_used_for_generation": False,
        "test_used_for_coefficient_fitting": False,
        "test_used_for_selection": False,
        "test_refit": False,
        "test_reselection": False,
        "metric_definition": (
            "pooled R2/RMSE/MAE over valid approach-sample pairs for fixed-17 "
            "ranking; raw approach-macro metrics also retained"
        ),
        "existing_test_previously_seen_during_project_development": True,
        "fresh_unseen_test_still_required_for_strictly_blind_claim": True,
    }
    _write_json(output / "TEST_EVALUATION_PROTOCOL.json", evaluation_protocol)

    evaluations = []
    prediction_rows = []
    for intersection, result in zip(INTERSECTIONS, results):
        evaluation, rows = _evaluate_one(
            intersection=intersection,
            result=result,
            search_root=search_root,
            data_dir=data_dir,
        )
        evaluations.append(evaluation)
        prediction_rows.extend(rows)
        _write_json(output / f"i{intersection:02d}_metrics.json", evaluation)
    predictions_path = output / "predictions.csv"
    _write_predictions(predictions_path, prediction_rows)

    aggregate = {
        "schema_version": 2,
        "scope": "frozen existing-Test prediction-only evaluation",
        "intersections": list(INTERSECTIONS),
        "mean_pooled_r2": float(
            np.mean([item["pooled_test_metrics"]["r2"] for item in evaluations])
        ),
        "mean_pooled_rmse": float(
            np.mean([item["pooled_test_metrics"]["rmse"] for item in evaluations])
        ),
        "mean_pooled_mae": float(
            np.mean([item["pooled_test_metrics"]["mae"] for item in evaluations])
        ),
        "mean_raw_approach_macro_r2": float(
            np.mean(
                [
                    item["approach_macro_test_metrics"]["macro_raw_r2"]
                    for item in evaluations
                ]
            )
        ),
        "mean_winner_physical_score": float(
            np.mean([item["selected_physical_score"] for item in evaluations])
        ),
        "strict_joint_passing_winners": int(
            sum(item["selected_strict_joint_pass"] for item in evaluations)
        ),
        "prediction_rows": len(prediction_rows),
        "predictions_file": predictions_path.name,
        "predictions_sha256": sha256_file(predictions_path),
        "prediction_only": True,
        "test_used_for_selection": False,
        "test_refit": False,
        "test_reselection": False,
        "existing_test_previously_seen_during_project_development": True,
        "fresh_unseen_test_still_required_for_strictly_blind_claim": True,
        "evaluations": evaluations,
        "completed_utc": utc_now(),
    }
    _write_json(output / "aggregate.json", aggregate)
    comparison = None
    if not args.skip_baseline_ranking:
        comparison = _rank_against_fixed17(
            baseline_dir=args.baseline_dir.resolve(),
            evaluations=evaluations,
            output=output,
            expected_baseline_sha256=baseline_sha256,
        )
    _write_json(
        output / "COMPLETE.json",
        {
            "schema_version": 2,
            "status": "complete",
            "completed_utc": utc_now(),
            "intersections": 6,
            "prediction_only": True,
            "test_refit": False,
            "test_reselection": False,
            "aggregate_sha256": sha256_file(output / "aggregate.json"),
            "predictions_sha256": sha256_file(predictions_path),
            "fixed17_comparison_completed": comparison is not None,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
