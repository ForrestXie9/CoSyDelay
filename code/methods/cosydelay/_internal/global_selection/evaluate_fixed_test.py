"""Prediction-only Test evaluation for a completed V21 I1--I6 search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from methods.cosydelay._internal.training_protocol import evaluate_frozen_test as base


METHOD_LABEL = "cosydelay_v21_numeric_domain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=base.DEFAULT_DATA_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    search_root = args.search_root.resolve()
    output = args.output.resolve()
    data_dir = args.data_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing Test output: {output}")

    results = []
    for intersection in base.INTERSECTIONS:
        path = search_root / f"intersection_{intersection:02d}" / "run_01" / "result.json"
        result = read_json(path)
        if result.get("intersection_id") != intersection:
            raise RuntimeError(f"I{intersection} result identity mismatch")
        if result.get("accessed_splits") != ["train"]:
            raise RuntimeError(f"I{intersection} search was not Training-only")
        if result.get("test_file_opened") is not False:
            raise RuntimeError(f"I{intersection} reports Test access during search")
        results.append(result)

    output.mkdir(parents=True, exist_ok=False)
    previous_label, previous_policy = base.METHOD_LABEL, base.V16_POLICY
    base.METHOD_LABEL = METHOD_LABEL
    base.V16_POLICY = SimpleNamespace(method_id="cosydelay_v21_global_numeric_domain")
    try:
        evaluations, prediction_rows = [], []
        for intersection, result in zip(base.INTERSECTIONS, results):
            evaluation, rows = base._evaluate_one(
                intersection=intersection,
                result=result,
                search_root=search_root,
                data_dir=data_dir,
            )
            evaluations.append(evaluation)
            prediction_rows.extend(rows)
            base._write_json(output / f"i{intersection:02d}_metrics.json", evaluation)
    finally:
        base.METHOD_LABEL, base.V16_POLICY = previous_label, previous_policy

    predictions_path = output / "predictions.csv"
    base._write_predictions(predictions_path, prediction_rows)
    aggregate = {
        "scope": "fixed V21 prediction-only diagnostic evaluation on existing Test",
        "method": METHOD_LABEL,
        "mean_pooled_r2": float(np.mean([x["pooled_test_metrics"]["r2"] for x in evaluations])),
        "mean_pooled_rmse": float(np.mean([x["pooled_test_metrics"]["rmse"] for x in evaluations])),
        "mean_pooled_mae": float(np.mean([x["pooled_test_metrics"]["mae"] for x in evaluations])),
        "strict_joint_passing_winners": int(sum(x["selected_strict_joint_pass"] for x in evaluations)),
        "prediction_rows": len(prediction_rows),
        "prediction_only": True,
        "test_used_for_generation": False,
        "test_used_for_coefficient_fitting": False,
        "test_used_for_selection": False,
        "test_refit": False,
        "test_reselection": False,
        "existing_test_previously_seen_during_project_development": True,
        "fresh_unseen_test_still_required_for_strictly_blind_claim": True,
        "evaluations": evaluations,
        "completed_utc": base.utc_now(),
    }
    base._write_json(output / "aggregate.json", aggregate)
    for item in evaluations:
        metrics = item["pooled_test_metrics"]
        print(f"I{item['intersection_id']}: R2={metrics['r2']:.6f}, RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}")
    print(f"Mean: R2={aggregate['mean_pooled_r2']:.6f}, RMSE={aggregate['mean_pooled_rmse']:.6f}, MAE={aggregate['mean_pooled_mae']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
