"""Prediction-only Test evaluation of the completed, frozen run04 winners."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]
if str(GMINI) not in sys.path:
    sys.path.insert(0, str(GMINI))

from constants import INTERSECTION_CONFIGS  # noqa: E402
from data_processing import load_dataset_flexible, preprocess_data_flexible  # noqa: E402
from methods.cosydelay_v9_clean_from_scratch import (  # noqa: E402
    launch_formal_i1_i6 as launcher,
)
from methods.cosydelay_v9_clean_from_scratch.metrics import (  # noqa: E402
    score_predictions,
)
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import (  # noqa: E402
    DEFAULT_DATA_DIR,
    lanes_for,
)
from optimization_lane import calculate_approach_delays_from_universal  # noqa: E402


DEFAULT_SEARCH_ROOT = HERE / (
    "experiments/formal_v9_paper_fitness_promptv3_providerdefaults_"
    "p10g10_i1_i6_run04_20260810"
)
DEFAULT_OUTPUT = DEFAULT_SEARCH_ROOT / "test_20260811_run02"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", type=Path, default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(
                json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False)
                + "\n"
            )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_search(search_root: Path) -> tuple[dict, list[dict]]:
    protocol = read_json(search_root / "FROZEN_PROTOCOL.json")
    batch = read_json(search_root / "BATCH_COMPLETE.json")
    if batch.get("status") != "complete" or batch.get("formal_training_runs") != 6:
        raise RuntimeError("run04 search batch is not formally complete")

    current_mismatches = []
    for relative, expected in protocol["source_sha256"].items():
        path = GMINI / relative
        observed = sha256_file(path) if path.is_file() else None
        if observed != expected:
            current_mismatches.append(relative)
    if current_mismatches:
        raise RuntimeError(
            "frozen method source changed before Test evaluation: "
            + ", ".join(current_mismatches)
        )

    results = []
    for intersection in range(1, 7):
        path = (
            search_root
            / f"intersection_{intersection:02d}"
            / "run_01"
            / "result.json"
        )
        result = read_json(path)
        launcher.validate_formal_result(
            result, protocol, intersection=intersection
        )
        if result.get("test_file_opened") is not False:
            raise RuntimeError(f"I{intersection} search accessed Test")
        if result.get("post_evolution_refit") is not False:
            raise RuntimeError(f"I{intersection} has post-search refit")
        results.append(result)
    return protocol, results


def evaluate_one(
    *,
    intersection: int,
    result: dict,
    search_root: Path,
    data_dir: Path,
    output: Path,
) -> dict[str, Any]:
    test_path = (data_dir / f"Intersection_{intersection}_Test.jsonl").resolve()
    test_sha256 = sha256_file(test_path)
    test = preprocess_data_flexible(
        load_dataset_flexible(str(test_path), intersection), intersection
    ).reset_index(drop=True)
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
        approach: test[f"Delay_{approach}"].reset_index(drop=True)
        for approach in approaches
    }
    metrics = score_predictions(targets, predictions)

    prediction_rows = []
    for row_index in range(len(test)):
        row = {"row_index": row_index}
        for approach in approaches:
            row[f"actual_{approach}"] = float(targets[approach].iloc[row_index])
            row[f"predicted_{approach}"] = float(predictions[approach][row_index])
        prediction_rows.append(row)
    prediction_path = output / f"i{intersection:02d}_pred.jsonl"
    write_jsonl(prediction_path, prediction_rows)

    evaluation = {
        "intersection_id": intersection,
        "method_id": result["method_id"],
        "search_result_sha256": sha256_file(
            search_root
            / f"intersection_{intersection:02d}"
            / "run_01"
            / "result.json"
        ),
        "test_file_name": test_path.name,
        "test_file_sha256": test_sha256,
        "test_rows": len(test),
        "expression": expression,
        "lane_parameters_sha256": hashlib.sha256(
            json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "metrics": metrics,
        "formal_comparison_test_metrics": {
            "raw_approach_macro_r2": metrics["macro_raw_r2"],
            "pooled_rmse": metrics["pooled_rmse"],
            "pooled_mae": metrics["pooled_mae"],
        },
        "test_used_for_selection": False,
        "test_refit": False,
        "test_reselection": False,
        "prediction_only": True,
        "selected_training_fitness": result["training_fitness"],
        "selected_physics_joint_pass": result[
            "selected_enhanced_physics_from_evolution"
        ]["joint_pass"],
        "predictions_file": prediction_path.name,
        "predictions_sha256": sha256_file(prediction_path),
        "evaluated_utc": utc_now(),
    }
    write_json(output / f"i{intersection:02d}_metrics.json", evaluation)
    return evaluation


def main() -> int:
    args = arguments()
    search_root = args.search_root.resolve()
    data_dir = args.data_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing Test output: {output}")

    protocol, results = verify_frozen_search(search_root)
    output.mkdir(parents=True, exist_ok=False)
    evaluator_path = Path(__file__).resolve()
    write_json(
        output / "protocol.json",
        {
            "schema_version": 1,
            "status": "frozen_before_this_evaluator_opened_any_test_file",
            "frozen_utc": utc_now(),
            "search_root": str(search_root),
            "search_batch_completed_utc": read_json(
                search_root / "BATCH_COMPLETE.json"
            )["completed_utc"],
            "search_protocol_sha256": sha256_file(
                search_root / "FROZEN_PROTOCOL.json"
            ),
            "search_source_sha256": protocol["source_sha256"],
            "evaluator_file": str(evaluator_path),
            "evaluator_sha256": sha256_file(evaluator_path),
            "intersections": list(range(1, 7)),
            "prediction_only": True,
            "test_refit": False,
            "test_reselection": False,
            "test_used_for_selection": False,
            "metric_definition": (
                "raw approach-macro R2 plus pooled RMSE/MAE; same formal "
                "comparison definition as the data-driven models"
            ),
            "existing_test_previously_seen_during_project_development": True,
            "fresh_unseen_test_still_required_for_unbiased_claim": True,
        },
    )

    evaluations = [
        evaluate_one(
            intersection=intersection,
            result=results[intersection - 1],
            search_root=search_root,
            data_dir=data_dir,
            output=output,
        )
        for intersection in range(1, 7)
    ]
    aggregate = {
        "schema_version": 1,
        "scope": "locked existing Test prediction-only evaluation",
        "intersections": list(range(1, 7)),
        "mean_raw_approach_macro_r2": float(
            np.mean(
                [
                    item["formal_comparison_test_metrics"][
                        "raw_approach_macro_r2"
                    ]
                    for item in evaluations
                ]
            )
        ),
        "mean_pooled_rmse": float(
            np.mean(
                [
                    item["formal_comparison_test_metrics"]["pooled_rmse"]
                    for item in evaluations
                ]
            )
        ),
        "mean_pooled_mae": float(
            np.mean(
                [
                    item["formal_comparison_test_metrics"]["pooled_mae"]
                    for item in evaluations
                ]
            )
        ),
        "all_selected_physics_pass": all(
            item["selected_physics_joint_pass"] for item in evaluations
        ),
        "test_used_for_selection": False,
        "test_refit": False,
        "test_reselection": False,
        "existing_test_previously_seen_during_project_development": True,
        "fresh_unseen_test_still_required_for_unbiased_claim": True,
        "evaluations": evaluations,
        "completed_utc": utc_now(),
    }
    write_json(output / "aggregate.json", aggregate)
    write_json(
        output / "complete.json",
        {
            "status": "complete",
            "completed_utc": utc_now(),
            "intersections": 6,
            "prediction_only": True,
            "test_refit": False,
            "test_reselection": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
