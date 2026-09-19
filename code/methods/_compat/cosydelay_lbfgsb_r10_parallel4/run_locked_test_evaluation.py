"""Two-stage frozen search followed by one locked-Test evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


METHOD_DIR = Path(__file__).resolve().parent
GMINI_DIR = METHOD_DIR.parents[1]

if str(GMINI_DIR) not in sys.path:
    sys.path.insert(0, str(GMINI_DIR))

from constants import INTERSECTION_CONFIGS  # noqa: E402
from data_processing import load_dataset_flexible, preprocess_data_flexible  # noqa: E402
from optimization_lane import calculate_approach_delays_from_universal  # noqa: E402
from reviewer_revision_experiments.evolution_matrix.shared import to_jsonable  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersections", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=20260712)
    parser.add_argument("--parallel-searches", type=int, default=2)
    parser.add_argument("--max-wall-seconds", type=float, default=900.0)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=GMINI_DIR.parent.parent / "Final_cosy_delay" / "jsonl_files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=METHOD_DIR / "experiments" / "locked_test_p4g2_all9_r3_20260803",
    )
    return parser.parse_args()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            to_jsonable(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_sd(values: Iterable[float]) -> Dict[str, float]:
    data = [float(value) for value in values]
    return {
        "mean": statistics.mean(data),
        "sd": statistics.stdev(data) if len(data) > 1 else 0.0,
    }


def _frozen_manifest(root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    relative_files = [
        "population_evolution_lane.py",
        "optimization_lane.py",
        "expression_validation_lane.py",
        "expression_adaptation_lane.py",
        "methods/cosydelay_lbfgsb_r10_parallel4/search_policy.py",
        "methods/cosydelay_lbfgsb_r10_parallel4/prefit_gate.py",
        "methods/cosydelay_lbfgsb_r10_parallel4/prefit_integration.py",
        "methods/cosydelay_lbfgsb_r10_parallel4/parallel_fitter.py",
        "methods/cosydelay_lbfgsb_r10_parallel4/method_config.json",
        "methods/cosydelay_lbfgsb_r10_parallel4/run_frozen_search_task.py",
        "methods/cosydelay_lbfgsb_r10_parallel4/run_locked_test_evaluation.py",
    ]
    files = []
    for relative in relative_files:
        path = GMINI_DIR / relative
        files.append({"path": relative, "sha256": _sha256(path)})
    combined = hashlib.sha256(
        "\n".join(f"{item['path']}:{item['sha256']}" for item in files).encode()
    ).hexdigest()
    config = json.loads(
        (METHOD_DIR / "method_config.json").read_text(encoding="utf-8")
    )
    if config.get("retention_status") != "official":
        raise RuntimeError("method configuration is not marked official")
    if not str(config["retained_search_policy"]["status"]).startswith("official"):
        raise RuntimeError("P4/G2 search policy is not marked official")
    manifest = {
        "freeze_utc": _utc(),
        "freeze_id": combined,
        "method_id": config["method_id"],
        "method_config": config,
        "files": files,
        "intersections": args.intersections,
        "runs_per_intersection": args.runs,
        "base_seed": args.base_seed,
        "test_files_opened_at_freeze": False,
        "test_evaluation_authorized": True,
        "protocol": (
            "finish all Train-only searches before opening any Test file; "
            "then perform prediction-only evaluation with no refit"
        ),
    }
    _write_json(root / "FROZEN_METHOD.json", manifest)
    return manifest


def _search_task(task: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(task["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "methods.cosydelay_lbfgsb_r10_parallel4.run_frozen_search_task",
        "--intersection",
        str(task["intersection"]),
        "--run-id",
        str(task["run_id"]),
        "--base-seed",
        str(args.base_seed),
        "--max-wall-seconds",
        str(args.max_wall_seconds),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output",
        str(run_dir),
    ]
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    started = time.perf_counter()
    with (
        (run_dir / "search.stdout.log").open("w", encoding="utf-8") as stdout,
        (run_dir / "search.stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=GMINI_DIR,
            stdout=stdout,
            stderr=stderr,
            env=environment,
            check=False,
        )
    outcome = dict(task)
    outcome.update(
        {
            "return_code": completed.returncode,
            "subprocess_wall_seconds": time.perf_counter() - started,
            "search_result_path": str(run_dir / "search_result.json"),
        }
    )
    return outcome


def _lanes(config):
    lanes = []
    mapping = {}
    for approach in config["approaches"]:
        for movement in config["movements"][approach]:
            lane = f"{approach}_{movement}"
            lanes.append(lane)
            mapping[lane] = approach
    return lanes, mapping


def _mape(truth: np.ndarray, prediction: np.ndarray) -> float:
    mask = np.isfinite(truth) & np.isfinite(prediction) & (truth != 0)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((truth[mask] - prediction[mask]) / truth[mask])) * 100)


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(truth) & np.isfinite(prediction)
    if np.count_nonzero(mask) < 2:
        raise ValueError("fewer than two finite prediction rows")
    y = truth[mask]
    p = prediction[mask]
    return {
        "r2": float(r2_score(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "mape": _mape(y, p),
        "rows": int(len(y)),
    }


def _evaluate_test(
    root: Path,
    outcomes: List[Dict[str, Any]],
    args: argparse.Namespace,
    freeze_id: str,
) -> List[Dict[str, Any]]:
    evaluations = []
    test_manifest = {
        "test_access_started_utc": _utc(),
        "all_searches_completed_before_test_access": True,
        "freeze_id": freeze_id,
        "files": [],
    }
    by_intersection = defaultdict(list)
    for outcome in outcomes:
        by_intersection[int(outcome["intersection"])].append(outcome)

    for intersection in sorted(by_intersection):
        test_path = args.data_dir / f"Intersection_{intersection}_Test.jsonl"
        test_manifest["files"].append(
            {
                "intersection": intersection,
                "path_name": test_path.name,
                "sha256": _sha256(test_path),
                "opened_utc": _utc(),
            }
        )
        test = preprocess_data_flexible(
            load_dataset_flexible(str(test_path), intersection), intersection
        )
        config = INTERSECTION_CONFIGS[intersection]
        approaches = list(config["approaches"])
        lanes, lane_to_approach = _lanes(config)
        for outcome in sorted(by_intersection[intersection], key=lambda item: item["run_id"]):
            search = json.loads(
                Path(outcome["search_result_path"]).read_text(encoding="utf-8")
            )
            predictions = calculate_approach_delays_from_universal(
                df=test,
                universal_expr=search["expression"],
                lane_parameters=search["lane_parameters"],
                lanes=lanes,
                lane_to_approach=lane_to_approach,
                intersection_id=intersection,
                strict=True,
            )
            per_approach = {}
            pooled_truth = []
            pooled_prediction = []
            for approach in approaches:
                truth = np.asarray(test[f"Delay_{approach}"].values, dtype=float)
                prediction = np.asarray(predictions[approach], dtype=float)
                per_approach[approach] = _metrics(truth, prediction)
                finite = np.isfinite(truth) & np.isfinite(prediction)
                pooled_truth.append(truth[finite])
                pooled_prediction.append(prediction[finite])
            macro = {
                metric: float(np.mean([values[metric] for values in per_approach.values()]))
                for metric in ("r2", "rmse", "mae", "mape")
            }
            macro["r2_clipped"] = max(0.0, macro["r2"])
            pooled = _metrics(
                np.concatenate(pooled_truth), np.concatenate(pooled_prediction)
            )
            evaluation = {
                "freeze_id": freeze_id,
                "method_id": search["method_id"],
                "intersection_id": intersection,
                "run_id": search["run_id"],
                "expression": search["expression"],
                "physical_joint_pass": search["physical_joint_pass"],
                "physical_rule_scores": search["physical_rule_scores"],
                "train_r2": search["train_r2"],
                "test_rows": len(test),
                "test_metrics_by_approach": per_approach,
                "test_macro": macro,
                "test_pooled": pooled,
                "test_used_for_selection": False,
                "refit_on_test": False,
                "test_evaluated_utc": _utc(),
            }
            _write_json(Path(outcome["run_dir"]) / "test_evaluation.json", evaluation)
            evaluations.append(evaluation)
    test_manifest["test_access_completed_utc"] = _utc()
    _write_json(root / "TEST_ACCESS_MANIFEST.json", test_manifest)
    return evaluations


def _aggregate(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups = []
    grouped = defaultdict(list)
    for item in evaluations:
        grouped[int(item["intersection_id"])].append(item)
    for intersection, items in sorted(grouped.items()):
        groups.append(
            {
                "intersection": intersection,
                "runs": len(items),
                "physical_passes": sum(bool(item["physical_joint_pass"]) for item in items),
                "train_r2": _mean_sd(item["train_r2"] for item in items),
                "test_macro_r2": _mean_sd(item["test_macro"]["r2"] for item in items),
                "test_macro_r2_clipped": _mean_sd(
                    item["test_macro"]["r2_clipped"] for item in items
                ),
                "test_macro_rmse": _mean_sd(
                    item["test_macro"]["rmse"] for item in items
                ),
                "test_macro_mae": _mean_sd(item["test_macro"]["mae"] for item in items),
                "test_macro_mape": _mean_sd(
                    item["test_macro"]["mape"] for item in items
                ),
                "test_pooled_r2": _mean_sd(item["test_pooled"]["r2"] for item in items),
                "test_pooled_rmse": _mean_sd(
                    item["test_pooled"]["rmse"] for item in items
                ),
            }
        )
    overall = {
        "intersection_run_count": len(evaluations),
        "physical_pass_rate": statistics.mean(
            bool(item["physical_joint_pass"]) for item in evaluations
        ),
        "test_macro_r2": _mean_sd(item["test_macro"]["r2"] for item in evaluations),
        "test_macro_rmse": _mean_sd(
            item["test_macro"]["rmse"] for item in evaluations
        ),
        "test_macro_mae": _mean_sd(item["test_macro"]["mae"] for item in evaluations),
    }
    return {"groups": groups, "overall": overall, "evaluations": evaluations}


def _markdown(aggregate: Dict[str, Any]) -> str:
    lines = [
        "# Frozen P4/G2 locked-Test evaluation",
        "",
        "All Train-only searches completed before any Test file was opened.",
        "No Test refitting or selection occurred.",
        "",
        "| I | Runs | Physical | Test macro R2 | Test RMSE | Test MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in aggregate["groups"]:
        lines.append(
            "| I{i} | {runs} | {physical}/{runs} | {r2:.4f} +/- {r2sd:.4f} | "
            "{rmse:.4f} +/- {rmsesd:.4f} | {mae:.4f} +/- {maesd:.4f} |".format(
                i=group["intersection"],
                runs=group["runs"],
                physical=group["physical_passes"],
                r2=group["test_macro_r2"]["mean"],
                r2sd=group["test_macro_r2"]["sd"],
                rmse=group["test_macro_rmse"]["mean"],
                rmsesd=group["test_macro_rmse"]["sd"],
                mae=group["test_macro_mae"]["mean"],
                maesd=group["test_macro_mae"]["sd"],
            )
        )
    overall = aggregate["overall"]
    lines.extend(
        [
            "",
            "Overall across intersection-runs: physical pass rate "
            f"{overall['physical_pass_rate']:.1%}; macro R2 "
            f"{overall['test_macro_r2']['mean']:.4f}; macro RMSE "
            f"{overall['test_macro_rmse']['mean']:.4f}; macro MAE "
            f"{overall['test_macro_mae']['mean']:.4f}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _arguments()
    if args.runs < 1 or args.parallel_searches < 1:
        raise ValueError("runs and parallel-searches must be positive")
    if sorted(set(args.intersections)) != sorted(args.intersections):
        raise ValueError("intersection IDs must be unique")
    unknown = [item for item in args.intersections if item not in INTERSECTION_CONFIGS]
    if unknown:
        raise ValueError(f"unknown intersections: {unknown}")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    freeze = _frozen_manifest(root, args)

    tasks = []
    for intersection in args.intersections:
        for run_id in range(1, args.runs + 1):
            tasks.append(
                {
                    "intersection": intersection,
                    "run_id": run_id,
                    "run_dir": str(
                        root / f"intersection_{intersection:02d}" / f"run_{run_id:02d}"
                    ),
                }
            )

    outcomes = []
    search_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel_searches) as executor:
        futures = {executor.submit(_search_task, task, args): task for task in tasks}
        for future in as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            outcomes.sort(key=lambda item: (item["intersection"], item["run_id"]))
            _write_json(root / "SEARCH_MANIFEST.json", outcomes)
            print(
                f"search {len(outcomes)}/{len(tasks)}: I{outcome['intersection']} "
                f"run={outcome['run_id']} rc={outcome['return_code']} "
                f"wall={outcome['subprocess_wall_seconds']:.1f}s",
                flush=True,
            )

    failed = [item for item in outcomes if item["return_code"] != 0]
    if failed:
        raise RuntimeError(f"frozen search failure; Test remains unopened: {failed}")
    for outcome in outcomes:
        search = json.loads(
            Path(outcome["search_result_path"]).read_text(encoding="utf-8")
        )
        if search["test_file_opened"] or search["validation_evaluation_events"] != 0:
            raise RuntimeError("search protocol invariant failed; Test remains unopened")
    search_complete = {
        "freeze_id": freeze["freeze_id"],
        "search_completed_utc": _utc(),
        "search_wall_seconds": time.perf_counter() - search_started,
        "searches": len(outcomes),
        "all_return_codes_zero": True,
        "test_files_opened": False,
    }
    _write_json(root / "SEARCH_COMPLETE.json", search_complete)
    print("all frozen searches complete; beginning one-time Test evaluation", flush=True)

    evaluations = _evaluate_test(root, outcomes, args, freeze["freeze_id"])
    aggregate = _aggregate(evaluations)
    aggregate["freeze_id"] = freeze["freeze_id"]
    aggregate["search_wall_seconds"] = search_complete["search_wall_seconds"]
    _write_json(root / "aggregate.json", aggregate)
    (root / "SUMMARY.md").write_text(_markdown(aggregate), encoding="utf-8")
    print(f"locked-Test evaluation complete for {len(evaluations)} runs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
