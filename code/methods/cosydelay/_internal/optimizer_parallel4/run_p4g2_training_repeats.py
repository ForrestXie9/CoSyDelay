"""Repeat the frozen P4/G2 policy using training and physics only."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List


METHOD_DIR = Path(__file__).resolve().parent
GMINI_DIR = METHOD_DIR.parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--intersections", nargs="+", type=int, default=[1, 6])
    parser.add_argument("--seed-start", type=int, default=20260803)
    parser.add_argument("--split-seed", type=int, default=20260712)
    parser.add_argument("--parallel-runs", type=int, default=2)
    parser.add_argument("--max-wall-seconds", type=float, default=900.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=METHOD_DIR
        / "experiments"
        / "training_only_p4g2_repeats_i1_i6_5x_20260803",
    )
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _mean_sd(values: Iterable[float]) -> Dict[str, float]:
    data = [float(value) for value in values]
    return {
        "mean": statistics.mean(data),
        "sd": statistics.stdev(data) if len(data) > 1 else 0.0,
    }


def _run_task(task: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    task_dir = Path(task["task_dir"])
    task_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "methods.cosydelay._internal.optimizer_parallel4.run_training_only_smoke",
        "--intersections",
        str(task["intersection"]),
        "--population",
        "4",
        "--generations",
        "2",
        "--seed",
        str(task["seed"]),
        "--split-seed",
        str(args.split_seed),
        "--prefit-mode",
        "conservative",
        "--search-policy",
        "feasible_adaptive",
        "--validation-mode",
        "none",
        "--max-wall-seconds",
        str(args.max_wall_seconds),
        "--output",
        str(task_dir),
    ]
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    started = time.perf_counter()
    with (
        (task_dir / "stdout.log").open("w", encoding="utf-8") as stdout,
        (task_dir / "stderr.log").open("w", encoding="utf-8") as stderr,
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
            "result_path": str(
                task_dir
                / f"intersection_{int(task['intersection']):02d}"
                / "result.json"
            ),
        }
    )
    return outcome


def _aggregate(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for outcome in outcomes:
        result_path = Path(outcome["result_path"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        history = json.loads(
            result_path.with_name("history.json").read_text(encoding="utf-8")
        )
        candidates = [item for item in history if item.get("event") == "evaluated"]
        best_any_train_r2 = max(float(item["train_r2"]) for item in candidates)
        targeted_feedback_candidates = sum(
            bool(
                item.get("evaluation_details", {})
                .get("targeted_physical_feedback", {})
                .get("failed_rules", [])
            )
            for item in candidates
        )
        row = {**outcome, **result}
        row.update(
            {
                "best_any_train_r2": best_any_train_r2,
                "final_train_r2_gap": max(
                    0.0, best_any_train_r2 - float(result["train_r2"])
                ),
                "targeted_feedback_candidates": targeted_feedback_candidates,
            }
        )
        rows.append(row)

    groups = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["intersection_id"])].append(row)
    for intersection, items in sorted(grouped.items()):
        groups.append(
            {
                "intersection": intersection,
                "runs": len(items),
                "final_train_r2": _mean_sd(item["train_r2"] for item in items),
                "best_any_train_r2": _mean_sd(
                    item["best_any_train_r2"] for item in items
                ),
                "final_train_r2_gap": _mean_sd(
                    item["final_train_r2_gap"] for item in items
                ),
                "final_physical_passes": sum(
                    bool(item["physical_joint_pass"]) for item in items
                ),
                "early_stops": sum(bool(item["early_stop_triggered"]) for item in items),
                "candidate_evaluations": _mean_sd(
                    item["candidate_evaluations"] for item in items
                ),
                "completed_generations": _mean_sd(
                    item["completed_generations"] for item in items
                ),
                "wall_seconds": _mean_sd(item["wall_seconds"] for item in items),
                "prefit_gate_rejections": sum(
                    int(item["prefit_gate_rejections"]) for item in items
                ),
                "targeted_feedback_candidates": sum(
                    int(item["targeted_feedback_candidates"]) for item in items
                ),
            }
        )

    invariant_failures = []
    for row in rows:
        checks = {
            "test_locked": not bool(row["test_file_opened"]),
            "validation_mode_none": row.get("validation_mode") == "none",
            "validation_not_evaluated": not bool(row.get("validation_evaluated")),
            "validation_events_zero": int(row["validation_evaluation_events"]) == 0,
            "validation_not_used_in_evolution": not bool(
                row["validation_accessed_during_evolution"]
            ),
            "ten_restarts": int(row["optimizer_restarts"]) == 10,
            "p4g2": int(row["population"]) == 4 and int(row["generations"]) == 2,
        }
        if not all(checks.values()):
            invariant_failures.append(
                {
                    "intersection": row["intersection_id"],
                    "seed": row["run_seed"],
                    "checks": checks,
                }
            )
    return {
        "schema_version": 1,
        "rows": rows,
        "groups": groups,
        "protocol_invariant_failures": invariant_failures,
    }


def _markdown(aggregate: Dict[str, Any]) -> str:
    lines = [
        "# Training-only P4/G2 repeated pilot",
        "",
        "No validation or Test metric is evaluated in these runs.",
        "",
        "| I | Runs | Final physical | Early stop | Final train R2 | Best-any train R2 | Train gap | Candidates | Wall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in aggregate["groups"]:
        lines.append(
            "| I{i} | {runs} | {physical}/{runs} | {early}/{runs} | "
            "{final:.4f} +/- {final_sd:.4f} | {best:.4f} +/- {best_sd:.4f} | "
            "{gap:.4f} +/- {gap_sd:.4f} | {candidates:.1f} | {wall:.1f}s |".format(
                i=group["intersection"],
                runs=group["runs"],
                physical=group["final_physical_passes"],
                early=group["early_stops"],
                final=group["final_train_r2"]["mean"],
                final_sd=group["final_train_r2"]["sd"],
                best=group["best_any_train_r2"]["mean"],
                best_sd=group["best_any_train_r2"]["sd"],
                gap=group["final_train_r2_gap"]["mean"],
                gap_sd=group["final_train_r2_gap"]["sd"],
                candidates=group["candidate_evaluations"]["mean"],
                wall=group["wall_seconds"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "Protocol invariant failures: "
            f"{len(aggregate['protocol_invariant_failures'])}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _arguments()
    if args.repetitions < 1 or args.parallel_runs < 1:
        raise ValueError("repetitions and parallel-runs must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for repetition in range(1, args.repetitions + 1):
        seed = args.seed_start + repetition - 1
        for intersection in args.intersections:
            tasks.append(
                {
                    "repetition": repetition,
                    "seed": seed,
                    "intersection": intersection,
                    "task_dir": str(
                        root
                        / f"rep_{repetition:02d}_seed_{seed}"
                        / f"I{intersection:02d}"
                    ),
                }
            )

    outcomes = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel_runs) as executor:
        futures = {executor.submit(_run_task, task, args): task for task in tasks}
        for future in as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            outcomes.sort(key=lambda item: (item["repetition"], item["intersection"]))
            _write_json(root / "run_manifest.json", outcomes)
            print(
                f"completed {len(outcomes)}/{len(tasks)}: "
                f"rep={outcome['repetition']}, I{outcome['intersection']}, "
                f"seed={outcome['seed']}, rc={outcome['return_code']}, "
                f"wall={outcome['subprocess_wall_seconds']:.1f}s",
                flush=True,
            )

    failed = [item for item in outcomes if item["return_code"] != 0]
    if failed:
        raise RuntimeError(f"one or more P4/G2 runs failed: {failed}")
    aggregate = _aggregate(outcomes)
    aggregate["batch_wall_seconds"] = time.perf_counter() - started
    _write_json(root / "aggregate.json", aggregate)
    (root / "SUMMARY.md").write_text(_markdown(aggregate), encoding="utf-8")
    print(
        f"completed {len(outcomes)} runs in {aggregate['batch_wall_seconds']:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
