"""Run exploratory P10/G10 searches, then reuse the consumed Test diagnostically."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


GMINI_DIR = Path(__file__).resolve().parents[2]
if str(GMINI_DIR) not in sys.path:
    sys.path.insert(0, str(GMINI_DIR))

from reviewer_revision_experiments.evolution_matrix.shared import to_jsonable  # noqa: E402

from .run_locked_test_evaluation import (  # noqa: E402
    _aggregate,
    _evaluate_test,
    _utc,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersections", nargs="+", type=int, default=list(range(1, 7)))
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260712)
    parser.add_argument("--parallel-searches", type=int, default=2)
    parser.add_argument("--max-wall-seconds", type=float, default=7200.0)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(value), ensure_ascii=False, allow_nan=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _config_id(args: argparse.Namespace) -> str:
    payload = json.dumps(
        {
            "method": "cosydelay_lbfgsb_r10_parallel4",
            "population": args.population,
            "generations": args.generations,
            "optimizer_restarts": 10,
            "selection": "training_only_physical_feasible_archive",
            "intersections": sorted(args.intersections),
            "runs": 1,
            "base_seed": args.base_seed,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_task(task, args):
    run_dir = Path(task["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "methods.cosydelay_lbfgsb_r10_parallel4.run_exploratory_budget_task",
        "--intersection",
        str(task["intersection"]),
        "--run-id",
        "1",
        "--population",
        str(args.population),
        "--generations",
        str(args.generations),
        "--base-seed",
        str(args.base_seed),
        "--max-wall-seconds",
        str(args.max_wall_seconds),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output",
        str(run_dir.resolve()),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=GMINI_DIR,
            capture_output=True,
            text=True,
            timeout=args.max_wall_seconds + 600.0,
            check=False,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        return_code = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "\nsubprocess timeout\n"
    (run_dir / "search.stdout.log").write_text(stdout, encoding="utf-8")
    (run_dir / "search.stderr.log").write_text(stderr, encoding="utf-8")
    return {
        "intersection": task["intersection"],
        "run_id": 1,
        "run_dir": str(run_dir.resolve()),
        "search_result_path": str((run_dir / "search_result.json").resolve()),
        "return_code": return_code,
        "subprocess_wall_seconds": time.perf_counter() - started,
    }


def _summary(aggregate, args, config_id):
    lines = [
        "# Exploratory P10/G10 diagnostic",
        "",
        "> The locked Test was consumed before this experiment. These Test metrics are",
        "> diagnostic reuse only and are not new confirmatory locked-Test evidence.",
        "",
        f"Config ID: `{config_id}`.",
        "",
        "Expression search and selection used Train only; Test was prediction-only with no refit.",
        "",
        "| I | Physical | Train R2 | Test macro R2 | Test pooled R2 | Test RMSE | Test MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in aggregate["groups"]:
        evaluation = next(
            item
            for item in aggregate["evaluations"]
            if item["intersection_id"] == group["intersection"]
        )
        lines.append(
            "| I{i} | {physical}/1 | {train:.4f} | {macro:.4f} | {pooled:.4f} | "
            "{rmse:.4f} | {mae:.4f} |".format(
                i=group["intersection"],
                physical=group["physical_passes"],
                train=evaluation["train_r2"],
                macro=evaluation["test_macro"]["r2"],
                pooled=evaluation["test_pooled"]["r2"],
                rmse=evaluation["test_pooled"]["rmse"],
                mae=evaluation["test_pooled"]["mae"],
            )
        )
    values = aggregate["evaluations"]
    lines.extend(
        [
            "",
            "I1-I6 pooled/micro intersection mean: R2 {r2:.4f}, RMSE {rmse:.4f}, "
            "MAE {mae:.4f}.".format(
                r2=statistics.mean(item["test_pooled"]["r2"] for item in values),
                rmse=statistics.mean(item["test_pooled"]["rmse"] for item in values),
                mae=statistics.mean(item["test_pooled"]["mae"] for item in values),
            ),
            "",
            f"Budget: P{args.population}/G{args.generations}, L-BFGS-B 10 restarts, "
            "at most four approach workers, no training-side early stop.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _arguments()
    if args.population < 1 or args.generations < 0 or args.parallel_searches < 1:
        raise ValueError("invalid budget or parallelism")
    if len(set(args.intersections)) != len(args.intersections):
        raise ValueError("intersection IDs must be unique")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_id = _config_id(args)
    notice = {
        "config_id": config_id,
        "status": "exploratory_reuse_of_previously_consumed_test",
        "created_utc": _utc(),
        "population": args.population,
        "generations": args.generations,
        "optimizer_restarts": 10,
        "approach_worker_cap": 4,
        "selection_data": "Train only",
        "test_role": "diagnostic prediction only; no selection or refit",
        "intersections": sorted(args.intersections),
        "runs_per_intersection": 1,
    }
    _write_json(root / "EXPLORATORY_PROTOCOL.json", notice)

    tasks = [
        {
            "intersection": intersection,
            "run_dir": str(root / f"intersection_{intersection:02d}" / "run_01"),
        }
        for intersection in args.intersections
    ]
    outcomes = []
    search_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel_searches) as executor:
        futures = {executor.submit(_run_task, task, args): task for task in tasks}
        for future in as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            outcomes.sort(key=lambda item: item["intersection"])
            _write_json(root / "SEARCH_MANIFEST.json", outcomes)
            print(
                f"search {len(outcomes)}/{len(tasks)}: I{outcome['intersection']} "
                f"rc={outcome['return_code']} wall={outcome['subprocess_wall_seconds']:.1f}s",
                flush=True,
            )

    failed = [item for item in outcomes if item["return_code"] != 0]
    if failed:
        raise RuntimeError(f"exploratory search failure; Test not re-opened: {failed}")
    searches = []
    for outcome in outcomes:
        search = json.loads(Path(outcome["search_result_path"]).read_text(encoding="utf-8"))
        if search["test_file_opened"] or search["validation_evaluation_events"] != 0:
            raise RuntimeError("Train-only search invariant failed; Test not re-opened")
        searches.append(search)
    _write_json(
        root / "SEARCH_COMPLETE.json",
        {
            "config_id": config_id,
            "completed_utc": _utc(),
            "search_wall_seconds": time.perf_counter() - search_started,
            "searches": len(outcomes),
            "all_return_codes_zero": True,
            "test_files_opened_during_search": False,
            "validation_evaluation_events": 0,
        },
    )
    print(
        "all Train-only P10/G10 searches complete; beginning exploratory Test reuse",
        flush=True,
    )
    evaluations = _evaluate_test(
        root,
        outcomes,
        SimpleNamespace(data_dir=args.data_dir),
        config_id,
    )
    aggregate = _aggregate(evaluations)
    aggregate.update(
        {
            "config_id": config_id,
            "evidence_status": "exploratory_reuse_of_previously_consumed_test",
            "test_is_fresh_locked_holdout": False,
        }
    )
    _write_json(root / "aggregate.json", aggregate)
    (root / "SUMMARY.md").write_text(
        _summary(aggregate, args, config_id), encoding="utf-8"
    )
    print(f"exploratory evaluation complete for {len(evaluations)} runs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
