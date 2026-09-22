"""Run V20 on I1--I6 sequentially and write a reproducibility summary."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    runs: list[dict] = []
    started_all = time.perf_counter()
    for intersection in range(1, 7):
        parent = args.output_root / f"intersection_{intersection:02d}"
        parent.mkdir()
        output = parent / "run_01"
        stdout_path = parent / "run_01.stdout.log"
        stderr_path = parent / "run_01.stderr.log"
        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "methods.cosydelay._internal.accelerated_search.run_i1_p10g10_100",
                    "--intersection",
                    str(intersection),
                    "--output",
                    str(output),
                ],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=os.environ.copy(),
                check=False,
            )
        runs.append(
            {
                "intersection_id": intersection,
                "return_code": completed.returncode,
                "wall_seconds": time.perf_counter() - started,
                "output": str(output.resolve()),
                "stdout": str(stdout_path.resolve()),
                "stderr": str(stderr_path.resolve()),
            }
        )
        (args.output_root / "progress.json").write_text(
            json.dumps(runs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if completed.returncode != 0:
            return completed.returncode

    result_paths = {
        i: args.output_root / f"intersection_{i:02d}" / "run_01" / "result.json"
        for i in range(1, 7)
    }
    missing = [str(path) for path in result_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing result files: {missing}")
    results = {i: json.loads(path.read_text(encoding="utf-8")) for i, path in result_paths.items()}
    metrics = {i: value["selected_training_metrics"] for i, value in results.items()}
    summaries = {i: value["fit_guard_summary"] for i, value in results.items()}
    def mean(key: str) -> float:
        return sum(float(value[key]) for value in metrics.values()) / len(metrics)
    summary = {
        "method_id": "cosydelay_v20_pairwise_simple_restarts_p10g10_100_t1",
        "generation_operation_name": "regeneration",
        "regeneration_policy": {
            "checked_attempts": 3,
            "duplicate_retry_mode": (
                "same_global_regeneration_prompt_with_rejection_reason"
            ),
            "fallback": "initialization_prompt",
            "fallback_attempts": 5,
            "fit_only_after_candidate_checks_pass": True,
        },
        "intersections": list(range(1, 7)),
        "mean_training_r2": mean("macro_raw_r2"),
        "mean_training_rmse": mean("pooled_rmse"),
        "mean_training_mae": mean("pooled_mae"),
        "strict_physics_passes": sum(
            bool(value["selected_enhanced_physics"]["joint_pass"])
            for value in results.values()
        ),
        "successful_candidates": sum(int(value["successful_candidates_counted"]) for value in summaries.values()),
        "required_candidates": sum(int(value["required_successful_candidates"]) for value in summaries.values()),
        "fit_timeouts": sum(int(value["fit_timeouts"]) for value in summaries.values()),
        "validation_or_test_supervision_present": any(
            bool(value["accuracy_or_split_supervision_present"]) for value in summaries.values()
        ),
        "i1_i6_wall_seconds": time.perf_counter() - started_all,
        "per_intersection": {str(i): metrics[i] for i in metrics},
    }
    (args.output_root / "aggregate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
