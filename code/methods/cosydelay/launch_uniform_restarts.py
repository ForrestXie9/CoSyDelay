"""Run identical, seed-indexed CoSyDelay Training searches without Test access."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--intersections", type=int, nargs="+", default=list(range(1, 7)))
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=20260901)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=10)
    args = parser.parse_args()
    if any(value < 1 for value in (args.restarts, args.workers, args.population, args.generations)):
        raise ValueError("restarts, workers, population, and generations must be positive")
    intersections = sorted(set(args.intersections))
    if not intersections or any(item not in range(1, 7) for item in intersections):
        raise ValueError("intersections must be selected from I1-I6")
    root, data = args.output_root.resolve(), args.data_dir.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    tasks = [
        (restart, intersection, args.seed_base + restart)
        for restart in range(1, args.restarts + 1)
        for intersection in intersections
    ]

    def execute(task: tuple[int, int, int]) -> dict:
        restart, intersection, seed = task
        # Keep the layout compatible with the frozen Validation selector:
        # repeat/search/intersection/run_01. Every repeat has the same code
        # path and the same declared search budget.
        output = root / f"repeat_{restart:02d}" / "search" / f"intersection_{intersection:02d}" / "run_01"
        output.parent.mkdir(parents=True, exist_ok=True)
        log = output.parent / "search.stdout.log"
        command = [
            sys.executable, "-u", "-m",
            "methods.cosydelay.run",
            "--intersection", str(intersection), "--output", str(output),
            "--data-dir", str(data), "--seed-base", str(seed),
            "--population", str(args.population), "--generations", str(args.generations),
        ]
        with log.open("w", encoding="utf-8") as stream:
            code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy())
        if code:
            raise RuntimeError(f"restart {restart} I{intersection} failed; see {log}")
        return {"restart": restart, "intersection": intersection, "seed_base": seed, "output": str(output)}

    completed: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, task): task for task in tasks}
        for future in as_completed(futures):
            item = future.result()
            completed.append(item)
            (root / "training_progress.json").write_text(
                json.dumps(sorted(completed, key=lambda x: (x["restart"], x["intersection"])), indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"TRAINING SEARCH DONE restart={item['restart']} I{item['intersection']}", flush=True)
    manifest = {
        "status": "complete_training_searches_only",
        "formal_version": "CoSyDelay",
        "intersections": intersections,
        "restarts_per_intersection": args.restarts,
        "seed_schedule": "seed_base + restart_index",
        "identical_search_protocol_per_restart": True,
        "validation_used": False,
        "test_used": False,
    }
    (root / "TRAINING_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
