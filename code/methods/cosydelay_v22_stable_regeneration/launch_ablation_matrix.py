"""Launch V21/V23/V22 with one shared seed schedule and output layout.

This is a Training-search launcher only.  It never opens Validation/Test and
does not overwrite any existing experiment directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


VARIANTS = {
    "v21": "methods.cosydelay_v21_global_selection.run_p10g10_100",
    "v23": "methods.cosydelay_v23_regeneration_control.run_p10g10_100",
    "v22": "methods.cosydelay_v22_stable_regeneration.run_p10g10_100",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--intersections", type=int, nargs="+", default=list(range(1, 7)))
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=20261001)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.restarts < 1 or args.workers < 1:
        raise ValueError("restarts and workers must be positive")
    intersections = sorted(set(args.intersections))
    if not intersections or any(i not in range(1, 7) for i in intersections):
        raise ValueError("intersections must be selected from I1-I6")
    root, data = args.output_root.resolve(), args.data_dir.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    tasks = [
        (variant, restart, intersection, args.seed_base + restart)
        for variant in VARIANTS
        for restart in range(1, args.restarts + 1)
        for intersection in intersections
    ]

    def execute(task: tuple[str, int, int, int]) -> dict:
        variant, restart, intersection, seed = task
        output = root / variant / f"repeat_{restart:02d}" / "search" / f"intersection_{intersection:02d}" / "run_01"
        output.parent.mkdir(parents=True, exist_ok=True)
        log = output.parent / "search.stdout.log"
        command = [
            sys.executable, "-u", "-m", VARIANTS[variant],
            "--intersection", str(intersection), "--output", str(output),
            "--data-dir", str(data), "--seed-base", str(seed),
        ]
        with log.open("w", encoding="utf-8") as stream:
            code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy())
        if code:
            raise RuntimeError(f"{variant} restart {restart} I{intersection} failed; see {log}")
        return {
            "variant": variant, "restart": restart,
            "intersection": intersection, "seed_base": seed,
            "output": str(output),
        }

    completed: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, task): task for task in tasks}
        for future in as_completed(futures):
            item = future.result()
            completed.append(item)
            (root / "training_progress.json").write_text(
                json.dumps(sorted(completed, key=lambda x: (x["variant"], x["restart"], x["intersection"])), indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"TRAINING SEARCH DONE {item['variant']} restart={item['restart']} I{item['intersection']}",
                flush=True,
            )
    (root / "TRAINING_MANIFEST.json").write_text(
        json.dumps({
            "status": "complete_training_searches_only",
            "variants": VARIANTS,
            "intersections": intersections,
            "restarts_per_variant_intersection": args.restarts,
            "seed_schedule": "same seed_base + restart_index for every variant",
            "identical_outer_search_budget": "P10/G10",
            "validation_used": False,
            "test_used": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
