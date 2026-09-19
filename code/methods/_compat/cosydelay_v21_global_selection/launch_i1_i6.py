"""Run formal V21 sequentially on I1--I6 and produce an aggregate summary."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    records = []
    for intersection in range(1, 7):
        parent = args.output_root / f"intersection_{intersection:02d}"
        parent.mkdir()
        output = parent / "run_01"
        started = time.perf_counter()
        stdout_path = parent / "run_01.stdout.log"
        stderr_path = parent / "run_01.stderr.log"
        # Stream the complete child output to this terminal while retaining an
        # identical per-intersection log. stderr is merged so warnings and
        # tracebacks are visible immediately and in their correct order.
        stderr_path.write_text("stderr is merged into run_01.stdout.log\n", encoding="utf-8")
        print(f"\n{'=' * 70}\nV21 INTERSECTION I{intersection}\n{'=' * 70}", flush=True)
        with stdout_path.open("w", encoding="utf-8") as log:
            command = [sys.executable, "-u", "-m", "methods.cosydelay_v21_global_selection.run_p10g10_100",
                       "--intersection", str(intersection), "--output", str(output)]
            if args.data_dir is not None:
                command.extend(["--data-dir", str(args.data_dir.resolve())])
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=os.environ.copy(),
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        records.append({"intersection": intersection, "return_code": return_code,
                        "wall_seconds": time.perf_counter() - started,
                        "output": str(output.resolve())})
        (args.output_root / "progress.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if return_code:
            return return_code
    results = []
    for intersection in range(1, 7):
        path = args.output_root / f"intersection_{intersection:02d}" / "run_01" / "result.json"
        results.append(json.loads(path.read_text(encoding="utf-8")))
    metrics = [item["selected_training_metrics"] for item in results]
    summary = {
        "formal_version": "V21",
        "method_id": "cosydelay_v21_global_numeric_domain_p10g10_100_t1",
        "intersections": list(range(1, 7)),
        "mean_training_r2": sum(float(x["macro_raw_r2"]) for x in metrics) / 6,
        "mean_rmse": sum(float(x["pooled_rmse"]) for x in metrics) / 6,
        "mean_mae": sum(float(x["pooled_mae"]) for x in metrics) / 6,
        "strict_physics_passes": sum(bool(x["selected_enhanced_physics"]["joint_pass"]) for x in results),
        "successful_candidates": sum(int(x["candidate_evaluations"]) for x in results),
        "test_or_validation_accessed": any(bool(x.get("outer_validation_accessed") or x.get("test_file_opened")) for x in results),
    }
    (args.output_root / "aggregate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
