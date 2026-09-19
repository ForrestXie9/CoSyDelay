"""Run paired repeated A/B experiments without opening locked Test files."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
    parser.add_argument("--population", type=int, default=2)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--max-wall-seconds", type=float, default=900.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=METHOD_DIR
        / "experiments"
        / "repeated_ab_i1_i6_p2g1_5x_20260803",
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


def _run_pair(
    root: Path,
    repetition: int,
    seed: int,
    intersection: int,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    processes = []
    tasks = []
    for label, prefit_mode in (("A", "none"), ("B", "conservative")):
        task_dir = (
            root
            / f"rep_{repetition:02d}_seed_{seed}"
            / f"I{intersection:02d}"
            / f"{label}_{prefit_mode}"
        )
        task_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = task_dir / "stdout.log"
        stderr_path = task_dir / "stderr.log"
        command = [
            sys.executable,
            "-m",
            "methods.cosydelay_lbfgsb_r10_parallel4.run_training_only_smoke",
            "--intersections",
            str(intersection),
            "--population",
            str(args.population),
            "--generations",
            str(args.generations),
            "--seed",
            str(seed),
            "--split-seed",
            str(args.split_seed),
            "--prefit-mode",
            prefit_mode,
            "--max-wall-seconds",
            str(args.max_wall_seconds),
            "--output",
            str(task_dir),
        ]
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        environment = dict(os.environ)
        environment["PYTHONUTF8"] = "1"
        process = subprocess.Popen(
            command,
            cwd=GMINI_DIR,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=environment,
        )
        tasks.append(
            {
                "label": label,
                "prefit_mode": prefit_mode,
                "task_dir": task_dir,
                "stdout_handle": stdout_handle,
                "stderr_handle": stderr_handle,
            }
        )
        processes.append(process)

    outcomes = []
    for process, task in zip(processes, tasks):
        return_code = process.wait()
        task["stdout_handle"].close()
        task["stderr_handle"].close()
        result_path = task["task_dir"] / f"intersection_{intersection:02d}" / "result.json"
        outcomes.append(
            {
                "repetition": repetition,
                "seed": seed,
                "intersection": intersection,
                "label": task["label"],
                "prefit_mode": task["prefit_mode"],
                "return_code": return_code,
                "result_path": str(result_path),
            }
        )
    failed = [item for item in outcomes if item["return_code"] != 0]
    if failed:
        raise RuntimeError(f"paired subprocess failure: {failed}")
    return outcomes


def _candidate_counts(result_path: Path) -> Dict[str, int]:
    history_path = result_path.with_name("history.json")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    candidates = [item for item in history if item.get("event") == "evaluated"]
    physical = [item for item in candidates if item.get("physical_joint_pass")]
    accurate_physical = [
        item
        for item in physical
        if float(item.get("train_r2", float("-inf"))) >= 0.5
    ]
    return {
        "candidate_count": len(candidates),
        "physical_candidate_count": len(physical),
        "accurate_physical_candidate_count": len(accurate_physical),
    }


def _aggregate(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for outcome in outcomes:
        result_path = Path(outcome["result_path"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        row = dict(outcome)
        row.update(result)
        row.update(_candidate_counts(result_path))
        rows.append(row)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["intersection_id"]), row["prefit_mode"])].append(row)

    groups = []
    for (intersection, mode), items in sorted(grouped.items()):
        candidate_total = sum(int(item["candidate_count"]) for item in items)
        groups.append(
            {
                "intersection": intersection,
                "prefit_mode": mode,
                "runs": len(items),
                "validation_r2": _mean_sd(item["validation_r2"] for item in items),
                "validation_rmse": _mean_sd(
                    item["validation_rmse"] for item in items
                ),
                "train_r2": _mean_sd(item["train_r2"] for item in items),
                "wall_seconds": _mean_sd(item["wall_seconds"] for item in items),
                "final_physical_passes": sum(
                    bool(item["physical_joint_pass"]) for item in items
                ),
                "final_physical_pass_rate": statistics.mean(
                    bool(item["physical_joint_pass"]) for item in items
                ),
                "final_R8_pass_rate": statistics.mean(
                    float(item["physical_rule_scores"].get("R8_zero_flow_boundary", 0))
                    for item in items
                ),
                "final_R9_pass_rate": statistics.mean(
                    float(item["physical_rule_scores"].get("R9_zero_green_limit", 0))
                    for item in items
                ),
                "candidate_total": candidate_total,
                "physical_candidate_rate": (
                    sum(int(item["physical_candidate_count"]) for item in items)
                    / candidate_total
                ),
                "accurate_physical_candidate_rate": (
                    sum(
                        int(item["accurate_physical_candidate_count"])
                        for item in items
                    )
                    / candidate_total
                ),
                "prefit_gate_rejections": sum(
                    int(item["prefit_gate_rejections"]) for item in items
                ),
            }
        )

    paired = []
    row_lookup = {
        (int(row["intersection_id"]), int(row["run_seed"]), row["prefit_mode"]): row
        for row in rows
    }
    for intersection in sorted({int(row["intersection_id"]) for row in rows}):
        deltas = []
        seeds = sorted(
            {
                int(row["run_seed"])
                for row in rows
                if int(row["intersection_id"]) == intersection
            }
        )
        for seed in seeds:
            baseline = row_lookup[(intersection, seed, "none")]
            gated = row_lookup[(intersection, seed, "conservative")]
            deltas.append(
                {
                    "seed": seed,
                    "validation_r2": gated["validation_r2"] - baseline["validation_r2"],
                    "validation_rmse": (
                        gated["validation_rmse"] - baseline["validation_rmse"]
                    ),
                    "physical_pass": (
                        int(bool(gated["physical_joint_pass"]))
                        - int(bool(baseline["physical_joint_pass"]))
                    ),
                }
            )
        paired.append(
            {
                "intersection": intersection,
                "pairs": len(deltas),
                "delta_B_minus_A": {
                    "validation_r2": _mean_sd(
                        item["validation_r2"] for item in deltas
                    ),
                    "validation_rmse": _mean_sd(
                        item["validation_rmse"] for item in deltas
                    ),
                    "physical_pass": _mean_sd(
                        item["physical_pass"] for item in deltas
                    ),
                },
                "per_seed": deltas,
            }
        )

    invariant_failures = []
    for row in rows:
        checks = {
            "test_file_opened_false": not bool(row["test_file_opened"]),
            "validation_not_used_during_evolution": not bool(
                row["validation_accessed_during_evolution"]
            ),
            "one_final_validation_event": int(row["validation_evaluation_events"])
            == 1,
            "ten_restarts": int(row["optimizer_restarts"]) == 10,
        }
        if not all(checks.values()):
            invariant_failures.append(
                {
                    "intersection": row["intersection_id"],
                    "seed": row["run_seed"],
                    "prefit_mode": row["prefit_mode"],
                    "checks": checks,
                }
            )

    return {
        "schema_version": 1,
        "rows": rows,
        "groups": groups,
        "paired_comparisons": paired,
        "protocol_invariant_failures": invariant_failures,
    }


def _markdown(aggregate: Dict[str, Any]) -> str:
    lines = [
        "# Repeated A/B: no gate versus conservative pre-fit gate",
        "",
        "A uses the training-only parallel4 fitter without the new gate; B uses",
        "the conservative structural/exact-R8 pre-fit gate. Test files remain locked.",
        "",
        "| I | Mode | Runs | Val R2 mean±sd | RMSE mean±sd | Physical | Candidate physical | Accurate+physical | Gate rejects |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in aggregate["groups"]:
        label = "A none" if group["prefit_mode"] == "none" else "B conservative"
        lines.append(
            "| I{intersection} | {label} | {runs} | {r2m:.4f} ± {r2s:.4f} | "
            "{rmsem:.4f} ± {rmses:.4f} | {passes}/{runs} | {cphys:.1%} | "
            "{aphys:.1%} | {rejects} |".format(
                intersection=group["intersection"],
                label=label,
                runs=group["runs"],
                r2m=group["validation_r2"]["mean"],
                r2s=group["validation_r2"]["sd"],
                rmsem=group["validation_rmse"]["mean"],
                rmses=group["validation_rmse"]["sd"],
                passes=group["final_physical_passes"],
                cphys=group["physical_candidate_rate"],
                aphys=group["accurate_physical_candidate_rate"],
                rejects=group["prefit_gate_rejections"],
            )
        )
    lines.extend(["", "## Paired B - A differences", ""])
    for pair in aggregate["paired_comparisons"]:
        lines.append(
            "- I{intersection}: validation R2 {r2:+.4f} ± {r2sd:.4f}; "
            "RMSE {rmse:+.4f} ± {rmsesd:.4f}; physical-pass fraction "
            "{physical:+.2f}.".format(
                intersection=pair["intersection"],
                r2=pair["delta_B_minus_A"]["validation_r2"]["mean"],
                r2sd=pair["delta_B_minus_A"]["validation_r2"]["sd"],
                rmse=pair["delta_B_minus_A"]["validation_rmse"]["mean"],
                rmsesd=pair["delta_B_minus_A"]["validation_rmse"]["sd"],
                physical=pair["delta_B_minus_A"]["physical_pass"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Protocol audit",
            "",
            f"Invariant failures: {len(aggregate['protocol_invariant_failures'])}.",
            "",
            "This small repeated experiment estimates run-to-run behavior; it is not",
            "a locked-Test performance estimate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _arguments()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    outcomes: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for repetition in range(1, args.repetitions + 1):
        seed = args.seed_start + repetition - 1
        for intersection in args.intersections:
            pair_started = time.perf_counter()
            pair = _run_pair(root, repetition, seed, intersection, args)
            outcomes.extend(pair)
            _write_json(root / "run_manifest.json", outcomes)
            print(
                f"completed rep={repetition}/{args.repetitions}, I{intersection}, "
                f"seed={seed}, pair_wall={time.perf_counter() - pair_started:.1f}s",
                flush=True,
            )
    aggregate = _aggregate(outcomes)
    aggregate["batch_wall_seconds"] = time.perf_counter() - started
    _write_json(root / "aggregate.json", aggregate)
    (root / "SUMMARY.md").write_text(
        _markdown(aggregate), encoding="utf-8"
    )
    print(
        f"completed {len(outcomes)} runs in {aggregate['batch_wall_seconds']:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
