"""Run downstream signal control on observed Intersection-1 movement flows."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from experiment.config import OUTPUT_DIR
from experiment.config import FIXED_PLAN_PATH, MODEL_PATH
from experiment.controllers import require_formal_matrix_model
from experiment.demand import count_route_vehicles, generate_route_file_from_rates
from experiment.real_demand import (
    capacity_regime,
    load_intersection1_demands,
    minimum_achievable_max_saturation,
)
from experiment.simulation import append_result, run_simulation


CONTROLLERS = (
    "webster", "akcelik", "hcm", "fixed", "actuated", "max_pressure",
    "rule", "symbolic",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use only the 12 observed flow fields from Intersection 1; original "
            "green times and delay labels are never used."
        )
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--rows", nargs="+", type=int, help="1-based JSONL row IDs")
    selection.add_argument("--all-rows", action="store_true")
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=list(CONTROLLERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--duration", type=int, default=900)
    parser.add_argument("--update-interval", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _completed_keys(path: Path):
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            (
                row["controller"],
                row["demand_split"],
                int(row["demand_row_id"]),
                int(row["seed"]),
                int(row["duration_s"]),
                row["demand_sha256"],
            )
            for row in csv.DictReader(handle)
        }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(args, output: Path, selected) -> Path:
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    if args.resume and manifest_path.exists():
        return manifest_path
    source_paths = sorted({row.source_path.resolve() for row in selected})
    tracked_code = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "experiment" / "controllers.py",
        Path(__file__).resolve().parent / "experiment" / "demand.py",
        Path(__file__).resolve().parent / "experiment" / "real_demand.py",
        Path(__file__).resolve().parent / "experiment" / "simulation.py",
        Path(__file__).resolve().parent / "analyze_intersection1_real_flows.py",
    ]
    manifest = {
        "status": "preregistered_before_simulation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output.resolve()),
        "design": {
            "intersection": 1,
            "split": args.split,
            "row_ids": [row.row_id for row in selected],
            "row_count": len(selected),
            "controllers": args.controllers,
            "seeds": args.seeds,
            "duration_s": args.duration,
            "update_interval_s": args.update_interval,
            "paired_route_realization": True,
            "original_green_times_used": False,
            "delay_labels_used": False,
        },
        "capacity_strata": {
            "feasible": "minimum achievable max x <= 0.98",
            "light_oversaturation": "0.98 < minimum achievable max x <= 1.20",
            "heavy_oversaturation": "minimum achievable max x > 1.20",
        },
        "primary_estimand": (
            "paired baseline-minus-symbolic mean avg_time_loss among fully "
            "observed pairs, reported overall and by capacity stratum"
        ),
        "secondary_metrics": [
            "completion_rate", "p95_queue", "max_queue",
            "avg_waiting_time", "throughput_veh_per_hour",
        ],
        "source_files": {
            str(path): _file_hash(path) for path in source_paths
        },
        "frozen_artifacts": {
            str(MODEL_PATH.resolve()): _file_hash(MODEL_PATH),
            str(FIXED_PLAN_PATH.resolve()): _file_hash(FIXED_PLAN_PATH),
        },
        "code_sha256": {
            str(path): _file_hash(path) for path in tracked_code
        },
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest_path


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.update_interval <= 0:
        raise ValueError("duration and update interval must be positive")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("seeds must be non-negative")
    if "symbolic" in args.controllers:
        require_formal_matrix_model(MODEL_PATH)

    all_demands = load_intersection1_demands(args.split)
    by_id = {row.row_id: row for row in all_demands}
    if args.all_rows:
        selected = all_demands
    else:
        unknown = sorted(set(args.rows) - set(by_id))
        if unknown:
            raise ValueError(
                f"Unknown {args.split} row IDs {unknown}; valid range is 1..{len(all_demands)}"
            )
        selected = [by_id[row_id] for row_id in dict.fromkeys(args.rows)]

    output = args.output or OUTPUT_DIR / f"intersection1_{args.split}_real_flows.csv"
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {output}; use --resume or choose another file")
    completed = _completed_keys(output) if args.resume else set()
    manifest_path = _write_manifest(args, output, selected)
    print(f"preregistered manifest: {manifest_path}")
    routes_dir = OUTPUT_DIR / "routes" / "intersection1_real"
    trip_dir = OUTPUT_DIR / "tripinfo" / "intersection1_real"

    for demand in selected:
        minimum_max_x = minimum_achievable_max_saturation(demand.rates)
        scenario = f"intersection1_{args.split}_row_{demand.row_id:04d}"
        for seed in args.seeds:
            route_file = routes_dir / (
                f"{scenario}_seed{seed}_d{args.duration}_"
                f"{demand.demand_sha256[:12]}.rou.xml"
            )
            if not route_file.exists():
                scheduled_count = generate_route_file_from_rates(
                    route_file, demand.rates, seed, args.duration
                )
                print(f"generated {scheduled_count} vehicles: {route_file}")
            else:
                scheduled_count = count_route_vehicles(route_file)
            for controller in args.controllers:
                key = (
                    controller,
                    demand.split,
                    demand.row_id,
                    seed,
                    args.duration,
                    demand.demand_sha256,
                )
                if key in completed:
                    print(f"skipping completed {scenario} seed={seed} controller={controller}")
                    continue
                print(f"running {scenario} seed={seed} controller={controller}")
                result = run_simulation(
                    controller_name=controller,
                    scenario=scenario,
                    seed=seed,
                    duration_s=args.duration,
                    route_file=route_file,
                    output_dir=trip_dir,
                    update_interval_s=args.update_interval,
                    rates_override=demand.rates,
                )
                tripinfo_vehicles = int(result["generated"])
                result["completion_rate"] = (
                    result["completed"] / scheduled_count
                    if scheduled_count else float("nan")
                )
                result["unfinished"] = scheduled_count - result["completed"]
                result.update({
                    "demand_mode": "observed_intersection1_constant_flow",
                    "demand_split": demand.split,
                    "demand_row_id": demand.row_id,
                    "demand_source_row_id": demand.source_row_id,
                    "demand_sha256": demand.demand_sha256,
                    "demand_source_sha256": demand.source_sha256,
                    "demand_source_path": str(demand.source_path.resolve()),
                    "demand_total_vph": sum(demand.rates.values()),
                    "minimum_achievable_max_x": minimum_max_x,
                    "capacity_regime": capacity_regime(minimum_max_x),
                    "scheduled_vehicles": scheduled_count,
                    "tripinfo_vehicles": tripinfo_vehicles,
                    "route_entry_rate": (
                        tripinfo_vehicles / scheduled_count
                        if scheduled_count else float("nan")
                    ),
                })
                result.update({
                    f"demand_{lane}_vph": demand.rates[lane]
                    for lane in sorted(demand.rates)
                })
                append_result(output, result)
                completed.add(key)
                print(
                    f"  delay={result['avg_time_loss']:.2f}s "
                    f"p95_queue={result['p95_queue']:.1f} "
                    f"completed={result['completed']}/{result['generated']}"
                )

    print(output.resolve())


if __name__ == "__main__":
    main()
