from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Tuple

from experiment.config import MAX_CLEARANCE_SECONDS, MODEL_PATH, OUTPUT_DIR, ROOT
from experiment.controllers import require_formal_matrix_model
from experiment.demand import count_route_vehicles, generate_route_file
from experiment.simulation import (
    append_result,
    build_run_identity,
    canonical_sha256,
    file_sha256,
    run_simulation,
)


CONTROLLERS = (
    "webster", "akcelik", "hcm", "fixed", "actuated", "max_pressure",
    "rule", "symbolic",
)
SCENARIOS = ("balanced", "asymmetric", "near_saturation", "turning_surge", "switch")
LOGICAL_KEY_FIELDS = (
    "scenario", "controller", "seed", "duration_s", "update_interval_s",
    "clearance_limit_s",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=list(CONTROLLERS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 31)))
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--update-interval", type=int, default=300)
    parser.add_argument("--clearance-limit", type=int, default=MAX_CLEARANCE_SECONDS)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "results.csv")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _logical_key(values: Dict) -> Tuple[str, ...]:
    return tuple(str(values[field]) for field in LOGICAL_KEY_FIELDS)


def _load_existing(path: Path) -> Tuple[Dict[Tuple[str, ...], Dict], Dict[str, Dict]]:
    if not path.exists():
        return {}, {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = set(LOGICAL_KEY_FIELDS) | {"config_sha256", "run_sha256"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"Existing result CSV lacks audit-key columns {missing}: {path}; "
                "choose a new output file instead of mixing legacy and audited rows"
            )
        by_logical: Dict[Tuple[str, ...], Dict] = {}
        by_run: Dict[str, Dict] = {}
        for row_number, row in enumerate(reader, start=2):
            logical = _logical_key(row)
            run_hash = row["run_sha256"]
            if logical in by_logical:
                raise ValueError(
                    f"Duplicate logical run key at CSV row {row_number}: "
                    f"{dict(zip(LOGICAL_KEY_FIELDS, logical))}"
                )
            if run_hash in by_run:
                raise ValueError(f"Duplicate run_sha256={run_hash} at CSV row {row_number}")
            by_logical[logical] = row
            by_run[run_hash] = row
    return by_logical, by_run


def _route_identity(scenario: str, seed: int, duration_s: int) -> Dict:
    demand_code = ROOT / "experiment" / "demand.py"
    payload = {
        "route_schema_version": 1,
        "scenario": scenario,
        "seed": int(seed),
        "duration_s": int(duration_s),
        "demand_code_sha256": file_sha256(demand_code),
    }
    return {**payload, "route_config_sha256": canonical_sha256(payload)}


def _write_manifest(args, expected_runs, route_records) -> Path:
    path = args.output.with_name(f"{args.output.stem}_manifest.json")
    expected_run_records = []
    for item in expected_runs:
        identity = item["identity"]
        expected_run_records.append({
            key: identity[key]
            for key in (
                "scenario", "controller", "seed", "duration_s", "update_interval_s",
                "clearance_limit_s", "route_sha256", "symbolic_model_sha256",
                "controller_artifact_sha256", "network_sha256", "protocol_sha256",
                "config_sha256", "run_sha256", "tripinfo_filename",
            )
        })
    design = {
        "controllers": list(args.controllers),
        "scenarios": list(args.scenarios),
        "seeds": list(args.seeds),
        "duration_s": args.duration,
        "update_interval_s": args.update_interval,
        "clearance_limit_s": args.clearance_limit,
        "output": str(args.output.resolve()),
        "expected_run_sha256": [item["identity"]["run_sha256"] for item in expected_runs],
        "expected_runs": expected_run_records,
        "routes": route_records,
    }
    manifest = {
        "schema_version": 1,
        "status": "preregistered_before_simulation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_sha256": canonical_sha256(design),
        "design": design,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not args.resume:
            raise FileExistsError(f"Manifest exists: {path}; use --resume or a new output")
        if existing.get("design_sha256") != manifest["design_sha256"]:
            raise ValueError(
                f"Resume manifest design mismatch: {path}; do not mix configurations"
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.update_interval <= 0 or args.clearance_limit <= 0:
        raise ValueError("duration, update interval, and clearance limit must be positive")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("seeds must be non-negative")
    if len(set(args.controllers)) != len(args.controllers):
        raise ValueError("--controllers contains duplicates")
    if len(set(args.scenarios)) != len(args.scenarios):
        raise ValueError("--scenarios contains duplicates")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")
    if "symbolic" in args.controllers:
        require_formal_matrix_model(MODEL_PATH)
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {args.output}; use --resume or choose another file")

    existing_by_logical, existing_by_run = _load_existing(args.output)
    routes_dir = OUTPUT_DIR / "routes" / "synthetic_audited"
    tripinfo_dir = OUTPUT_DIR / "tripinfo" / "synthetic_audited"
    expected_runs = []
    route_records = []

    for scenario in args.scenarios:
        for seed in args.seeds:
            route_identity = _route_identity(scenario, seed, args.duration)
            route_file = routes_dir / (
                f"route_{scenario}_s{seed}_d{args.duration}_"
                f"q{route_identity['route_config_sha256'][:12]}.rou.xml"
            )
            if not route_file.exists():
                scheduled = generate_route_file(route_file, scenario, seed, args.duration)
                print(f"generated {scheduled} vehicles: {route_file}")
            else:
                scheduled = count_route_vehicles(route_file)
            route_record = {
                **route_identity,
                "path": str(route_file.resolve()),
                "route_sha256": file_sha256(route_file),
                "scheduled_vehicles": scheduled,
            }
            route_records.append(route_record)
            for controller in args.controllers:
                identity = build_run_identity(
                    controller,
                    scenario,
                    seed,
                    args.duration,
                    args.update_interval,
                    route_file,
                    clearance_limit_s=args.clearance_limit,
                )
                expected_runs.append({
                    "identity": identity,
                    "route_identity": route_identity,
                    "route_file": route_file,
                    "scheduled": scheduled,
                })

    manifest_path = _write_manifest(args, expected_runs, route_records)
    print(f"preregistered manifest: {manifest_path}")

    for item in expected_runs:
        identity = item["identity"]
        logical = _logical_key(identity)
        existing = existing_by_logical.get(logical)
        if existing is not None:
            if existing["config_sha256"] != identity["config_sha256"]:
                raise ValueError(
                    "Refusing to append the same logical run key with a different "
                    f"configuration: {dict(zip(LOGICAL_KEY_FIELDS, logical))}"
                )
            if existing["run_sha256"] != identity["run_sha256"]:
                raise ValueError(
                    "Logical key/config collision produced a different run hash: "
                    f"{dict(zip(LOGICAL_KEY_FIELDS, logical))}"
                )
            print(
                f"skipping completed scenario={identity['scenario']} seed={identity['seed']} "
                f"controller={identity['controller']} run={identity['run_sha256'][:12]}"
            )
            continue
        if identity["run_sha256"] in existing_by_run:
            raise ValueError(f"run_sha256 collision: {identity['run_sha256']}")

        print(
            f"running scenario={identity['scenario']} seed={identity['seed']} "
            f"controller={identity['controller']} run={identity['run_sha256'][:12]}"
        )
        result = run_simulation(
            controller_name=identity["controller"],
            scenario=identity["scenario"],
            seed=identity["seed"],
            duration_s=identity["duration_s"],
            route_file=item["route_file"],
            output_dir=tripinfo_dir,
            update_interval_s=identity["update_interval_s"],
            scheduled_vehicles=item["scheduled"],
            run_identity=identity,
            clearance_limit_s=identity["clearance_limit_s"],
        )
        result["route_config_sha256"] = item["route_identity"]["route_config_sha256"]
        append_result(args.output, result, key_fields=("run_sha256",))
        existing_by_logical[logical] = {key: str(value) for key, value in result.items()}
        existing_by_run[identity["run_sha256"]] = existing_by_logical[logical]
        print(
            f"  total_delay={result['avg_total_delay']:.2f}s "
            f"demand_p95_queue={result['demand_window_p95_queue']:.1f} "
            f"completed={result['completed']}/{result['scheduled_vehicles']} "
            f"fully_observed={result['fully_observed']}"
        )


if __name__ == "__main__":
    main()
