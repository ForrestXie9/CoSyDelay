"""Batch SUMO replay across compatible SingleTSCBaselines intersections."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_cosydelay_six_classics_true_tso import _load_specs, _plan  # noqa: E402
from run_cosydelay_true_tso import (  # noqa: E402
    SIGNAL_EXPERIMENT,
    _run_controller,
    _sha256,
    load_scenario,
)


DEFAULT_JUNCTIONS = ("Beijing_Gaojiaoyuan", "Chengdu_Guanghua", "Tianjin_zhijingdao")
DEFAULT_PATTERNS = (
    "low_density", "high_density", "fluctuating_commuter",
    "increasing_demand", "random_perturbation",
)
DEFAULT_CONTROLLERS = (
    "cosydelay", "webster", "webster-optimized", "hcm", "hcm-optimized",
    "akcelik", "akcelik-optimized",
)


def _write_summary(rows: list[dict], output_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    group_cols = ["model_key"]
    summary = frame.groupby(group_cols, as_index=False).agg(
        n_runs=("avg_total_delay_s", "count"),
        mean_avg_total_delay_s=("avg_total_delay_s", "mean"),
        sd_avg_total_delay_s=("avg_total_delay_s", "std"),
        mean_avg_time_loss_s=("avg_time_loss_s", "mean"),
        sd_avg_time_loss_s=("avg_time_loss_s", "std"),
        mean_completion_rate=("completion_rate", "mean"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    by_junction = frame.groupby(["junction", "model_key"], as_index=False).agg(
        n_runs=("avg_total_delay_s", "count"),
        mean_avg_total_delay_s=("avg_total_delay_s", "mean"),
        sd_avg_total_delay_s=("avg_total_delay_s", "std"),
        mean_completion_rate=("completion_rate", "mean"),
    )
    by_junction.to_csv(output_dir / "summary_by_junction.csv", index=False)


def run(args: argparse.Namespace) -> int:
    specs = _load_specs()
    cosydelay_model = __import__("experiment.controllers", fromlist=["require_formal_matrix_model"]).require_formal_matrix_model(
        SIGNAL_EXPERIMENT / "models" / "symbolic_lane_model.json"
    )
    unknown = [key for key in args.controllers if key not in specs]
    if unknown:
        raise ValueError(f"Unknown controller keys: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    scenario_audit: list[dict] = []
    for junction in args.junctions:
        for pattern in args.patterns:
            env_name = f"{args.difficulty}_{pattern}"
            scenario = load_scenario(junction, env_name)
            plans = {key: _plan(key, specs[key], scenario, cosydelay_model) for key in args.controllers}
            scenario_audit.append({
                "junction": junction,
                "env_name": env_name,
                "network": str(scenario.network.resolve()),
                "network_sha256": _sha256(scenario.network),
                "route": str(scenario.route.resolve()),
                "route_sha256": _sha256(scenario.route),
                "scheduled_vehicles": scenario.scheduled_vehicles,
                "rates_vph": scenario.rates,
                "plans": plans,
            })
            for seed in args.seeds:
                for key in args.controllers:
                    print(
                        f"running {junction}/{env_name} {specs[key].name} seed={seed}",
                        flush=True,
                    )
                    result = _run_controller(
                        scenario,
                        specs[key].name.replace("/", "-"),
                        plans[key],
                        seed,
                        args.output_dir / "tripinfo",
                    )
                    result["model_key"] = key
                    rows.append(result)
                    print(
                        f"  avg_total_delay={result['avg_total_delay_s']:.3f}s "
                        f"completion={result['completed_vehicles']}/{result['scheduled_vehicles']}",
                        flush=True,
                    )

    results_path = args.output_dir / "results.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(results_path, index=False)
    _write_summary(rows, args.output_dir)
    (args.output_dir / "scenario_audit.json").write_text(
        json.dumps(scenario_audit, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "complete",
        "protocol": "true_sumo_tso_fixed_plan_six_classics",
        "junctions": args.junctions,
        "patterns": args.patterns,
        "difficulty": args.difficulty,
        "controllers": args.controllers,
        "seeds": args.seeds,
        "classical_fit_source": "original Intersection-1 Train split only",
        "cosydelay_model": str((SIGNAL_EXPERIMENT / "models" / "symbolic_lane_model.json").resolve()),
        "cosydelay_model_sha256": _sha256(SIGNAL_EXPERIMENT / "models" / "symbolic_lane_model.json"),
        "results": str(results_path.resolve()),
        "summary": str((args.output_dir / "summary.csv").resolve()),
        "summary_by_junction": str((args.output_dir / "summary_by_junction.csv").resolve()),
        "scenario_audit": str((args.output_dir / "scenario_audit.json").resolve()),
        "test_labels_used_for_selection": False,
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(results_path.resolve())
    print((args.output_dir / "summary.csv").resolve())
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junctions", nargs="+", default=list(DEFAULT_JUNCTIONS))
    parser.add_argument("--patterns", nargs="+", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--difficulty", choices=("easy", "normal"), default="normal")
    parser.add_argument("--controllers", nargs="+", default=list(DEFAULT_CONTROLLERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
