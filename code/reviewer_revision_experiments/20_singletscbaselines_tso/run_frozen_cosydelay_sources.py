"""Evaluate each I1--I6 CoSyDelay model as a frozen transfer model.

The source models are selected from their own Training-only CoSyDelay searches.  A
source expression is never refit on the SingleTSCBaselines target junction;
only a signal plan is optimized from the target route-derived demand.  This
keeps the experiment separate from the existing I1-frozen six-classics batch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[2]
GMini = PACKAGE_ROOT / "code"
SIGNAL_EXPERIMENT = GMini / "signal_optimization_experiment"
SOURCE_RUN = (
    GMini / "methods" / "cosydelay" / "experiments"
    / "cosydelay_i1_i6_single_20260826_195254"
)
for path in (GMini, SIGNAL_EXPERIMENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment.controllers import SymbolicModel, symbolic_plan  # noqa: E402
from run_cosydelay_true_tso import (  # noqa: E402
    LANES,
    _run_controller,
    _sha256,
    load_scenario,
)


DEFAULT_JUNCTIONS = (
    "Beijing_Gaojiaoyuan",
    "Chengdu_Guanghua",
    "Tianjin_zhijingdao",
)
DEFAULT_PATTERNS = (
    "low_density",
    "high_density",
    "fluctuating_commuter",
    "increasing_demand",
    "random_perturbation",
)


def _model_hash(expression: str, lane_parameters: dict) -> str:
    payload = json.dumps(
        {"expression": expression, "lane_parameters": lane_parameters},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_source_model(source_intersection: int) -> tuple[SymbolicModel, dict]:
    path = SOURCE_RUN / f"intersection_{source_intersection:02d}" / "run_01" / "result.json"
    if not path.exists():
        raise FileNotFoundError(f"CoSyDelay source result not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    expression = data.get("expression") or data.get("selected_expression")
    # ``selected_parameters`` is the complete 12-movement table in the CoSyDelay
    # result artifact.  Some legacy result objects retain a sparse
    # ``lane_parameters`` diagnostic field, so prefer the selected table.
    lane_parameters = data.get("selected_parameters") or data.get("lane_parameters")
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError(f"Source {source_intersection} has no expression: {path}")
    if not isinstance(lane_parameters, dict):
        raise ValueError(f"Source {source_intersection} has no lane parameters: {path}")
    missing = sorted(set(LANES) - set(lane_parameters))
    if missing:
        raise ValueError(
            f"Source I{source_intersection} has no fitted coefficients for movements "
            f"{missing}; it cannot be transferred to a 12-movement target without "
            "an arbitrary imputation."
        )
    model = SymbolicModel(
        expression=expression,
        lane_parameters=lane_parameters,
        model_sha256=_model_hash(expression, lane_parameters),
        schema_version=1,
        status="frozen_source_transfer",
    )
    return model, {
        "source_intersection": int(source_intersection),
        "source_result": str(path.resolve()),
        "source_result_sha256": _sha256(path),
        "expression": expression,
        "lane_parameters": lane_parameters,
        "source_selected_training_fitness": data.get("selected_training_fitness"),
        "source_selected_training_metrics": data.get("selected_training_metrics"),
        "source_selected_physics": data.get("selected_enhanced_physics"),
        "source_accessed_splits": data.get("accessed_splits"),
        "source_test_file_opened": data.get("test_file_opened"),
    }


def _write_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def _append_partial(row: dict, path: Path) -> None:
    record = dict(row)
    record["applied_plan_json"] = json.dumps(record.pop("applied_plan"), sort_keys=True)
    fields = list(record)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(record)


def run(args: argparse.Namespace) -> int:
    model, artifact = load_source_model(args.source_intersection)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tripinfo_dir = args.output_dir / "tripinfo"
    partial_path = args.output_dir / "results_partial.csv"
    manifest_path = args.output_dir / "MANIFEST.json"
    manifest = {
        "status": "running",
        "protocol": "true_sumo_tso_frozen_cosydelay_source_sensitivity",
        "source_intersection": args.source_intersection,
        "junctions": args.junctions,
        "patterns": args.patterns,
        "difficulty": args.difficulty,
        "seeds": args.seeds,
        "test_labels_used_for_selection": False,
        "source_artifact": artifact,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows: list[dict] = []
    for junction in args.junctions:
        for pattern in args.patterns:
            env_name = f"{args.difficulty}_{pattern}"
            scenario = load_scenario(junction, env_name)
            plan = symbolic_plan(scenario.rates, model)
            for seed in args.seeds:
                print(
                    f"source=I{args.source_intersection} running {junction}/{env_name} seed={seed}",
                    flush=True,
                )
                row = _run_controller(
                    scenario,
                    f"CoSyDelay-source-I{args.source_intersection}",
                    plan,
                    seed,
                    tripinfo_dir,
                    port=args.port,
                )
                row["source_intersection"] = int(args.source_intersection)
                rows.append(row)
                _append_partial(row, partial_path)
                print(
                    f"  avg_total_delay={row['avg_total_delay_s']:.3f}s "
                    f"completion={row['completed_vehicles']}/{row['scheduled_vehicles']}",
                    flush=True,
                )

    result_path = args.output_dir / "results.csv"
    _write_rows(rows, result_path)
    summary = (
        pd.DataFrame(rows)
        .groupby(["source_intersection"], as_index=False)
        .agg(
            n_runs=("avg_total_delay_s", "count"),
            mean_avg_total_delay_s=("avg_total_delay_s", "mean"),
            sd_avg_total_delay_s=("avg_total_delay_s", "std"),
            mean_avg_time_loss_s=("avg_time_loss_s", "mean"),
            sd_avg_time_loss_s=("avg_time_loss_s", "std"),
            mean_completion_rate=("completion_rate", "mean"),
        )
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    by_target = (
        pd.DataFrame(rows)
        .groupby(["source_intersection", "junction"], as_index=False)
        .agg(
            n_runs=("avg_total_delay_s", "count"),
            mean_avg_total_delay_s=("avg_total_delay_s", "mean"),
            sd_avg_total_delay_s=("avg_total_delay_s", "std"),
            mean_completion_rate=("completion_rate", "mean"),
        )
    )
    by_target.to_csv(args.output_dir / "summary_by_target_junction.csv", index=False)
    (args.output_dir / "model_artifact.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    manifest.update(
        {
            "status": "complete",
            "results": str(result_path.resolve()),
            "partial_results": str(partial_path.resolve()),
            "summary": str((args.output_dir / "summary.csv").resolve()),
            "summary_by_target_junction": str(
                (args.output_dir / "summary_by_target_junction.csv").resolve()
            ),
            "model_artifact": str((args.output_dir / "model_artifact.json").resolve()),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(result_path.resolve())
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-intersection", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--junctions", nargs="+", default=list(DEFAULT_JUNCTIONS))
    parser.add_argument("--patterns", nargs="+", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--difficulty", choices=("easy", "normal"), default="normal")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 21)))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--port", type=int, default=None,
        help="Dedicated TraCI port for this source job; omit for sequential use.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
