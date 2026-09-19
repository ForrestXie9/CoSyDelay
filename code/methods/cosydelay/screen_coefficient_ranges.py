"""Training-only paired screen for CoSyDelay coefficient-range profiles.

It does not evolve structures and never opens Validation/Test.  Every profile
fits the same archived strict V21 structure with the same three RNG seeds, so
only coefficient ranges can explain a difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import lanes_for
from methods.cosydelay_v16_manuscript_principlewise.evaluate_frozen_test import pooled_metrics
from methods.prospective_optimizer_conditioning_v1.all_log_fitter import fit_lane_parameters_all_log_parallel
from methods.prospective_optimizer_conditioning_v1.role_policy import coefficient_bounds_for_expression, nonlinear_role_conflicts
from optimization_lane import calculate_approach_delays_from_universal, prepare_optimization_context


PROFILES = {
    "default": {
        "scale": (0.001, 1000.0),
        "power_exponent": (0.05, 5.0),
        "exp_coefficient": (0.0001, 1.0),
    },
    "conditioned_nonlinear": {
        "scale": (0.001, 1000.0),
        "power_exponent": (0.1, 3.0),
        "exp_coefficient": (0.0001, 2.0),
    },
    "expanded_nonlinear": {
        "scale": (0.001, 1000.0),
        "power_exponent": (0.01, 8.0),
        "exp_coefficient": (0.00001, 2.0),
    },
}


def _load(path: Path, intersection: int):
    return preprocess_data_flexible(load_dataset_flexible(str(path), intersection), intersection).reset_index(drop=True)


def _evaluate(frame, expression, parameters, lanes, mapping, intersection):
    prediction = calculate_approach_delays_from_universal(
        df=frame, universal_expr=expression, lane_parameters=parameters,
        lanes=lanes, lane_to_approach=mapping, intersection_id=intersection, strict=True,
    )
    truth = {item: frame[f"Delay_{item}"].to_numpy(dtype=float) for item in INTERSECTION_CONFIGS[intersection]["approaches"]}
    return pooled_metrics(truth, prediction)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intersections", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260921, 20260922, 20260923])
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILES), default=sorted(PROFILES))
    args = parser.parse_args()
    output, data, source = args.output.resolve(), args.data_dir.resolve(), args.source_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    intersections = sorted(set(args.intersections))
    if not intersections or any(i not in range(1, 7) for i in intersections):
        raise ValueError("intersections must be selected from I1-I6")
    if not args.seeds:
        raise ValueError("at least one seed is required")
    active_profiles = list(dict.fromkeys(args.profiles))
    if "default" not in active_profiles:
        raise ValueError("default must be retained as the paired reference")
    rows: list[dict] = []
    for intersection in intersections:
        result_path = source / f"intersection_{intersection:02d}" / "run_01" / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("test_file_opened") is not False or result.get("outer_validation_accessed") is not False:
            raise RuntimeError(f"source run is not Training-only: {result_path}")
        expression = str(result["selected_expression"])
        conflicts = nonlinear_role_conflicts(expression)
        if conflicts:
            raise RuntimeError(f"mixed nonlinear roles cannot receive unambiguous bounds: {conflicts}")
        train = _load(data / f"Intersection_{intersection}_Train.jsonl", intersection)
        config = INTERSECTION_CONFIGS[intersection]
        lanes, mapping = lanes_for(config)
        targets = {item: train[f"Delay_{item}"].reset_index(drop=True) for item in config["approaches"]}
        prepared = prepare_optimization_context(train, lanes, mapping, intersection)
        for seed in args.seeds:
            for profile_name in active_profiles:
                profile = PROFILES[profile_name]
                diagnostics: dict = {}
                parameters = fit_lane_parameters_all_log_parallel(
                    universal_expr=expression, df=train, lanes=lanes, lane_to_approach=mapping,
                    approach_targets=targets, intersection_id=intersection, prepared_context=prepared,
                    rng=np.random.default_rng(int(seed)), n_restarts=10, diagnostics=diagnostics,
                    coefficient_bounds_override=coefficient_bounds_for_expression(expression, profile),
                    parallel_workers=4, maxiter=200, maxfun=20_000, r9_constrained_restart_selection=False,
                )
                metrics = _evaluate(train, expression, parameters, lanes, mapping, intersection)
                row = {
                    "intersection_id": intersection,
                    "seed": int(seed),
                    "profile": profile_name,
                    "expression": expression,
                    "bounds": coefficient_bounds_for_expression(expression, profile),
                    "r2": float(metrics["r2"]), "rmse": float(metrics["rmse"]), "mae": float(metrics["mae"]),
                    "function_evaluations": int(diagnostics["total_function_evaluations"]),
                    "wall_seconds": float(diagnostics["outer_wall_seconds"]),
                }
                rows.append(row)
                print(f"I{intersection} seed={seed} {profile_name}: R2={row['r2']:.6f}, RMSE={row['rmse']:.6f}", flush=True)
    deltas = []
    for intersection in intersections:
        for seed in args.seeds:
            base = next(item for item in rows if item["intersection_id"] == intersection and item["seed"] == seed and item["profile"] == "default")
            for profile_name in active_profiles:
                if profile_name == "default":
                    continue
                item = next(item for item in rows if item["intersection_id"] == intersection and item["seed"] == seed and item["profile"] == profile_name)
                deltas.append({
                    "intersection_id": intersection, "seed": int(seed), "profile": profile_name,
                    "delta_r2": item["r2"] - base["r2"], "delta_rmse": item["rmse"] - base["rmse"],
                    "delta_mae": item["mae"] - base["mae"], "delta_function_evaluations": item["function_evaluations"] - base["function_evaluations"],
                })
    summary = {}
    for profile_name in active_profiles:
        if profile_name == "default":
            continue
        values = [item for item in deltas if item["profile"] == profile_name]
        summary[profile_name] = {
            "pairs": len(values),
            "mean_delta_r2": float(np.mean([item["delta_r2"] for item in values])),
            "median_delta_r2": float(np.median([item["delta_r2"] for item in values])),
            "r2_wins": int(sum(item["delta_r2"] > 1e-12 for item in values)),
            "mean_delta_rmse": float(np.mean([item["delta_rmse"] for item in values])),
            "mean_delta_mae": float(np.mean([item["delta_mae"] for item in values])),
        }
    payload = {
        "status": "complete_training_only_paired_range_screen",
        "data_policy": "archived V21 strict structures and Training only; Validation/Test/API forbidden",
        "source_root": str(source), "profiles": {name: PROFILES[name] for name in active_profiles}, "intersections": intersections,
        "seeds": args.seeds, "rows": rows, "deltas": deltas, "summary": summary,
    }
    output.mkdir(parents=True)
    (output / "range_screen.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
