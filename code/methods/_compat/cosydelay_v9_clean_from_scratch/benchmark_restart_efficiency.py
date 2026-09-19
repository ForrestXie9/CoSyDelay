"""Training-only engineering benchmark for the fixed optimizer restart count."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]
if str(GMINI) not in sys.path:
    sys.path.insert(0, str(GMINI))

from constants import INTERSECTION_CONFIGS  # noqa: E402
from data_processing import load_dataset_flexible, preprocess_data_flexible  # noqa: E402
from methods.cosydelay_lbfgsb_r10_parallel4_v3.fitter import (  # noqa: E402
    MixedRestartJacobianFitter,
)
from methods.cosydelay_lbfgsb_r10_parallel4_v3.physics_audit import (  # noqa: E402
    audit_fitted_physics,
)
from methods.cosydelay_lbfgsb_r10_parallel4_v3.training_selection import (  # noqa: E402
    score_predictions,
)
from methods.cosydelay_v9_clean_from_scratch.policy import CLEAN_POLICY  # noqa: E402
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import (  # noqa: E402
    DEFAULT_DATA_DIR,
    SplitAccessGuard,
    jsonable,
    lanes_for,
)
import optimization_lane  # noqa: E402
from optimization_lane import (  # noqa: E402
    calculate_approach_delays_from_universal,
    prepare_optimization_context,
)


EXPRESSION_SUITE = (
    "Cycle_Time*a1*flow_lane/GR_phase",
    "Cycle_Time*a1*(flow_lane**a2)/(GR_phase**a3)",
    "Cycle_Time*flow_lane*(a1+a2*flow_lane)/(GR_phase**a3)",
    "Cycle_Time*a1*flow_lane*(1+a2*log(1+a3*flow_lane))/(GR_phase**a4)",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersections", nargs="+", type=int, default=[1, 3, 6])
    parser.add_argument("--restarts", nargs="+", type=int, default=[4, 6, 10])
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stable_seed(seed: int, intersection: int, restarts: int, expression: str) -> int:
    digest = hashlib.sha256(
        f"{seed}|{intersection}|{restarts}|{expression}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(item not in INTERSECTION_CONFIGS for item in args.intersections):
        raise ValueError("unknown intersection in benchmark request")
    if any(item < 2 for item in args.restarts):
        raise ValueError("mixed restart benchmark requires at least two starts")
    args.output.mkdir(parents=True, exist_ok=False)

    previous_bounds = {
        "scale": optimization_lane.DEFAULT_PARAM_BOUNDS,
        "power_exponent": optimization_lane.POWER_EXPONENT_BOUNDS,
        "exp_coefficient": optimization_lane.EXP_COEFFICIENT_BOUNDS,
    }
    bounds = CLEAN_POLICY.coefficient_bounds
    optimization_lane.DEFAULT_PARAM_BOUNDS = bounds["scale"]
    optimization_lane.POWER_EXPONENT_BOUNDS = bounds["power_exponent"]
    optimization_lane.EXP_COEFFICIENT_BOUNDS = bounds["exp_coefficient"]
    records = []
    data_access = []
    try:
        for intersection in args.intersections:
            train_path = (
                args.data_dir / f"Intersection_{intersection}_Train.jsonl"
            ).resolve()
            guard = SplitAccessGuard(train_path)
            guard.install()
            train = preprocess_data_flexible(
                load_dataset_flexible(str(train_path), intersection), intersection
            ).reset_index(drop=True)
            data_access.extend(guard.records)
            guard.disable()
            config = INTERSECTION_CONFIGS[intersection]
            approaches = list(config["approaches"])
            targets = {
                approach: train[f"Delay_{approach}"].reset_index(drop=True)
                for approach in approaches
            }
            lanes, lane_to_approach = lanes_for(config)
            prepared = prepare_optimization_context(
                train, lanes, lane_to_approach, intersection
            )
            for restarts in args.restarts:
                with MixedRestartJacobianFitter(
                    parallel_workers=CLEAN_POLICY.approach_workers_cap
                ) as fitter:
                    for expression in EXPRESSION_SUITE:
                        diagnostics = {}
                        started = time.perf_counter()
                        parameters = fitter.fit(
                            universal_expr=expression,
                            df=train,
                            lanes=lanes,
                            lane_to_approach=lane_to_approach,
                            approach_targets=targets,
                            intersection_id=intersection,
                            prepared_context=prepared,
                            rng=np.random.default_rng(
                                stable_seed(
                                    args.seed, intersection, restarts, expression
                                )
                            ),
                            n_restarts=restarts,
                            diagnostics=diagnostics,
                            warm_parameters=None,
                        )
                        predictions = calculate_approach_delays_from_universal(
                            df=train,
                            universal_expr=expression,
                            lane_parameters=parameters,
                            lanes=lanes,
                            lane_to_approach=lane_to_approach,
                            intersection_id=intersection,
                            prepared_context=prepared,
                            strict=True,
                        )
                        metrics = score_predictions(targets, predictions)
                        physics = audit_fitted_physics(
                            expression, parameters, lanes
                        )
                        record = {
                            "intersection_id": intersection,
                            "expression": expression,
                            "restarts": restarts,
                            "seed": stable_seed(
                                args.seed, intersection, restarts, expression
                            ),
                            "metrics": metrics,
                            "enhanced_physics_pass": physics["joint_pass"],
                            "enhanced_physics": physics,
                            "optimizer_objective_sum": diagnostics.get(
                                "objective_sum"
                            ),
                            "function_evaluations": diagnostics.get(
                                "total_function_evaluations"
                            ),
                            "gradient_evaluations": diagnostics.get(
                                "total_gradient_evaluations"
                            ),
                            "fit_outer_wall_seconds": diagnostics.get(
                                "outer_wall_seconds"
                            ),
                            "case_wall_seconds": time.perf_counter() - started,
                        }
                        records.append(record)
                        print(
                            f"I{intersection} r{restarts} "
                            f"expr={EXPRESSION_SUITE.index(expression)+1}: "
                            f"R2={metrics['macro_raw_r2']:.6f}, "
                            f"RMSE={metrics['pooled_rmse']:.6f}, "
                            f"physics={physics['joint_pass']}, "
                            f"wall={record['case_wall_seconds']:.2f}s"
                        )
    finally:
        optimization_lane.DEFAULT_PARAM_BOUNDS = previous_bounds["scale"]
        optimization_lane.POWER_EXPONENT_BOUNDS = previous_bounds[
            "power_exponent"
        ]
        optimization_lane.EXP_COEFFICIENT_BOUNDS = previous_bounds[
            "exp_coefficient"
        ]

    baseline = {
        (item["intersection_id"], item["expression"]): item
        for item in records
        if item["restarts"] == 10
    }
    comparisons = []
    for item in records:
        if item["restarts"] == 10:
            continue
        reference = baseline[(item["intersection_id"], item["expression"])]
        comparisons.append(
            {
                "intersection_id": item["intersection_id"],
                "expression": item["expression"],
                "restarts": item["restarts"],
                "r2_delta_vs_r10": (
                    item["metrics"]["macro_raw_r2"]
                    - reference["metrics"]["macro_raw_r2"]
                ),
                "pooled_rmse_delta_vs_r10": (
                    item["metrics"]["pooled_rmse"]
                    - reference["metrics"]["pooled_rmse"]
                ),
                "pooled_mae_delta_vs_r10": (
                    item["metrics"]["pooled_mae"]
                    - reference["metrics"]["pooled_mae"]
                ),
                "wall_ratio_vs_r10": (
                    item["case_wall_seconds"] / reference["case_wall_seconds"]
                ),
                "physics_same_as_r10": (
                    item["enhanced_physics_pass"]
                    == reference["enhanced_physics_pass"]
                ),
            }
        )
    output = {
        "schema_version": 1,
        "status": "engineering_only_not_expression_selection_evidence",
        "created_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "data_policy": "Training only; no Validation/Test; hand-authored suite",
        "formal_policy_unchanged_by_running_this_benchmark": True,
        "intersections": args.intersections,
        "restarts": args.restarts,
        "expressions": list(EXPRESSION_SUITE),
        "coefficient_bounds": bounds,
        "data_access_audit": data_access,
        "records": records,
        "comparisons_to_r10": comparisons,
    }
    write_json(args.output / "benchmark.json", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
