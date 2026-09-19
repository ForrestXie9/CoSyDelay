"""Training-only maxiter check for restart cases that reached the 200 cap."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]
if str(GMINI) not in sys.path:
    sys.path.insert(0, str(GMINI))

import optimization_lane  # noqa: E402
from constants import INTERSECTION_CONFIGS  # noqa: E402
from data_processing import load_dataset_flexible, preprocess_data_flexible  # noqa: E402
from methods.cosydelay_lbfgsb_r10_parallel4_v3.physics_audit import (  # noqa: E402
    audit_fitted_physics,
)
from methods.cosydelay_v9_clean_from_scratch.benchmark_clean_bounds_and_ranking import (  # noqa: E402
    DEFAULT_BOUNDS,
    EXPRESSIONS,
    active_bounds,
    stable_seed,
)
from methods.cosydelay_v9_clean_from_scratch.clean_fitter import (  # noqa: E402
    CleanMixedRestartJacobianFitter,
)
from methods.cosydelay_v9_clean_from_scratch.metrics import score_predictions  # noqa: E402
from methods.cosydelay_v9_clean_from_scratch.policy import CLEAN_POLICY  # noqa: E402
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import (  # noqa: E402
    DEFAULT_DATA_DIR,
    jsonable,
    lanes_for,
)


# Selected only because at least one default-bounds restart reached maxiter=200
# in the predeclared 24-pair fixed-structure benchmark, never by accuracy.
CASES = ((1, 2), (2, 2), (2, 3), (2, 4), (3, 2), (5, 2), (6, 2))
MAXITERS = (200, 400)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    rows = []
    started = time.perf_counter()
    for intersection in sorted({case[0] for case in CASES}):
        train_path = (
            args.data_dir / f"Intersection_{intersection}_Train.jsonl"
        ).resolve()
        train = preprocess_data_flexible(
            load_dataset_flexible(str(train_path), intersection), intersection
        ).reset_index(drop=True)
        config = INTERSECTION_CONFIGS[intersection]
        approaches = list(config["approaches"])
        targets = {
            approach: train[f"Delay_{approach}"].reset_index(drop=True)
            for approach in approaches
        }
        lanes, lane_to_approach = lanes_for(config)
        prepared = optimization_lane.prepare_optimization_context(
            train, lanes, lane_to_approach, intersection
        )
        with CleanMixedRestartJacobianFitter(
            parallel_workers=CLEAN_POLICY.approach_workers_cap
        ) as fitter:
            intersection_cases = [case for case in CASES if case[0] == intersection]
            for case_offset, (_, expression_index) in enumerate(intersection_cases):
                expression = EXPRESSIONS[expression_index - 1]
                order = list(MAXITERS)
                if (intersection + expression_index + case_offset) % 2:
                    order.reverse()
                for maxiter in order:
                    diagnostics = {}
                    fitter.maxiter = maxiter
                    arm_started = time.perf_counter()
                    with active_bounds(DEFAULT_BOUNDS):
                        parameters = fitter.fit(
                            universal_expr=expression,
                            df=train,
                            lanes=lanes,
                            lane_to_approach=lane_to_approach,
                            approach_targets=targets,
                            intersection_id=intersection,
                            prepared_context=prepared,
                            rng=np.random.default_rng(
                                stable_seed(intersection, expression)
                            ),
                            n_restarts=CLEAN_POLICY.optimizer_restarts,
                            diagnostics=diagnostics,
                        )
                        predictions = (
                            optimization_lane.calculate_approach_delays_from_universal(
                                df=train,
                                universal_expr=expression,
                                lane_parameters=parameters,
                                lanes=lanes,
                                lane_to_approach=lane_to_approach,
                                intersection_id=intersection,
                                prepared_context=prepared,
                                strict=True,
                            )
                        )
                        metrics = score_predictions(targets, predictions)
                        physics = audit_fitted_physics(expression, parameters, lanes)
                    chosen_cap_hits = 0
                    all_cap_hits = 0
                    for approach in diagnostics["approaches"]:
                        chosen = approach["restarts"][approach["chosen_restart"]]
                        chosen_cap_hits += int(chosen["iterations"] >= maxiter)
                        all_cap_hits += sum(
                            item["iterations"] >= maxiter
                            for item in approach["restarts"]
                        )
                    rows.append(
                        {
                            "intersection_id": intersection,
                            "expression_index": expression_index,
                            "expression": expression,
                            "maxiter": maxiter,
                            "maxfun": 20000,
                            "seed": stable_seed(intersection, expression),
                            "metrics": metrics,
                            "physics_joint_pass": physics["joint_pass"],
                            "chosen_cap_hits": chosen_cap_hits,
                            "all_restart_cap_hits": all_cap_hits,
                            "optimizer": diagnostics,
                            "wall_seconds": time.perf_counter() - arm_started,
                        }
                    )
                    print(
                        f"I{intersection} expr={expression_index} maxiter={maxiter} "
                        f"R2={metrics['macro_raw_r2']:.8f} "
                        f"RMSE={metrics['pooled_rmse']:.8f} "
                        f"chosen_cap={chosen_cap_hits}"
                    )

    pairs = []
    for intersection, expression_index in CASES:
        low = next(
            row
            for row in rows
            if row["intersection_id"] == intersection
            and row["expression_index"] == expression_index
            and row["maxiter"] == 200
        )
        high = next(
            row
            for row in rows
            if row["intersection_id"] == intersection
            and row["expression_index"] == expression_index
            and row["maxiter"] == 400
        )
        pairs.append(
            {
                "intersection_id": intersection,
                "expression_index": expression_index,
                "delta_400_minus_200": {
                    "macro_raw_r2": high["metrics"]["macro_raw_r2"]
                    - low["metrics"]["macro_raw_r2"],
                    "pooled_rmse": high["metrics"]["pooled_rmse"]
                    - low["metrics"]["pooled_rmse"],
                    "pooled_mae": high["metrics"]["pooled_mae"]
                    - low["metrics"]["pooled_mae"],
                    "wall_seconds": high["wall_seconds"] - low["wall_seconds"],
                },
                "physics_same": high["physics_joint_pass"] == low["physics_joint_pass"],
                "chosen_cap_hits_200": low["chosen_cap_hits"],
                "chosen_cap_hits_400": high["chosen_cap_hits"],
            }
        )

    deltas = [pair["delta_400_minus_200"] for pair in pairs]
    low_wall = sum(row["wall_seconds"] for row in rows if row["maxiter"] == 200)
    high_wall = sum(row["wall_seconds"] for row in rows if row["maxiter"] == 400)
    summary = {
        "pairs": len(pairs),
        "r2_better": sum(item["macro_raw_r2"] > 1e-12 for item in deltas),
        "rmse_better": sum(item["pooled_rmse"] < -1e-12 for item in deltas),
        "mae_better": sum(item["pooled_mae"] < -1e-12 for item in deltas),
        "mean_delta_macro_raw_r2": float(
            np.mean([item["macro_raw_r2"] for item in deltas])
        ),
        "mean_delta_pooled_rmse": float(
            np.mean([item["pooled_rmse"] for item in deltas])
        ),
        "mean_delta_pooled_mae": float(
            np.mean([item["pooled_mae"] for item in deltas])
        ),
        "all_physics_decisions_same": all(pair["physics_same"] for pair in pairs),
        "maxiter_200_wall_seconds_sum": float(low_wall),
        "maxiter_400_wall_seconds_sum": float(high_wall),
        "wall_ratio_400_vs_200": float(high_wall / low_wall),
        "chosen_cap_hits_200": sum(pair["chosen_cap_hits_200"] for pair in pairs),
        "chosen_cap_hits_400": sum(pair["chosen_cap_hits_400"] for pair in pairs),
    }
    result = {
        "schema_version": 1,
        "status": "training_only_convergence_engineering_evidence",
        "not_external_generalization_evidence": True,
        "created_utc": utc_now(),
        "data_policy": "I1-I6 Training only; selected by convergence cap, not accuracy",
        "optimizer": "L-BFGS-B",
        "optimizer_restarts": CLEAN_POLICY.optimizer_restarts,
        "bounds": DEFAULT_BOUNDS,
        "cases_predeclared_in_source": [list(case) for case in CASES],
        "arm_order_counterbalanced": True,
        "rows": rows,
        "pairs": pairs,
        "summary": summary,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "benchmark.json").write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
