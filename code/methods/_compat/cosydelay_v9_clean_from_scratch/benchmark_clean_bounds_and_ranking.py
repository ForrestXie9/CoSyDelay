"""Training-only fixed-structure benchmark for clean bounds and ranking."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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

import optimization_lane  # noqa: E402
from constants import INTERSECTION_CONFIGS  # noqa: E402
from data_processing import load_dataset_flexible, preprocess_data_flexible  # noqa: E402
from methods.cosydelay_lbfgsb_r10_parallel4.prefit_gate import (  # noqa: E402
    evaluate_structural_prefit_gate,
)
from methods.cosydelay_lbfgsb_r10_parallel4_v3.physics_audit import (  # noqa: E402
    audit_fitted_physics,
    audit_search_physics,
)
from methods.cosydelay_v9_clean_from_scratch.clean_fitter import (  # noqa: E402
    CleanMixedRestartJacobianFitter,
)
from methods.cosydelay_v9_clean_from_scratch.diagnostics import (  # noqa: E402
    audit_parameter_quality,
)
from methods.cosydelay_v9_clean_from_scratch.metrics import (  # noqa: E402
    score_predictions,
)
from methods.cosydelay_v9_clean_from_scratch.policy import CLEAN_POLICY  # noqa: E402
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import (  # noqa: E402
    DEFAULT_DATA_DIR,
    jsonable,
    lanes_for,
)


EXPRESSIONS = (
    "Cycle_Time*a1*flow_lane/GR_phase",
    "Cycle_Time*a1*flow_lane*(1+a2*flow_lane)/(GR_phase*(1+a3*flow_lane))",
    "Cycle_Time*a1*flow_lane*(1+a2*log(1+a3*flow_lane))/GR_phase",
    "Cycle_Time*a1*flow_lane*(1+a2*(flow_lane**a3))/GR_phase",
)
DEFAULT_BOUNDS = {
    "scale": (0.001, 1000.0),
    "power_exponent": (0.05, 5.0),
    "exp_coefficient": (0.0001, 1.0),
}
EXPANDED_BOUNDS = {
    "scale": (0.0001, 1000.0),
    "power_exponent": (0.01, 8.0),
    "exp_coefficient": (0.00001, 2.0),
}
ARMS = {
    "default": DEFAULT_BOUNDS,
    "expanded": EXPANDED_BOUNDS,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_seed(intersection: int, expression: str) -> int:
    digest = hashlib.sha256(
        f"clean-bounds-v1|{intersection}|{expression}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


@contextmanager
def active_bounds(bounds):
    previous = (
        optimization_lane.DEFAULT_PARAM_BOUNDS,
        optimization_lane.POWER_EXPONENT_BOUNDS,
        optimization_lane.EXP_COEFFICIENT_BOUNDS,
    )
    optimization_lane.DEFAULT_PARAM_BOUNDS = bounds["scale"]
    optimization_lane.POWER_EXPONENT_BOUNDS = bounds["power_exponent"]
    optimization_lane.EXP_COEFFICIENT_BOUNDS = bounds["exp_coefficient"]
    try:
        yield
    finally:
        (
            optimization_lane.DEFAULT_PARAM_BOUNDS,
            optimization_lane.POWER_EXPONENT_BOUNDS,
            optimization_lane.EXP_COEFFICIENT_BOUNDS,
        ) = previous


def fitness(metrics) -> float:
    return float(
        min(
            max(
                metrics["macro_nonnegative_r2"]
                + CLEAN_POLICY.physical_consistency_bonus,
                CLEAN_POLICY.fitness_lower_bound,
            ),
            CLEAN_POLICY.fitness_upper_bound,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    structural_audit = {}
    for expression in EXPRESSIONS:
        prefit = evaluate_structural_prefit_gate(expression)
        endpoints = audit_search_physics(expression, standard_joint_pass=prefit.passed)
        if not (
            prefit.passed
            and endpoints["symbolic_r8_exact"]
            and endpoints["symbolic_r9_positive_infinity"]
        ):
            raise RuntimeError(f"fixed suite is not clean-prefit physical: {expression}")
        structural_audit[expression] = {
            "prefit": prefit.to_dict(),
            "endpoints": endpoints,
        }

    rows = []
    started = time.perf_counter()
    for intersection in range(1, 7):
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
            for expression_index, expression in enumerate(EXPRESSIONS):
                seed = stable_seed(intersection, expression)
                arm_order = list(ARMS)
                if (intersection + expression_index) % 2:
                    arm_order.reverse()
                for arm in arm_order:
                    bounds = ARMS[arm]
                    diagnostics = {}
                    arm_started = time.perf_counter()
                    with active_bounds(bounds):
                        parameters = fitter.fit(
                            universal_expr=expression,
                            df=train,
                            lanes=lanes,
                            lane_to_approach=lane_to_approach,
                            approach_targets=targets,
                            intersection_id=intersection,
                            prepared_context=prepared,
                            rng=np.random.default_rng(seed),
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
                        physics = audit_fitted_physics(
                            expression, parameters, lanes
                        )
                        quality = audit_parameter_quality(
                            expression, parameters, bounds
                        )
                    rows.append(
                        {
                            "intersection_id": intersection,
                            "expression": expression,
                            "arm": arm,
                            "seed": seed,
                            "metrics": metrics,
                            "fitness": fitness(metrics),
                            "physics_joint_pass": physics["joint_pass"],
                            "parameter_quality": quality,
                            "optimizer": diagnostics,
                            "wall_seconds": time.perf_counter() - arm_started,
                        }
                    )
                    print(
                        f"I{intersection} {arm} expr={EXPRESSIONS.index(expression)+1} "
                        f"R2={metrics['macro_raw_r2']:.6f} "
                        f"RMSE={metrics['pooled_rmse']:.6f} "
                        f"physics={physics['joint_pass']}"
                    )

    pairs = []
    for intersection in range(1, 7):
        for expression in EXPRESSIONS:
            default = next(
                item
                for item in rows
                if item["intersection_id"] == intersection
                and item["expression"] == expression
                and item["arm"] == "default"
            )
            expanded = next(
                item
                for item in rows
                if item["intersection_id"] == intersection
                and item["expression"] == expression
                and item["arm"] == "expanded"
            )
            pairs.append(
                {
                    "intersection_id": intersection,
                    "expression": expression,
                    "delta_expanded_minus_default": {
                        "macro_raw_r2": expanded["metrics"]["macro_raw_r2"]
                        - default["metrics"]["macro_raw_r2"],
                        "pooled_rmse": expanded["metrics"]["pooled_rmse"]
                        - default["metrics"]["pooled_rmse"],
                        "pooled_mae": expanded["metrics"]["pooled_mae"]
                        - default["metrics"]["pooled_mae"],
                        "fitness": expanded["fitness"] - default["fitness"],
                        "wall_seconds": expanded["wall_seconds"]
                        - default["wall_seconds"],
                    },
                    "physics_same": (
                        expanded["physics_joint_pass"]
                        == default["physics_joint_pass"]
                    ),
                }
            )

    selected = []
    for intersection in range(1, 7):
        for arm in ARMS:
            candidates = [
                item
                for item in rows
                if item["intersection_id"] == intersection and item["arm"] == arm
            ]
            current = max(candidates, key=lambda item: item["fitness"])
            r2_only = max(
                candidates, key=lambda item: item["metrics"]["macro_raw_r2"]
            )
            selected.append(
                {
                    "intersection_id": intersection,
                    "arm": arm,
                    "current_composite": {
                        "expression": current["expression"],
                        "metrics": current["metrics"],
                        "fitness": current["fitness"],
                    },
                    "r2_only": {
                        "expression": r2_only["expression"],
                        "metrics": r2_only["metrics"],
                        "fitness": r2_only["fitness"],
                    },
                }
            )

    deltas = [item["delta_expanded_minus_default"] for item in pairs]
    summary = {
        "pairs": len(pairs),
        "expanded_r2_better": sum(item["macro_raw_r2"] > 1e-12 for item in deltas),
        "expanded_rmse_better": sum(item["pooled_rmse"] < -1e-12 for item in deltas),
        "expanded_mae_better": sum(item["pooled_mae"] < -1e-12 for item in deltas),
        "mean_delta_macro_raw_r2": float(
            np.mean([item["macro_raw_r2"] for item in deltas])
        ),
        "mean_delta_pooled_rmse": float(
            np.mean([item["pooled_rmse"] for item in deltas])
        ),
        "mean_delta_pooled_mae": float(
            np.mean([item["pooled_mae"] for item in deltas])
        ),
        "all_physics_decisions_same": all(item["physics_same"] for item in pairs),
        "default_wall_seconds_sum": float(
            sum(item["wall_seconds"] for item in rows if item["arm"] == "default")
        ),
        "expanded_wall_seconds_sum": float(
            sum(item["wall_seconds"] for item in rows if item["arm"] == "expanded")
        ),
    }
    result = {
        "schema_version": 1,
        "status": "training_only_engineering_evidence",
        "not_external_generalization_evidence": True,
        "created_utc": utc_now(),
        "data_policy": "I1-I6 Training only; no Validation/Test/history winner/API",
        "optimizer": "L-BFGS-B",
        "optimizer_restarts": CLEAN_POLICY.optimizer_restarts,
        "expressions_predeclared_in_source": list(EXPRESSIONS),
        "arms": ARMS,
        "arm_order_counterbalanced": True,
        "structural_audit": structural_audit,
        "rows": rows,
        "pairs": pairs,
        "selected_within_fixed_suite": selected,
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
