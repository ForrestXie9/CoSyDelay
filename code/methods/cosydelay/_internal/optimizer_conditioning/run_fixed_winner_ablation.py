"""Training-only paired optimizer ablation on six frozen V9 expressions.

The expressions are literal stress cases copied from the completed V9
Training search.  This runner opens only Intersection_1--6_Train.jsonl.  It
does not perform evolution, read Validation/Test, or use accuracy from any
non-Training split.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]
if str(GMINI) not in sys.path:
    sys.path.insert(0, str(GMINI))

import optimization_lane  # noqa: E402
from constants import INTERSECTION_CONFIGS  # noqa: E402
from data_processing import load_dataset_flexible, preprocess_data_flexible  # noqa: E402
from methods.cosydelay._internal.optimizer_parallel4_v3.fitter import (  # noqa: E402
    fit_lane_parameters_mixed_jacobian_parallel,
)
from methods.cosydelay._internal.optimizer_parallel4_v3.physics_audit import (  # noqa: E402
    audit_fitted_physics,
)
from methods.cosydelay._internal.data_protocol.diagnostics import (  # noqa: E402
    audit_parameter_quality,
)
from methods.cosydelay._internal.data_protocol.metrics import (  # noqa: E402
    score_predictions,
)
from methods.cosydelay._internal.data_protocol.policy import CLEAN_POLICY  # noqa: E402
from methods.cosydelay._internal.data_protocol.run_formal_training_search import (  # noqa: E402
    DEFAULT_DATA_DIR,
    jsonable,
    lanes_for,
)
from methods.cosydelay._internal.optimizer_conditioning.all_log_fitter import (  # noqa: E402
    fit_lane_parameters_all_log_parallel,
)
from methods.cosydelay._internal.optimizer_conditioning.role_policy import (  # noqa: E402
    CONDITIONED_ROLE_BOUNDS,
    DEFAULT_ROLE_BOUNDS,
    coefficient_bounds_for_expression,
    nonlinear_role_conflicts,
)


EXPRESSIONS_BY_INTERSECTION = {
    1: (
        "Cycle_Time * flow_lane / GR_phase * (a1 * (1 + a2 * "
        "flow_lane**0.5) / (1 + a3 * log(1 + flow_lane) * (1 + a4 / "
        "GR_phase)) + a5 * (1 + a6 * flow_lane * log(1 + flow_lane)) / "
        "(1 + a7 * GR_phase * (1 + a8 * flow_lane)))"
    ),
    2: (
        "Cycle_Time * flow_lane / GR_phase * ( a1 * (1 + a2 * flow_lane * "
        "log(1 + a3 * flow_lane)) / (1 + a4 * (1 + a5 * flow_lane)**0.8 * "
        "(1 + a6 / GR_phase)) + a7 * flow_lane**1.1 * (1 + a8 * "
        "flow_lane) / (1 + a4 * (1 + flow_lane)**0.7 * (1 + a6 * "
        "GR_phase)) )"
    ),
    3: (
        "Cycle_Time * flow_lane / GR_phase * (a1 * (1 + a2 * "
        "flow_lane**a3) / (1 + a4 * flow_lane * (1 + a5 / GR_phase)) + "
        "a6 * (1 + a7 * flow_lane) / (1 + a8 * flow_lane * exp(a3 * "
        "GR_phase)))"
    ),
    4: (
        "Cycle_Time * flow_lane / GR_phase * (a1 * (1 + a2 * flow_lane * "
        "log(1 + a3 * flow_lane)) / (1 + a4 * flow_lane * (1 + a5 / "
        "GR_phase)) + a6 * flow_lane**2 / ((1 + a7 * GR_phase) * "
        "(1 + a8 * flow_lane)))"
    ),
    5: (
        "Cycle_Time * flow_lane / GR_phase * (a1 * (1 + a2 * flow_lane) / "
        "(1 + a3 * flow_lane * (1 + a4 / GR_phase)) + a5 * (1 + a6 * "
        "flow_lane) * exp(a7 * flow_lane) / (1 + a8 * GR_phase * "
        "(1 + flow_lane)))"
    ),
    6: (
        "Cycle_Time * flow_lane / GR_phase * (a1 * (1 + a2 * "
        "flow_lane)**a3 / (1 + a4 * flow_lane * (1 + a5 / GR_phase)) + "
        "a6 * exp(a7 * flow_lane) / (1 + a8 * flow_lane * GR_phase))"
    ),
}

ARM_BASELINE = "baseline_raw_local_plus_one_log_wide"
ARM_ALL_LOG = "all10_log_default_bounds"
ARM_CONDITIONED = "all10_log_strict_roles_conditioned_bounds"
ARMS = (ARM_BASELINE, ARM_ALL_LOG, ARM_CONDITIONED)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_seed(intersection: int, expression: str) -> int:
    digest = hashlib.sha256(
        f"prospective-optimizer-conditioning-v1|{intersection}|{expression}".encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def paper_fitness(metrics: Mapping[str, float]) -> float:
    return float(
        min(
            max(
                float(metrics["macro_nonnegative_r2"])
                + CLEAN_POLICY.physical_consistency_bonus,
                CLEAN_POLICY.fitness_lower_bound,
            ),
            CLEAN_POLICY.fitness_upper_bound,
        )
    )


def convergence_summary(diagnostics: Mapping[str, Any]) -> Dict[str, Any]:
    approaches = list(diagnostics.get("approaches", []))
    maxiter = int(diagnostics.get("maxiter", CLEAN_POLICY.optimizer_maxiter))
    chosen_cap_hits = sum(
        int((not bool(item.get("success"))) and int(item.get("iterations", 0)) >= maxiter)
        for item in approaches
    )
    all_restart_cap_hits = sum(
        int(int(restart.get("iterations", 0)) >= maxiter)
        for item in approaches
        for restart in item.get("restarts", [])
    )
    return {
        "approaches": len(approaches),
        "chosen_cap_hits": int(chosen_cap_hits),
        "all_restart_cap_hits": int(all_restart_cap_hits),
        "chosen_successes": int(
            sum(bool(item.get("success")) for item in approaches)
        ),
        "maximum_chosen_gradient_inf_norm": (
            float(
                max(
                    float(item.get("gradient_inf_norm_optimization_space", 0.0))
                    for item in approaches
                )
            )
            if approaches
            else None
        ),
    }


def run_fit(
    *,
    fitter: Callable[..., Dict[str, Dict[str, float]]],
    expression: str,
    train,
    lanes,
    lane_to_approach,
    targets,
    intersection: int,
    prepared,
    seed: int,
    executor: ProcessPoolExecutor,
    bounds_override: Optional[Mapping[str, tuple[float, float]]],
    quality_profile: Mapping[str, tuple[float, float]],
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {}
    started = time.perf_counter()
    parameters = fitter(
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
        coefficient_bounds_override=bounds_override,
        parallel_workers=CLEAN_POLICY.approach_workers_cap,
        maxiter=CLEAN_POLICY.optimizer_maxiter,
        maxfun=CLEAN_POLICY.optimizer_maxfun,
        executor=executor,
    )
    predictions = optimization_lane.calculate_approach_delays_from_universal(
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
    physics = audit_fitted_physics(expression, parameters, lanes)
    quality = audit_parameter_quality(expression, parameters, quality_profile)
    return {
        "status": "evaluated",
        "metrics": metrics,
        "fitness": paper_fitness(metrics),
        "physics_joint_pass": bool(physics["joint_pass"]),
        "parameter_quality": quality,
        "convergence": convergence_summary(diagnostics),
        "optimizer": diagnostics,
        "wall_seconds": float(time.perf_counter() - started),
    }


def delta(new: Mapping[str, Any], old: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "macro_raw_r2": float(new["metrics"]["macro_raw_r2"])
        - float(old["metrics"]["macro_raw_r2"]),
        "pooled_rmse": float(new["metrics"]["pooled_rmse"])
        - float(old["metrics"]["pooled_rmse"]),
        "pooled_mae": float(new["metrics"]["pooled_mae"])
        - float(old["metrics"]["pooled_mae"]),
        "fitness": float(new["fitness"]) - float(old["fitness"]),
        "wall_seconds": float(new["wall_seconds"]) - float(old["wall_seconds"]),
        "function_evaluations": float(
            new["optimizer"]["total_function_evaluations"]
        )
        - float(old["optimizer"]["total_function_evaluations"]),
        "chosen_cap_hits": float(new["convergence"]["chosen_cap_hits"])
        - float(old["convergence"]["chosen_cap_hits"]),
        "near_boundary_rate": float(
            new["parameter_quality"]["near_boundary_rate"]
        )
        - float(old["parameter_quality"]["near_boundary_rate"]),
    }


def aggregate_pairs(pairs: list[Mapping[str, Any]]) -> Dict[str, Any]:
    if not pairs:
        return {"pairs": 0}
    deltas = [item["delta"] for item in pairs]
    return {
        "pairs": len(pairs),
        "r2_better": int(sum(item["macro_raw_r2"] > 1e-12 for item in deltas)),
        "rmse_better": int(sum(item["pooled_rmse"] < -1e-12 for item in deltas)),
        "mae_better": int(sum(item["pooled_mae"] < -1e-12 for item in deltas)),
        "mean_delta_macro_raw_r2": float(
            np.mean([item["macro_raw_r2"] for item in deltas])
        ),
        "mean_delta_pooled_rmse": float(
            np.mean([item["pooled_rmse"] for item in deltas])
        ),
        "mean_delta_pooled_mae": float(
            np.mean([item["pooled_mae"] for item in deltas])
        ),
        "mean_delta_wall_seconds": float(
            np.mean([item["wall_seconds"] for item in deltas])
        ),
        "sum_delta_function_evaluations": int(
            sum(item["function_evaluations"] for item in deltas)
        ),
        "sum_delta_chosen_cap_hits": int(
            sum(item["chosen_cap_hits"] for item in deltas)
        ),
        "mean_delta_near_boundary_rate": float(
            np.mean([item["near_boundary_rate"] for item in deltas])
        ),
        "all_physics_decisions_same": bool(
            all(item["physics_same"] for item in pairs)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--intersections",
        type=int,
        nargs="+",
        default=list(range(1, 7)),
        choices=range(1, 7),
        help="Training intersections to run; defaults to I1--I6.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    intersections = list(dict.fromkeys(args.intersections))

    role_audit = {
        str(intersection): {
            "expression": expression,
            "nonlinear_role_conflicts": nonlinear_role_conflicts(expression),
        }
        for intersection, expression in EXPRESSIONS_BY_INTERSECTION.items()
        if intersection in intersections
    }
    rows: list[Dict[str, Any]] = []
    started = time.perf_counter()
    for intersection in intersections:
        expression = EXPRESSIONS_BY_INTERSECTION[intersection]
        train_path = (
            args.data_dir / f"Intersection_{intersection}_Train.jsonl"
        ).resolve()
        if "test" in train_path.name.lower() or "validation" in train_path.name.lower():
            raise RuntimeError(f"forbidden non-Training path: {train_path}")
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
        seed = stable_seed(intersection, expression)
        arm_order = list(ARMS)
        if intersection % 2 == 0:
            arm_order[:2] = reversed(arm_order[:2])

        with ProcessPoolExecutor(
            max_workers=CLEAN_POLICY.approach_workers_cap,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            for arm in arm_order:
                conflicts = nonlinear_role_conflicts(expression)
                if arm == ARM_CONDITIONED and conflicts:
                    row = {
                        "intersection_id": intersection,
                        "expression": expression,
                        "arm": arm,
                        "seed": seed,
                        "status": "rejected_role_conflict",
                        "nonlinear_role_conflicts": conflicts,
                        "test_file_opened": False,
                    }
                    rows.append(row)
                    print(f"I{intersection} {arm}: rejected {conflicts}", flush=True)
                    continue

                if arm == ARM_BASELINE:
                    fitter = fit_lane_parameters_mixed_jacobian_parallel
                    bounds_override = None
                    quality_profile = DEFAULT_ROLE_BOUNDS
                elif arm == ARM_ALL_LOG:
                    fitter = fit_lane_parameters_all_log_parallel
                    bounds_override = None
                    quality_profile = DEFAULT_ROLE_BOUNDS
                else:
                    fitter = fit_lane_parameters_all_log_parallel
                    bounds_override = coefficient_bounds_for_expression(
                        expression,
                        CONDITIONED_ROLE_BOUNDS,
                        reject_nonlinear_role_conflicts=True,
                    )
                    quality_profile = CONDITIONED_ROLE_BOUNDS

                fit_result = run_fit(
                    fitter=fitter,
                    expression=expression,
                    train=train,
                    lanes=lanes,
                    lane_to_approach=lane_to_approach,
                    targets=targets,
                    intersection=intersection,
                    prepared=prepared,
                    seed=seed,
                    executor=executor,
                    bounds_override=bounds_override,
                    quality_profile=quality_profile,
                )
                row = {
                    "intersection_id": intersection,
                    "expression": expression,
                    "arm": arm,
                    "seed": seed,
                    "train_file_name": train_path.name,
                    "train_rows": len(train),
                    "coefficient_bounds_override": bounds_override,
                    "nonlinear_role_conflicts": conflicts,
                    "test_file_opened": False,
                    **fit_result,
                }
                rows.append(row)
                print(
                    f"I{intersection} {arm}: "
                    f"R2={fit_result['metrics']['macro_raw_r2']:.8f} "
                    f"RMSE={fit_result['metrics']['pooled_rmse']:.8f} "
                    f"MAE={fit_result['metrics']['pooled_mae']:.8f} "
                    f"cap={fit_result['convergence']['chosen_cap_hits']} "
                    f"wall={fit_result['wall_seconds']:.2f}s",
                    flush=True,
                )

    all_log_pairs = []
    conditioned_pairs = []
    for intersection in intersections:
        evaluated = [
            row
            for row in rows
            if row["intersection_id"] == intersection
            and row.get("status") == "evaluated"
        ]
        by_arm = {row["arm"]: row for row in evaluated}
        baseline = by_arm[ARM_BASELINE]
        all_log = by_arm[ARM_ALL_LOG]
        all_log_pairs.append(
            {
                "intersection_id": intersection,
                "new_arm": ARM_ALL_LOG,
                "reference_arm": ARM_BASELINE,
                "delta": delta(all_log, baseline),
                "physics_same": (
                    all_log["physics_joint_pass"] == baseline["physics_joint_pass"]
                ),
            }
        )
        conditioned = by_arm.get(ARM_CONDITIONED)
        if conditioned is not None:
            conditioned_pairs.append(
                {
                    "intersection_id": intersection,
                    "new_arm": ARM_CONDITIONED,
                    "reference_arm": ARM_ALL_LOG,
                    "delta": delta(conditioned, all_log),
                    "physics_same": (
                        conditioned["physics_joint_pass"]
                        == all_log["physics_joint_pass"]
                    ),
                }
            )

    summary = {
        "all_log_vs_baseline": aggregate_pairs(all_log_pairs),
        "conditioned_bounds_vs_all_log_on_role_clean_cases": aggregate_pairs(
            conditioned_pairs
        ),
        "role_conflict_expressions": int(
            sum(bool(item["nonlinear_role_conflicts"]) for item in role_audit.values())
        ),
        "evaluated_rows": int(sum(row.get("status") == "evaluated" for row in rows)),
        "rejected_rows": int(
            sum(row.get("status") == "rejected_role_conflict" for row in rows)
        ),
    }
    result = {
        "schema_version": 1,
        "method_id": "prospective_optimizer_conditioning_v1",
        "status": "training_only_engineering_ablation",
        "not_external_generalization_evidence": True,
        "created_utc": utc_now(),
        "data_policy": (
            "I1-I6 Training only; no Validation/Test/API/evolution; fixed V9 "
            "Training-selected expressions used only as optimizer stress cases"
        ),
        "test_file_opened": False,
        "optimizer": "L-BFGS-B",
        "optimizer_restarts": CLEAN_POLICY.optimizer_restarts,
        "optimizer_maxiter": CLEAN_POLICY.optimizer_maxiter,
        "optimizer_maxfun": CLEAN_POLICY.optimizer_maxfun,
        "objective": "unchanged per-approach MSE",
        "fitness": CLEAN_POLICY.fitness_definition,
        "expressions_predeclared_in_source": EXPRESSIONS_BY_INTERSECTION,
        "intersections_run": intersections,
        "default_role_bounds": DEFAULT_ROLE_BOUNDS,
        "conditioned_role_bounds": CONDITIONED_ROLE_BOUNDS,
        "role_audit": role_audit,
        "arms": list(ARMS),
        "first_two_arm_order_counterbalanced_by_intersection": True,
        "rows": rows,
        "all_log_pairs": all_log_pairs,
        "conditioned_pairs": conditioned_pairs,
        "summary": summary,
        "wall_seconds": float(time.perf_counter() - started),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "benchmark.json").write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
