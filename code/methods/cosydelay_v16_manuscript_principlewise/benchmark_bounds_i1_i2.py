"""Training-only paired coefficient-bound ablation for V16 I1/I2 winners."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

import optimization_lane
from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import (
    DEFAULT_DATA_DIR,
    jsonable,
    lanes_for,
    sha256_file,
)
from methods.prospective_optimizer_conditioning_v1.role_policy import (
    coefficient_bounds_for_expression,
    nonlinear_role_conflicts,
)

from .benchmark_maxiter_i1_i2 import PILOT_RESULTS, _fit_arm, _now
from .physics import PersistentSymbolicLimitEvaluator
from .policy import V16_POLICY


DEFAULT_BOUNDS = {
    "scale": (0.001, 1000.0),
    "power_exponent": (0.05, 5.0),
    "exp_coefficient": (0.0001, 1.0),
}
EXPANDED_BOUNDS = {
    "scale": (0.001, 1000.0),
    "power_exponent": (0.01, 8.0),
    "exp_coefficient": (0.00001, 2.0),
}
ARMS = ("default_role_bounds", "expanded_nonlinear_role_bounds")


def _stable_seed(intersection: int, expression: str) -> int:
    digest = hashlib.sha256(
        f"v16-bounds-paired-v1|{intersection}|{expression}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _delta(new: dict, reference: dict) -> dict:
    return {
        "macro_raw_r2": (
            float(new["metrics"]["macro_raw_r2"])
            - float(reference["metrics"]["macro_raw_r2"])
        ),
        "pooled_rmse": (
            float(new["metrics"]["pooled_rmse"])
            - float(reference["metrics"]["pooled_rmse"])
        ),
        "pooled_mae": (
            float(new["metrics"]["pooled_mae"])
            - float(reference["metrics"]["pooled_mae"])
        ),
        "wall_seconds": float(new["wall_seconds"]) - float(reference["wall_seconds"]),
        "wall_percent": 100.0 * (
            float(new["wall_seconds"]) / float(reference["wall_seconds"]) - 1.0
        ),
        "function_evaluations": (
            int(new["optimizer"]["total_function_evaluations"])
            - int(reference["optimizer"]["total_function_evaluations"])
        ),
        "chosen_cap_hits": (
            int(new["convergence"]["chosen_cap_hits"])
            - int(reference["convergence"]["chosen_cap_hits"])
        ),
        "near_boundary_rate": (
            float(new["parameter_quality"]["near_boundary_rate"])
            - float(reference["parameter_quality"]["near_boundary_rate"])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if V16_POLICY.coefficient_bounds != DEFAULT_BOUNDS:
        raise RuntimeError("V16 default coefficient bounds drifted")

    old_bounds = (
        optimization_lane.DEFAULT_PARAM_BOUNDS,
        optimization_lane.POWER_EXPONENT_BOUNDS,
        optimization_lane.EXP_COEFFICIENT_BOUNDS,
    )
    optimization_lane.DEFAULT_PARAM_BOUNDS = DEFAULT_BOUNDS["scale"]
    optimization_lane.POWER_EXPONENT_BOUNDS = DEFAULT_BOUNDS["power_exponent"]
    optimization_lane.EXP_COEFFICIENT_BOUNDS = DEFAULT_BOUNDS["exp_coefficient"]

    rows = []
    started_utc = _now()
    started = time.perf_counter()
    role_audit = {}
    try:
        for intersection in (1, 2):
            source_path = PILOT_RESULTS[intersection].resolve()
            source = json.loads(source_path.read_text(encoding="utf-8"))
            expression = str(source["selected_expression"])
            conflicts = nonlinear_role_conflicts(expression)
            if conflicts:
                raise RuntimeError(f"mixed nonlinear roles: {conflicts}")
            role_audit[str(intersection)] = {
                "expression": expression,
                "nonlinear_role_conflicts": conflicts,
                "default_bounds": coefficient_bounds_for_expression(
                    expression, DEFAULT_BOUNDS
                ),
                "expanded_bounds": coefficient_bounds_for_expression(
                    expression, EXPANDED_BOUNDS
                ),
            }
            warm_parameters = dict(source["selected_parameters"])
            train_path = (
                args.data_dir / f"Intersection_{intersection}_Train.jsonl"
            ).resolve()
            if "test" in train_path.name.lower() or "validation" in train_path.name.lower():
                raise RuntimeError(f"forbidden non-Training path: {train_path}")
            train = preprocess_data_flexible(
                load_dataset_flexible(str(train_path), intersection), intersection
            ).reset_index(drop=True)
            config = INTERSECTION_CONFIGS[intersection]
            targets = {
                approach: train[f"Delay_{approach}"].reset_index(drop=True)
                for approach in config["approaches"]
            }
            lanes, lane_to_approach = lanes_for(config)
            prepared = optimization_lane.prepare_optimization_context(
                train, lanes, lane_to_approach, intersection
            )
            seed = _stable_seed(intersection, expression)
            arm_order = list(ARMS)
            if intersection % 2 == 0:
                arm_order.reverse()
            limit_evaluator = PersistentSymbolicLimitEvaluator()
            try:
                with ProcessPoolExecutor(
                    max_workers=V16_POLICY.approach_workers_cap,
                    mp_context=multiprocessing.get_context("spawn"),
                ) as executor:
                    for arm in arm_order:
                        profile = (
                            DEFAULT_BOUNDS
                            if arm == ARMS[0]
                            else EXPANDED_BOUNDS
                        )
                        override = coefficient_bounds_for_expression(
                            expression,
                            profile,
                            reject_nonlinear_role_conflicts=True,
                        )
                        result = _fit_arm(
                            intersection=intersection,
                            expression=expression,
                            warm_parameters=warm_parameters,
                            maxiter=V16_POLICY.optimizer_maxiter,
                            train=train,
                            lanes=lanes,
                            lane_to_approach=lane_to_approach,
                            targets=targets,
                            prepared=prepared,
                            seed=seed,
                            executor=executor,
                            limit_evaluator=limit_evaluator,
                            bounds_override=override,
                            quality_profile=profile,
                        )
                        rows.append(
                            {
                                "intersection_id": intersection,
                                "expression": expression,
                                "arm": arm,
                                "coefficient_bounds_override": override,
                                "source_result": str(source_path),
                                "source_result_sha256": sha256_file(source_path),
                                "train_file": train_path.name,
                                "train_sha256": sha256_file(train_path),
                                "seed": seed,
                                "test_file_opened": False,
                                **result,
                            }
                        )
                        print(
                            f"I{intersection} {arm}: "
                            f"R2={result['metrics']['macro_raw_r2']:.9f} "
                            f"RMSE={result['metrics']['pooled_rmse']:.9f} "
                            f"MAE={result['metrics']['pooled_mae']:.9f} "
                            f"boundary={result['parameter_quality']['near_boundary_rate']:.3f} "
                            f"wall={result['wall_seconds']:.2f}s",
                            flush=True,
                        )
            finally:
                limit_evaluator.close()
    finally:
        (
            optimization_lane.DEFAULT_PARAM_BOUNDS,
            optimization_lane.POWER_EXPONENT_BOUNDS,
            optimization_lane.EXP_COEFFICIENT_BOUNDS,
        ) = old_bounds

    pairs = []
    for intersection in (1, 2):
        by_arm = {
            row["arm"]: row
            for row in rows
            if row["intersection_id"] == intersection
        }
        pairs.append(
            {
                "intersection_id": intersection,
                "new_arm": ARMS[1],
                "reference_arm": ARMS[0],
                "delta": _delta(by_arm[ARMS[1]], by_arm[ARMS[0]]),
                "physical_scores_same": (
                    by_arm[ARMS[1]]["physical_score"]
                    == by_arm[ARMS[0]]["physical_score"]
                    and by_arm[ARMS[1]]["physical_rule_scores"]
                    == by_arm[ARMS[0]]["physical_rule_scores"]
                ),
            }
        )
    deltas = [pair["delta"] for pair in pairs]
    summary = {
        "pairs": len(pairs),
        "mean_delta_macro_raw_r2": float(
            np.mean([item["macro_raw_r2"] for item in deltas])
        ),
        "mean_delta_pooled_rmse": float(
            np.mean([item["pooled_rmse"] for item in deltas])
        ),
        "mean_delta_pooled_mae": float(
            np.mean([item["pooled_mae"] for item in deltas])
        ),
        "mean_delta_wall_percent": float(
            np.mean([item["wall_percent"] for item in deltas])
        ),
        "mean_delta_near_boundary_rate": float(
            np.mean([item["near_boundary_rate"] for item in deltas])
        ),
        "all_physical_scores_same": all(
            pair["physical_scores_same"] for pair in pairs
        ),
        "recommendation_threshold": (
            "do not widen formal bounds unless paired Training gains are "
            "material and consistent across sites"
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "complete_training_only_paired_bounds_ablation",
        "not_a_formal_or_test_result": True,
        "started_utc": started_utc,
        "completed_utc": _now(),
        "data_policy": (
            "I1/I2 Training only; no Validation/Test/API/evolution; fixed V16 "
            "P3/G2 winners and source parameters used as paired warm starts"
        ),
        "optimizer": V16_POLICY.optimizer,
        "optimizer_restarts": V16_POLICY.optimizer_restarts,
        "optimizer_maxiter": V16_POLICY.optimizer_maxiter,
        "optimizer_maxfun": V16_POLICY.optimizer_maxfun,
        "approach_workers_cap": V16_POLICY.approach_workers_cap,
        "arms": list(ARMS),
        "default_bounds": DEFAULT_BOUNDS,
        "expanded_bounds": EXPANDED_BOUNDS,
        "role_audit": role_audit,
        "rows": rows,
        "pairs": pairs,
        "summary": summary,
        "wall_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "benchmark.json").write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
