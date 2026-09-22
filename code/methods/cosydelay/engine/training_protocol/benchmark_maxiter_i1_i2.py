"""Training-only paired maxiter ablation for the frozen V16 pilot winners."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

import optimization_lane
from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
from methods.cosydelay.engine.data_protocol.diagnostics import (
    audit_parameter_quality,
)
from methods.cosydelay.engine.data_protocol.metrics import score_predictions
from methods.cosydelay.engine.data_protocol.run_formal_training_search import (
    DEFAULT_DATA_DIR,
    jsonable,
    lanes_for,
    sha256_file,
)
from methods.cosydelay.engine.optimizer_conditioning.all_log_fitter import (
    fit_lane_parameters_all_log_parallel,
)

from .physics import (
    ManuscriptVerifierConfig,
    PersistentSymbolicLimitEvaluator,
    score_fitted_lanes_manuscript_principlewise,
)
from .policy import V16_POLICY


HERE = Path(__file__).resolve().parent
PILOT_RESULTS = {
    1: HERE / "experiments" / "v16_i1_p3g2_training_only_20260812_run01" / "result.json",
    2: HERE / "experiments" / "v16_i2_p3g2_training_only_20260812_run01" / "result.json",
}
MAXITER_ARMS = (200, 400)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_seed(intersection: int, expression: str) -> int:
    digest = hashlib.sha256(
        f"v16-maxiter-paired-v1|{intersection}|{expression}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _convergence(diagnostics: dict) -> dict:
    approaches = list(diagnostics.get("approaches", ()))
    maxiter = int(diagnostics["maxiter"])
    restarts = [
        restart
        for approach in approaches
        for restart in approach.get("restarts", ())
    ]
    return {
        "approaches": len(approaches),
        "restart_records": len(restarts),
        "restart_successes": sum(bool(item.get("success")) for item in restarts),
        "restart_cap_hits": sum(
            int(item.get("iterations", 0)) >= maxiter for item in restarts
        ),
        "chosen_successes": sum(bool(item.get("success")) for item in approaches),
        "chosen_cap_hits": sum(
            int(item.get("iterations", 0)) >= maxiter for item in approaches
        ),
        "chosen_iterations": [
            int(item.get("iterations", 0)) for item in approaches
        ],
        "chosen_gradient_inf_norms": [
            float(item.get("gradient_inf_norm_optimization_space", 0.0))
            for item in approaches
        ],
    }


def _fit_arm(
    *,
    intersection: int,
    expression: str,
    warm_parameters: dict,
    maxiter: int,
    train,
    lanes,
    lane_to_approach,
    targets,
    prepared,
    seed: int,
    executor: ProcessPoolExecutor,
    limit_evaluator: PersistentSymbolicLimitEvaluator,
    bounds_override,
    quality_profile,
) -> dict:
    diagnostics: dict = {}
    started = time.perf_counter()
    parameters = fit_lane_parameters_all_log_parallel(
        universal_expr=expression,
        df=train,
        lanes=lanes,
        lane_to_approach=lane_to_approach,
        approach_targets=targets,
        intersection_id=intersection,
        prepared_context=prepared,
        rng=np.random.default_rng(seed),
        n_restarts=V16_POLICY.optimizer_restarts,
        diagnostics=diagnostics,
        coefficient_bounds_override=bounds_override,
        warm_parameters=warm_parameters,
        parallel_workers=V16_POLICY.approach_workers_cap,
        maxiter=maxiter,
        maxfun=V16_POLICY.optimizer_maxfun,
        executor=executor,
        r9_constrained_restart_selection=False,
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
    physical_started = time.perf_counter()
    physical = score_fitted_lanes_manuscript_principlewise(
        expression,
        parameters,
        lanes,
        ("flow_lane", "GR_phase", "Cycle_Time"),
        config=ManuscriptVerifierConfig(),
        limit_evaluator=limit_evaluator,
    )
    physical_wall = time.perf_counter() - physical_started
    return {
        "status": "evaluated",
        "maxiter": maxiter,
        "maxfun": V16_POLICY.optimizer_maxfun,
        "optimizer_restarts": V16_POLICY.optimizer_restarts,
        "restart_selection": "minimum_Training_MSE",
        "same_warm_start_across_arms": True,
        "same_restart_seed_across_arms": True,
        "metrics": metrics,
        "physical_score": physical.score,
        "physical_joint_pass": physical.joint_pass,
        "physical_rule_scores": physical.rule_scores,
        "physical_wall_seconds": physical_wall,
        "parameters": parameters,
        "parameter_quality": audit_parameter_quality(
            expression, parameters, quality_profile
        ),
        "convergence": _convergence(diagnostics),
        "optimizer": diagnostics,
        "wall_seconds": time.perf_counter() - started,
    }


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

    old_bounds = (
        optimization_lane.DEFAULT_PARAM_BOUNDS,
        optimization_lane.POWER_EXPONENT_BOUNDS,
        optimization_lane.EXP_COEFFICIENT_BOUNDS,
    )
    optimization_lane.DEFAULT_PARAM_BOUNDS = V16_POLICY.coefficient_bounds["scale"]
    optimization_lane.POWER_EXPONENT_BOUNDS = V16_POLICY.coefficient_bounds[
        "power_exponent"
    ]
    optimization_lane.EXP_COEFFICIENT_BOUNDS = V16_POLICY.coefficient_bounds[
        "exp_coefficient"
    ]

    rows = []
    started_utc = _now()
    started = time.perf_counter()
    try:
        for intersection in (1, 2):
            source_path = PILOT_RESULTS[intersection].resolve()
            source = json.loads(source_path.read_text(encoding="utf-8"))
            expression = str(source["selected_expression"])
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
            arm_order = list(MAXITER_ARMS)
            if intersection % 2 == 0:
                arm_order.reverse()
            limit_evaluator = PersistentSymbolicLimitEvaluator()
            try:
                with ProcessPoolExecutor(
                    max_workers=V16_POLICY.approach_workers_cap,
                    mp_context=multiprocessing.get_context("spawn"),
                ) as executor:
                    for maxiter in arm_order:
                        result = _fit_arm(
                            intersection=intersection,
                            expression=expression,
                            warm_parameters=warm_parameters,
                            maxiter=maxiter,
                            train=train,
                            lanes=lanes,
                            lane_to_approach=lane_to_approach,
                            targets=targets,
                            prepared=prepared,
                            seed=seed,
                            executor=executor,
                            limit_evaluator=limit_evaluator,
                            bounds_override=None,
                            quality_profile=V16_POLICY.coefficient_bounds,
                        )
                        rows.append(
                            {
                                "intersection_id": intersection,
                                "expression": expression,
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
                            f"I{intersection} maxiter={maxiter}: "
                            f"R2={result['metrics']['macro_raw_r2']:.9f} "
                            f"RMSE={result['metrics']['pooled_rmse']:.9f} "
                            f"MAE={result['metrics']['pooled_mae']:.9f} "
                            f"cap={result['convergence']['chosen_cap_hits']} "
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
            int(row["maxiter"]): row
            for row in rows
            if row["intersection_id"] == intersection
        }
        pairs.append(
            {
                "intersection_id": intersection,
                "new_arm": "maxiter_400",
                "reference_arm": "maxiter_200",
                "delta": _delta(by_arm[400], by_arm[200]),
                "physical_scores_same": (
                    by_arm[400]["physical_score"] == by_arm[200]["physical_score"]
                    and by_arm[400]["physical_rule_scores"]
                    == by_arm[200]["physical_rule_scores"]
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
        "all_physical_scores_same": all(
            pair["physical_scores_same"] for pair in pairs
        ),
        "recommendation_threshold": (
            "do not change formal maxiter unless the paired Training gain is "
            "material relative to its runtime cost and consistent across sites"
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "complete_training_only_paired_maxiter_ablation",
        "not_a_formal_or_test_result": True,
        "started_utc": started_utc,
        "completed_utc": _now(),
        "data_policy": (
            "I1/I2 Training only; no Validation/Test/API/evolution; fixed V16 "
            "P3/G2 winners and source parameters used as paired warm starts"
        ),
        "optimizer": V16_POLICY.optimizer,
        "arms": list(MAXITER_ARMS),
        "optimizer_restarts": V16_POLICY.optimizer_restarts,
        "optimizer_maxfun": V16_POLICY.optimizer_maxfun,
        "approach_workers_cap": V16_POLICY.approach_workers_cap,
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
