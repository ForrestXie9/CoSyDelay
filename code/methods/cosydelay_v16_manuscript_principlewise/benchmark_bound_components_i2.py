"""Training-only factorial bound ablation for the V16 I2 winner."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

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


INTERSECTION = 2
PROFILES = {
    "default": {
        "scale": (0.001, 1000.0),
        "power_exponent": (0.05, 5.0),
        "exp_coefficient": (0.0001, 1.0),
    },
    "power_expanded_only": {
        "scale": (0.001, 1000.0),
        "power_exponent": (0.01, 8.0),
        "exp_coefficient": (0.0001, 1.0),
    },
    "exp_expanded_only": {
        "scale": (0.001, 1000.0),
        "power_exponent": (0.05, 5.0),
        "exp_coefficient": (0.00001, 2.0),
    },
    "both_expanded": {
        "scale": (0.001, 1000.0),
        "power_exponent": (0.01, 8.0),
        "exp_coefficient": (0.00001, 2.0),
    },
}


def _stable_seed(expression: str) -> int:
    digest = hashlib.sha256(
        f"v16-bound-components-i2-v1|{expression}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _metric_delta(row: dict, reference: dict) -> dict:
    return {
        "macro_raw_r2": (
            float(row["metrics"]["macro_raw_r2"])
            - float(reference["metrics"]["macro_raw_r2"])
        ),
        "pooled_rmse": (
            float(row["metrics"]["pooled_rmse"])
            - float(reference["metrics"]["pooled_rmse"])
        ),
        "pooled_mae": (
            float(row["metrics"]["pooled_mae"])
            - float(reference["metrics"]["pooled_mae"])
        ),
        "wall_percent": 100.0 * (
            float(row["wall_seconds"]) / float(reference["wall_seconds"]) - 1.0
        ),
        "near_boundary_rate": (
            float(row["parameter_quality"]["near_boundary_rate"])
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
    if V16_POLICY.coefficient_bounds != PROFILES["default"]:
        raise RuntimeError("V16 default coefficient bounds drifted")

    source_path = PILOT_RESULTS[INTERSECTION].resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    expression = str(source["selected_expression"])
    conflicts = nonlinear_role_conflicts(expression)
    if conflicts:
        raise RuntimeError(f"mixed nonlinear roles: {conflicts}")
    warm_parameters = dict(source["selected_parameters"])
    train_path = (
        args.data_dir / f"Intersection_{INTERSECTION}_Train.jsonl"
    ).resolve()
    if "test" in train_path.name.lower() or "validation" in train_path.name.lower():
        raise RuntimeError(f"forbidden non-Training path: {train_path}")
    train = preprocess_data_flexible(
        load_dataset_flexible(str(train_path), INTERSECTION), INTERSECTION
    ).reset_index(drop=True)
    config = INTERSECTION_CONFIGS[INTERSECTION]
    targets = {
        approach: train[f"Delay_{approach}"].reset_index(drop=True)
        for approach in config["approaches"]
    }
    lanes, lane_to_approach = lanes_for(config)
    prepared = optimization_lane.prepare_optimization_context(
        train, lanes, lane_to_approach, INTERSECTION
    )
    seed = _stable_seed(expression)

    old_bounds = (
        optimization_lane.DEFAULT_PARAM_BOUNDS,
        optimization_lane.POWER_EXPONENT_BOUNDS,
        optimization_lane.EXP_COEFFICIENT_BOUNDS,
    )
    optimization_lane.DEFAULT_PARAM_BOUNDS = PROFILES["default"]["scale"]
    optimization_lane.POWER_EXPONENT_BOUNDS = PROFILES["default"][
        "power_exponent"
    ]
    optimization_lane.EXP_COEFFICIENT_BOUNDS = PROFILES["default"][
        "exp_coefficient"
    ]

    rows = []
    started_utc = _now()
    started = time.perf_counter()
    evaluator = PersistentSymbolicLimitEvaluator()
    try:
        with ProcessPoolExecutor(
            max_workers=V16_POLICY.approach_workers_cap,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            for arm, profile in PROFILES.items():
                override = coefficient_bounds_for_expression(
                    expression,
                    profile,
                    reject_nonlinear_role_conflicts=True,
                )
                result = _fit_arm(
                    intersection=INTERSECTION,
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
                    limit_evaluator=evaluator,
                    bounds_override=override,
                    quality_profile=profile,
                )
                rows.append(
                    {
                        "arm": arm,
                        "profile": profile,
                        "coefficient_bounds_override": override,
                        **result,
                    }
                )
                print(
                    f"{arm}: R2={result['metrics']['macro_raw_r2']:.9f} "
                    f"RMSE={result['metrics']['pooled_rmse']:.9f} "
                    f"MAE={result['metrics']['pooled_mae']:.9f} "
                    f"wall={result['wall_seconds']:.2f}s",
                    flush=True,
                )
    finally:
        evaluator.close()
        (
            optimization_lane.DEFAULT_PARAM_BOUNDS,
            optimization_lane.POWER_EXPONENT_BOUNDS,
            optimization_lane.EXP_COEFFICIENT_BOUNDS,
        ) = old_bounds

    by_arm = {row["arm"]: row for row in rows}
    reference = by_arm["default"]
    deltas = {
        arm: _metric_delta(row, reference)
        for arm, row in by_arm.items()
        if arm != "default"
    }
    interaction = {
        metric: (
            deltas["both_expanded"][metric]
            - deltas["power_expanded_only"][metric]
            - deltas["exp_expanded_only"][metric]
        )
        for metric in ("macro_raw_r2", "pooled_rmse", "pooled_mae")
    }
    physical_vectors = {
        json.dumps(row["physical_rule_scores"], sort_keys=True)
        for row in rows
    }
    summary = {
        "deltas_from_default": deltas,
        "two_factor_interaction": interaction,
        "all_physical_scores_same": len(physical_vectors) == 1,
        "interpretation_rule": (
            "retain only a bound change whose isolated Training gain is "
            "material and whose physical vector remains unchanged"
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "complete_training_only_i2_bound_component_ablation",
        "not_a_formal_or_test_result": True,
        "started_utc": started_utc,
        "completed_utc": _now(),
        "data_policy": (
            "I2 Training only; no Validation/Test/API/evolution; fixed V16 "
            "P3/G2 winner and source parameters used as common warm start"
        ),
        "intersection_id": INTERSECTION,
        "expression": expression,
        "source_result": str(source_path),
        "source_result_sha256": sha256_file(source_path),
        "train_file": train_path.name,
        "train_sha256": sha256_file(train_path),
        "seed": seed,
        "same_seed_and_warm_start_across_arms": True,
        "nonlinear_role_conflicts": conflicts,
        "profiles": PROFILES,
        "rows": rows,
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
