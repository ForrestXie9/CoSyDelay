"""Training-only development screen of the fixed raw3/log7 portfolio."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
import optimization_lane
from methods.cosydelay.engine.data_protocol.policy import CLEAN_POLICY
from methods.cosydelay.engine.data_protocol.run_formal_training_search import (
    DEFAULT_DATA_DIR,
    jsonable,
    lanes_for,
)
from methods.cosydelay.engine.optimizer_conditioning.coordinate_portfolio_fitter import (
    fit_lane_parameters_coordinate_portfolio_parallel,
)
from methods.cosydelay.engine.optimizer_conditioning.run_fixed_winner_ablation import (
    ARM_ALL_LOG,
    ARM_BASELINE,
    DEFAULT_ROLE_BOUNDS,
    EXPRESSIONS_BY_INTERSECTION,
    aggregate_pairs,
    delta,
    run_fit,
)
from methods.cosydelay.engine.optimizer_conditioning.run_seed_robustness import (
    SEED_LABELS,
    paired_seed,
)


ARM_PORTFOLIO = "lbfgsb_r10_raw3_log7_fixed_indices_036"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if reference.get("test_file_opened") is not False:
        raise RuntimeError("reference is not a sealed Training-only result")
    if tuple(reference.get("seed_labels_predeclared", [])) != SEED_LABELS:
        raise RuntimeError("reference seed labels differ from frozen labels")
    reference_rows = {
        (int(row["intersection_id"]), str(row["seed_label"]), str(row["arm"])): row
        for row in reference["rows"]
    }

    rows = []
    versus_baseline = []
    versus_all_log = []
    started = time.perf_counter()
    for intersection in range(1, 7):
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
        with ProcessPoolExecutor(
            max_workers=CLEAN_POLICY.approach_workers_cap,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            for label in SEED_LABELS:
                seed = paired_seed(intersection, expression, label)
                fit_result = run_fit(
                    fitter=fit_lane_parameters_coordinate_portfolio_parallel,
                    expression=expression,
                    train=train,
                    lanes=lanes,
                    lane_to_approach=lane_to_approach,
                    targets=targets,
                    intersection=intersection,
                    prepared=prepared,
                    seed=seed,
                    executor=executor,
                    bounds_override=None,
                    quality_profile=DEFAULT_ROLE_BOUNDS,
                )
                row = {
                    "intersection_id": intersection,
                    "seed_label": label,
                    "seed": seed,
                    "arm": ARM_PORTFOLIO,
                    "expression": expression,
                    "train_file_name": train_path.name,
                    "train_rows": len(train),
                    "test_file_opened": False,
                    **fit_result,
                }
                rows.append(row)
                print(
                    f"I{intersection} {label} {ARM_PORTFOLIO}: "
                    f"R2={fit_result['metrics']['macro_raw_r2']:.8f} "
                    f"RMSE={fit_result['metrics']['pooled_rmse']:.8f} "
                    f"MAE={fit_result['metrics']['pooled_mae']:.8f} "
                    f"wall={fit_result['wall_seconds']:.2f}s",
                    flush=True,
                )
                for reference_arm, destination in (
                    (ARM_BASELINE, versus_baseline),
                    (ARM_ALL_LOG, versus_all_log),
                ):
                    old = reference_rows[(intersection, label, reference_arm)]
                    destination.append(
                        {
                            "intersection_id": intersection,
                            "seed_label": label,
                            "new_arm": ARM_PORTFOLIO,
                            "reference_arm": reference_arm,
                            "delta": delta(row, old),
                            "physics_same": (
                                row["physics_joint_pass"]
                                == old["physics_joint_pass"]
                            ),
                        }
                    )

    summary = {
        "portfolio_vs_baseline": aggregate_pairs(versus_baseline),
        "portfolio_vs_all_log": aggregate_pairs(versus_all_log),
        "paired_comparisons_per_reference": len(versus_baseline),
    }
    result = {
        "schema_version": 1,
        "method_id": "prospective_optimizer_conditioning_v1",
        "status": "training_only_development_screen_seen_seed_set",
        "not_confirmation_evidence": True,
        "created_utc": utc_now(),
        "data_policy": "I1-I6 Training only; Validation and Test forbidden",
        "test_file_opened": False,
        "portfolio": {
            "optimizer": "L-BFGS-B only",
            "restarts": 10,
            "raw_restart_indices": [0, 3, 6],
            "all_log_restart_indices": [1, 2, 4, 5, 7, 8, 9],
            "objective": "unchanged per-approach MSE",
            "bounds": DEFAULT_ROLE_BOUNDS,
        },
        "reference_training_result": str(args.reference.resolve()),
        "rows": rows,
        "portfolio_vs_baseline": versus_baseline,
        "portfolio_vs_all_log": versus_all_log,
        "summary": summary,
        "wall_seconds": float(time.perf_counter() - started),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "portfolio_ablation.json").write_text(
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
