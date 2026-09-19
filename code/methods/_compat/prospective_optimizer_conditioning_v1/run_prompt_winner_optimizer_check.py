"""Training-only paired optimizer check on the two new prompt-pilot winners."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
import optimization_lane
from methods.cosydelay_lbfgsb_r10_parallel4_v3.fitter import (
    fit_lane_parameters_mixed_jacobian_parallel,
)
from methods.cosydelay_v9_clean_from_scratch.policy import CLEAN_POLICY
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import (
    DEFAULT_DATA_DIR,
    jsonable,
    lanes_for,
)
from methods.prospective_optimizer_conditioning_v1.all_log_fitter import (
    fit_lane_parameters_all_log_parallel,
)
from methods.prospective_optimizer_conditioning_v1.run_fixed_winner_ablation import (
    ARM_ALL_LOG,
    ARM_BASELINE,
    DEFAULT_ROLE_BOUNDS,
    aggregate_pairs,
    delta,
    run_fit,
)


SEED_LABELS = ("fresh-fit-01", "fresh-fit-02", "fresh-fit-03")


def paired_seed(intersection: int, expression: str, label: str) -> int:
    digest = hashlib.sha256(
        (
            "prompt-winner-optimizer-check-v1|"
            f"{intersection}|{label}|{expression}"
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i1-result", type=Path, required=True)
    parser.add_argument("--i2-result", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    source_results = {
        1: json.loads(args.i1_result.read_text(encoding="utf-8")),
        2: json.loads(args.i2_result.read_text(encoding="utf-8")),
    }
    expressions = {}
    for intersection, result in source_results.items():
        if result.get("test_file_opened") is not False:
            raise RuntimeError("source prompt pilot is not Training-only")
        if int(result.get("intersection_id")) != intersection:
            raise RuntimeError("source prompt-pilot intersection mismatch")
        expressions[intersection] = str(result["selected_expression"])

    rows = []
    pairs = []
    started = time.perf_counter()
    for intersection in (1, 2):
        expression = expressions[intersection]
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
        with ProcessPoolExecutor(
            max_workers=CLEAN_POLICY.approach_workers_cap,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            for replicate_index, label in enumerate(SEED_LABELS):
                seed = paired_seed(intersection, expression, label)
                arm_order = [ARM_BASELINE, ARM_ALL_LOG]
                if (intersection + replicate_index) % 2 == 0:
                    arm_order.reverse()
                by_arm = {}
                for arm in arm_order:
                    fitter = (
                        fit_lane_parameters_mixed_jacobian_parallel
                        if arm == ARM_BASELINE
                        else fit_lane_parameters_all_log_parallel
                    )
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
                        bounds_override=None,
                        quality_profile=DEFAULT_ROLE_BOUNDS,
                    )
                    row = {
                        "intersection_id": intersection,
                        "seed_label": label,
                        "seed": seed,
                        "arm": arm,
                        "expression": expression,
                        "train_file_name": train_path.name,
                        "train_rows": len(train),
                        "test_file_opened": False,
                        **fit_result,
                    }
                    rows.append(row)
                    by_arm[arm] = row
                    print(
                        f"I{intersection} {label} {arm}: "
                        f"R2={fit_result['metrics']['macro_raw_r2']:.8f} "
                        f"RMSE={fit_result['metrics']['pooled_rmse']:.8f} "
                        f"MAE={fit_result['metrics']['pooled_mae']:.8f} "
                        f"wall={fit_result['wall_seconds']:.2f}s",
                        flush=True,
                    )
                baseline = by_arm[ARM_BASELINE]
                all_log = by_arm[ARM_ALL_LOG]
                pairs.append(
                    {
                        "intersection_id": intersection,
                        "seed_label": label,
                        "new_arm": ARM_ALL_LOG,
                        "reference_arm": ARM_BASELINE,
                        "delta": delta(all_log, baseline),
                        "physics_same": (
                            all_log["physics_joint_pass"]
                            == baseline["physics_joint_pass"]
                        ),
                    }
                )

    summary = {
        "overall": aggregate_pairs(pairs),
        "by_intersection": {
            str(intersection): aggregate_pairs(
                [item for item in pairs if item["intersection_id"] == intersection]
            )
            for intersection in (1, 2)
        },
        "paired_comparisons": len(pairs),
    }
    result = {
        "schema_version": 1,
        "status": "training_only_prompt_winner_optimizer_check",
        "not_external_generalization_evidence": True,
        "created_utc": utc_now(),
        "data_policy": "I1/I2 Training only; Validation and Test forbidden",
        "test_file_opened": False,
        "source_prompt_results": {
            "1": str(args.i1_result.resolve()),
            "2": str(args.i2_result.resolve()),
        },
        "expressions": expressions,
        "seed_labels_predeclared": list(SEED_LABELS),
        "common_settings": {
            "optimizer": "L-BFGS-B",
            "restarts": CLEAN_POLICY.optimizer_restarts,
            "objective": "unchanged per-approach MSE",
            "bounds": DEFAULT_ROLE_BOUNDS,
        },
        "rows": rows,
        "pairs": pairs,
        "summary": summary,
        "wall_seconds": float(time.perf_counter() - started),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "optimizer_check.json").write_text(
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
