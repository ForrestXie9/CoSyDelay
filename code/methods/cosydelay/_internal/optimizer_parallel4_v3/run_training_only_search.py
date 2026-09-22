"""Run a complete prospective v3 search without opening Validation or Test."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
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
from methods.cosydelay._internal.optimizer_parallel4_v3.integration import (  # noqa: E402
    install_v3_evolution,
)
from methods.cosydelay._internal.optimizer_parallel4_v3.policy import V3_POLICY  # noqa: E402
from population_evolution_lane import evolve_universal_lane_expression  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--population", type=int, default=V3_POLICY.population)
    parser.add_argument("--generations", type=int, default=V3_POLICY.generations)
    parser.add_argument("--cv-top-k", type=int, default=V3_POLICY.cross_validation_top_k)
    parser.add_argument("--max-wall-seconds", type=float, default=1200.0)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=GMINI.parent.parent / "Final_cosy_delay" / "jsonl_files",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--incumbent-result",
        type=Path,
        default=None,
        help="Retained v2 Training-selected result; reserves one top-k CV slot.",
    )
    return parser.parse_args()


def _lanes(config):
    lanes = []
    mapping = {}
    for approach in config["approaches"]:
        for movement in config["movements"][approach]:
            lane = f"{approach}_{movement}"
            lanes.append(lane)
            mapping[lane] = approach
    return lanes, mapping


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = _arguments()
    if args.intersection not in INTERSECTION_CONFIGS:
        raise ValueError(f"unknown intersection {args.intersection}")
    if min(args.population, args.generations, args.cv_top_k) < 1:
        raise ValueError("population, generations, and cv-top-k must be positive")
    config = INTERSECTION_CONFIGS[args.intersection]
    approaches = list(config["approaches"])
    train_path = args.data_dir / f"Intersection_{args.intersection}_Train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    train = preprocess_data_flexible(
        load_dataset_flexible(str(train_path), args.intersection),
        args.intersection,
    ).reset_index(drop=True)
    targets = {
        approach: train[f"Delay_{approach}"].reset_index(drop=True)
        for approach in approaches
    }
    lanes, lane_to_approach = _lanes(config)
    policy = replace(
        V3_POLICY,
        population=args.population,
        generations=args.generations,
        cross_validation_top_k=args.cv_top_k,
    )
    evolution_seed = args.seed + args.intersection * 10_000
    args.output.mkdir(parents=True, exist_ok=True)
    audit_path = args.output / "llm_audit.jsonl"
    previous_audit = os.environ.get("LLM_AUDIT_LOG")
    previous_content = os.environ.get("LLM_AUDIT_INCLUDE_CONTENT")
    os.environ["LLM_AUDIT_LOG"] = str(audit_path.resolve())
    os.environ["LLM_AUDIT_INCLUDE_CONTENT"] = "false"
    started = time.perf_counter()
    incumbent = None
    if args.incumbent_result is not None:
        incumbent = json.loads(args.incumbent_result.read_text(encoding="utf-8"))
        if int(incumbent["intersection_id"]) != args.intersection:
            raise ValueError("incumbent intersection does not match --intersection")
        if bool(incumbent.get("test_file_opened", False)):
            raise RuntimeError("incumbent result reports Test access")
    try:
        with install_v3_evolution(
            df_train=train,
            targets=targets,
            lanes=lanes,
            lane_to_approach=lane_to_approach,
            intersection_id=args.intersection,
            seed=evolution_seed,
            policy=policy,
        ) as runtime:
            core_result = evolve_universal_lane_expression(
                df_train=train,
                lanes=lanes,
                lane_to_approach=lane_to_approach,
                approach_targets=targets,
                universal_features=["flow_lane", "GR_phase", "Cycle_Time"],
                generations=policy.generations,
                pop_size=policy.population,
                intersection_id=args.intersection,
                score_mode="binary",
                physics_weight=1.0,
                prompt_knowledge=True,
                prompt_style="standard",
                seed=evolution_seed,
                optimizer_restarts=policy.optimizer_restarts,
                max_wall_seconds=args.max_wall_seconds,
                use_feasible_archive=True,
                targeted_physical_feedback=True,
            )
            if incumbent is not None:
                runtime.selector.evaluate_incumbent(
                    incumbent["expression"],
                    thought=incumbent.get("thought", ""),
                    explanation=incumbent.get("explanation", ""),
                )
            try:
                final_result, selection_report = runtime.selector.finalize(core_result)
            except Exception as exc:
                _write_json(
                    args.output / "selection_failure.json",
                    {
                        "method_id": policy.method_id,
                        "intersection_id": args.intersection,
                        "seed": evolution_seed,
                        "outer_validation_accessed": False,
                        "test_file_opened": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "search_records": runtime.selector.records,
                        "generated": runtime.selector.generated,
                        "cv_results": runtime.selector.last_cv_results,
                        "final_refit_attempts": runtime.selector.last_final_attempts,
                    },
                )
                raise
            prefit_audit = list(runtime.prefit_audit)
    finally:
        if previous_audit is None:
            os.environ.pop("LLM_AUDIT_LOG", None)
        else:
            os.environ["LLM_AUDIT_LOG"] = previous_audit
        if previous_content is None:
            os.environ.pop("LLM_AUDIT_INCLUDE_CONTENT", None)
        else:
            os.environ["LLM_AUDIT_INCLUDE_CONTENT"] = previous_content

    expression, thought, explanation, lane_parameters, history = final_result
    output = {
        "method_id": policy.method_id,
        "retention_status": policy.retention_status,
        "intersection_id": args.intersection,
        "seed": evolution_seed,
        "policy": policy.to_dict(),
        "train_file_name": train_path.name,
        "train_rows": len(train),
        "outer_validation_accessed": False,
        "test_file_opened": False,
        "incumbent_result": (
            str(args.incumbent_result.resolve())
            if args.incumbent_result is not None
            else None
        ),
        "incumbent_cv_slot_reserved": incumbent is not None,
        "expression": expression,
        "thought": thought,
        "explanation": explanation,
        "lane_parameters": lane_parameters,
        "selection_report": selection_report,
        "prefit_gate_attempts": len(prefit_audit),
        "prefit_gate_rejections": sum(
            not bool(item.get("passed")) for item in prefit_audit
        ),
        "wall_seconds": float(time.perf_counter() - started),
    }
    _write_json(args.output / "result.json", output)
    _write_json(args.output / "history.json", history)
    _write_json(args.output / "prefit_gate_audit.json", prefit_audit)
    selected = selection_report["selected_oof_metrics"]
    print(
        f"I{args.intersection} v3: OOF R2={selected['macro_r2']:.5f}, "
        f"RMSE={selected['macro_rmse']:.5f}, MAE={selected['macro_mae']:.5f}, "
        f"physical={selection_report['selected_enhanced_physics']['joint_pass']}, "
        f"wall={output['wall_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
