"""Run one Train-only exploratory CoSyDelay search at a requested P/G budget."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


GMINI_DIR = Path(__file__).resolve().parents[2]
if str(GMINI_DIR) not in sys.path:
    sys.path.insert(0, str(GMINI_DIR))

from constants import INTERSECTION_CONFIGS  # noqa: E402
from data_processing import load_dataset_flexible, preprocess_data_flexible  # noqa: E402
from population_evolution_lane import evolve_universal_lane_expression  # noqa: E402
from reviewer_revision_experiments.evolution_matrix.shared import to_jsonable  # noqa: E402

from . import OFFICIAL_RESTARTS, install_retained_evolution  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", type=int, required=True)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260712)
    parser.add_argument("--max-wall-seconds", type=float, default=7200.0)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            to_jsonable(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _lanes(config):
    lanes = []
    mapping = {}
    for approach in config["approaches"]:
        for movement in config["movements"][approach]:
            lane = f"{approach}_{movement}"
            lanes.append(lane)
            mapping[lane] = approach
    return lanes, mapping


def main() -> int:
    args = _arguments()
    if args.intersection not in INTERSECTION_CONFIGS:
        raise ValueError(f"unknown intersection {args.intersection}")
    if args.run_id < 1 or args.population < 1 or args.generations < 0:
        raise ValueError("run-id/population must be positive and generations non-negative")
    args.output.mkdir(parents=True, exist_ok=True)

    config = INTERSECTION_CONFIGS[args.intersection]
    approaches = list(config["approaches"])
    train_path = args.data_dir / f"Intersection_{args.intersection}_Train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    train = preprocess_data_flexible(
        load_dataset_flexible(str(train_path), args.intersection),
        args.intersection,
    )
    lanes, lane_to_approach = _lanes(config)
    targets = {approach: train[f"Delay_{approach}"] for approach in approaches}
    evolution_seed = (
        args.base_seed
        + args.intersection * 10_000
        + args.generations * 100
        + args.population * 10
        + args.run_id
        - 1
    )

    audit_path = args.output / "llm_audit.jsonl"
    previous_audit = os.environ.get("LLM_AUDIT_LOG")
    previous_content = os.environ.get("LLM_AUDIT_INCLUDE_CONTENT")
    os.environ["LLM_AUDIT_LOG"] = str(audit_path.resolve())
    os.environ["LLM_AUDIT_INCLUDE_CONTENT"] = "false"
    started = time.perf_counter()
    try:
        with install_retained_evolution() as runtime:
            result = evolve_universal_lane_expression(
                df_train=train,
                lanes=lanes,
                lane_to_approach=lane_to_approach,
                approach_targets=targets,
                universal_features=["flow_lane", "GR_phase", "Cycle_Time"],
                generations=args.generations,
                pop_size=args.population,
                intersection_id=args.intersection,
                score_mode="binary",
                physics_weight=1.0,
                prompt_knowledge=True,
                prompt_style="standard",
                seed=evolution_seed,
                optimizer_restarts=OFFICIAL_RESTARTS,
                max_wall_seconds=args.max_wall_seconds,
                use_feasible_archive=True,
                targeted_physical_feedback=True,
            )
            configured_workers = runtime.fitter.parallel_workers
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

    expression, thought, explanation, lane_parameters, history = result
    search_summary = next(
        item
        for item in reversed(history)
        if item.get("event") == "search_budget_summary"
    )
    if any(item.get("event") == "final_validation_evaluation" for item in history):
        raise RuntimeError("exploratory search unexpectedly evaluated validation")
    candidate_id = int(search_summary["final_candidate_id"])
    candidate = next(
        item
        for item in reversed(history)
        if item.get("event") == "evaluated"
        and int(item.get("candidate_id")) == candidate_id
    )
    fit_diagnostics = candidate.get("evaluation_details", {}).get("fit", {})
    method_id = (
        "cosydelay_lbfgsb_r10_parallel4_"
        f"p{args.population}g{args.generations}_training_selection_exploratory"
    )
    output = {
        "method_id": method_id,
        "experiment_status": "exploratory_after_locked_test_consumption",
        "method_frozen": False,
        "intersection_id": args.intersection,
        "run_id": args.run_id,
        "base_seed": args.base_seed,
        "evolution_seed": evolution_seed,
        "train_path_name": train_path.name,
        "train_rows": len(train),
        "test_file_opened": False,
        "validation_evaluation_events": 0,
        "validation_accessed_during_evolution": False,
        "population": args.population,
        "generations": args.generations,
        "optimizer_restarts": OFFICIAL_RESTARTS,
        "configured_worker_cap": configured_workers,
        "actual_workers_for_final_candidate": fit_diagnostics.get("parallel_workers"),
        "approaches": approaches,
        "expression": expression,
        "thought": thought,
        "explanation": explanation,
        "lane_parameters": lane_parameters,
        "train_r2": candidate.get("train_r2"),
        "train_rmse": candidate.get("train_rmse"),
        "physical_joint_pass": candidate.get("physical_joint_pass"),
        "physical_rule_scores": candidate.get("rule_scores", {}),
        "candidate_evaluations": search_summary.get("candidate_evaluations"),
        "completed_generations": search_summary.get("completed_generations"),
        "wall_budget_exhausted": search_summary.get("wall_budget_exhausted"),
        "early_stop_triggered": search_summary.get("early_stop_triggered"),
        "early_stop_reason": search_summary.get("early_stop_reason"),
        "feasible_archive_size": search_summary.get("feasible_archive_size"),
        "final_selection_policy": search_summary.get("final_selection_policy"),
        "prefit_gate_attempts": len(prefit_audit),
        "prefit_gate_rejections": sum(
            not bool(item.get("passed")) for item in prefit_audit
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(args.output / "search_result.json", output)
    _write_json(args.output / "history.json", history)
    _write_json(args.output / "prefit_gate_audit.json", prefit_audit)
    print(
        f"I{args.intersection} run={args.run_id}: "
        f"physical={output['physical_joint_pass']}, "
        f"train_R2={output['train_r2']:.5f}, "
        f"candidates={output['candidate_evaluations']}, "
        f"generations={output['completed_generations']}/{args.generations}, "
        f"wall={output['wall_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
