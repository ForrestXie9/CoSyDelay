"""Run one frozen prompt-v6 efficiency-v1 P10/G10 Training search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from llm_config import load_llm_config
from methods.cosydelay.engine.data_protocol.diagnostics import (
    audit_parameter_quality,
    summarize_compute_efficiency,
)
from methods.cosydelay.engine.data_protocol.run_formal_training_search import (
    DEFAULT_DATA_DIR,
    sha256_file,
    utc_now,
    write_json,
)

from . import run_final_prompt_pilot as engine
from .policy import V10_POLICY
from .source_manifest import SOURCE_FILES, validate_source_manifest


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _verify_engine_result(result: dict, history: list[dict]) -> None:
    required = {
        "population": V10_POLICY.population,
        "generations": V10_POLICY.generations,
        "optimizer_restarts": V10_POLICY.optimizer_restarts,
        "prompt_contract_version": V10_POLICY.prompt_contract_version,
        "execution_contract_version": V10_POLICY.execution_contract_version,
        "selection_and_evaluation_scope": "Training only",
        "outer_validation_accessed": False,
        "test_file_opened": False,
    }
    mismatches = {
        key: {"observed": result.get(key), "required": value}
        for key, value in required.items()
        if result.get(key) != value
    }
    budget = next(
        item
        for item in reversed(history)
        if item.get("event") == "search_budget_summary"
    )
    if int(budget["completed_generations"]) != V10_POLICY.generations:
        mismatches["completed_generations"] = {
            "observed": budget["completed_generations"],
            "required": V10_POLICY.generations,
        }
    initialization = [
        item
        for item in history
        if item.get("event") == "evaluated" and item.get("generation") == 0
    ]
    if len(initialization) != V10_POLICY.population:
        mismatches["initial_population_size"] = {
            "observed": len(initialization),
            "required": V10_POLICY.population,
        }
    if result.get("selected_enhanced_physics", {}).get("joint_pass") is not True:
        mismatches["selected_physics_joint_pass"] = {
            "observed": result.get("selected_enhanced_physics", {}).get(
                "joint_pass"
            ),
            "required": True,
        }
    if mismatches:
        raise RuntimeError(
            "formal engine result failed invariants: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def main() -> int:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite formal run: {args.output}")
    validate_source_manifest()
    llm = load_llm_config()
    observed = {
        "model": llm.model,
        "temperature": llm.temperature,
        "top_p": llm.top_p,
        "max_tokens": llm.max_tokens,
        "seed": llm.seed,
    }
    if observed != V10_POLICY.llm_sampling:
        raise RuntimeError(
            f"formal LLM configuration is not frozen: {observed} != "
            f"{V10_POLICY.llm_sampling}"
        )
    train_path = (
        args.data_dir / f"Intersection_{args.intersection}_Train.jsonl"
    ).resolve()
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    started_utc = utc_now()

    previous_population = engine.PILOT_POPULATION
    previous_generations = engine.PILOT_GENERATIONS
    previous_argv = sys.argv[:]
    try:
        engine.PILOT_POPULATION = V10_POLICY.population
        engine.PILOT_GENERATIONS = V10_POLICY.generations
        sys.argv = [
            str(engine.__file__),
            "--intersection",
            str(args.intersection),
            "--data-dir",
            str(args.data_dir),
            "--output",
            str(args.output),
        ]
        return_code = engine.main()
    finally:
        engine.PILOT_POPULATION = previous_population
        engine.PILOT_GENERATIONS = previous_generations
        sys.argv = previous_argv
    if return_code != 0:
        return int(return_code)

    result_path = args.output / "result.json"
    history_path = args.output / "history.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    _verify_engine_result(result, history)
    budget = next(
        item
        for item in reversed(history)
        if item.get("event") == "search_budget_summary"
    )
    metrics = dict(result["selected_training_metrics"])
    result.update(
        {
            "status": "completed_v10_prompt_v6_efficiency_v1_formal_p10g10",
            "not_a_formal_result": False,
            "formal_protocol_complete": True,
            "method_id": V10_POLICY.method_id,
            "method_status": V10_POLICY.method_status,
            "policy": V10_POLICY.to_dict(),
            "formal_started_utc": started_utc,
            "formal_completed_utc": utc_now(),
            "train_file_name": train_path.name,
            "train_file_sha256": sha256_file(train_path),
            "selection_source": "full_training_evolution_only",
            "project_artifact_read_policy": "allowlist_only",
            "post_evolution_cv_reranking": False,
            "post_evolution_refit": False,
            "winner_reuses_in_evolution_parameters": True,
            "external_incumbent_allowed": False,
            "initial_population_all_generated_in_current_run": True,
            "initial_population_size_verified": V10_POLICY.population,
            "completed_generations": int(budget["completed_generations"]),
            "expression": result["selected_expression"],
            "lane_parameters": result["selected_parameters"],
            "training_metrics": metrics,
            "formal_comparison_training_metrics": {
                "raw_approach_macro_r2": metrics["macro_raw_r2"],
                "pooled_rmse": metrics["pooled_rmse"],
                "pooled_mae": metrics["pooled_mae"],
            },
            "training_fitness": result["selected_training_fitness"],
            "selected_enhanced_physics_from_evolution": result[
                "selected_enhanced_physics"
            ],
            "parameter_quality_audit": audit_parameter_quality(
                result["selected_expression"],
                result["selected_parameters"],
                V10_POLICY.coefficient_bounds,
            ),
            "compute_efficiency": summarize_compute_efficiency(history),
            "llm_runtime": {
                "provider_endpoint": llm.endpoint,
                "request_path": llm.request_path,
                "sampling": V10_POLICY.llm_sampling,
                "sampling_parameter_sources": llm.sampling_parameter_sources,
                "api_key_recorded": False,
            },
            "source_sha256": {
                str(path.relative_to(GMINI)): sha256_file(path)
                for path in SOURCE_FILES
            },
            "legacy_validation_labels_removed_from_formal_history": True,
        }
    )
    write_json(result_path, result)
    print(
        f"I{args.intersection} V10 formal P10/G10: "
        f"fitness={result['training_fitness']:.6f}, "
        f"raw_macro_R2={metrics['macro_raw_r2']:.6f}, "
        f"pooled_RMSE={metrics['pooled_rmse']:.6f}, "
        f"pooled_MAE={metrics['pooled_mae']:.6f}, "
        f"physics={result['selected_enhanced_physics']['joint_pass']}, "
        f"wall={result['wall_seconds']:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
