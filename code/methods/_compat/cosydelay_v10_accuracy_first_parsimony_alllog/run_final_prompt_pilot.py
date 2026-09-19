"""Real-API P3/G2 Training-only pilot for the cleaned final V10 prompt."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import os
from pathlib import Path
import time

import optimization_lane
from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
from expression_rules import coefficient_names
from llm_config import load_llm_config
from population_evolution_lane import evolve_universal_lane_expression
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import (
    DEFAULT_DATA_DIR,
    SplitAccessGuard,
    lanes_for,
    read_jsonl,
    sanitize_formal_history,
    summarize_expression_attempts,
    summarize_llm_attempts,
    utc_now,
    write_json,
)
from methods.cosydelay_v9_clean_from_scratch.source_manifest import (
    validate_source_manifest,
)
from methods.prospective_optimizer_conditioning_v1.complexity_prompt import (
    select_epsilon_parsimonious,
    summarize_complexity_audit,
)

from .integration import install_v10_candidate
from .policy import V10_POLICY


PILOT_POPULATION = 3
PILOT_GENERATIONS = 2


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _prompt_audit(llm_attempts: list[dict]) -> dict:
    prompts = [
        str(item["prompt"])
        for item in llm_attempts
        if item.get("prompt") and item.get("status") == "success"
    ]
    mutation = [item for item in prompts if "MUTATION TASK:" in item]
    audit = {
        "successful_prompt_calls": len(prompts),
        "mutation_prompt_calls": len(mutation),
        "maximum_prompt_characters": max(map(len, prompts), default=0),
        "coefficient_count_conflict_absent": all(
            "3--6" not in item and "3-6" not in item for item in prompts
        ),
        "single_physical_contract_per_prompt": all(
            item.count("PHYSICAL AND STRUCTURAL REQUIREMENTS:") == 1
            for item in prompts
        ),
        "single_output_schema_per_prompt": all(
            item.count("### Expression") == 1 for item in prompts
        ),
        "mutation_has_parent": all(
            item.count("PARENT EXPRESSION:") == 1 for item in mutation
        ),
        "mutation_has_training_feedback": all(
            item.count("PARENT TRAINING EVALUATION (Training only):") == 1
            for item in mutation
        ),
        "validation_or_test_values_exposed": any(
            "validation r2:" in item.lower() or "test r2:" in item.lower()
            for item in prompts
        ),
    }
    required = (
        "coefficient_count_conflict_absent",
        "single_physical_contract_per_prompt",
        "single_output_schema_per_prompt",
        "mutation_has_parent",
        "mutation_has_training_feedback",
    )
    if not prompts or not mutation or not all(audit[name] for name in required):
        raise RuntimeError(f"final prompt audit failed: {audit}")
    if audit["validation_or_test_values_exposed"]:
        raise RuntimeError(f"split leakage in final prompt: {audit}")
    return audit


def main() -> int:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite pilot: {args.output}")

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
            f"pilot LLM configuration is not frozen: {observed} != "
            f"{V10_POLICY.llm_sampling}"
        )
    validate_source_manifest()

    train_path = (
        args.data_dir / f"Intersection_{args.intersection}_Train.jsonl"
    ).resolve()
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    audit_path = args.output / "llm_audit.jsonl"
    expression_audit_path = args.output / "expression_attempt_audit.jsonl"
    guard = SplitAccessGuard(
        train_path,
        guarded_roots=(Path(__file__).resolve().parents[4],),
        allowed_artifact_paths=(audit_path, expression_audit_path),
    )
    guard.install()
    args.output.mkdir(parents=True, exist_ok=False)

    config = INTERSECTION_CONFIGS[args.intersection]
    train = preprocess_data_flexible(
        load_dataset_flexible(str(train_path), args.intersection),
        args.intersection,
    ).reset_index(drop=True)
    targets = {
        approach: train[f"Delay_{approach}"].reset_index(drop=True)
        for approach in config["approaches"]
    }
    lanes, lane_to_approach = lanes_for(config)
    evolution_seed = V10_POLICY.evolution_base_seed + args.intersection * 10_000
    policy = replace(
        V10_POLICY,
        method_status="real_api_final_prompt_p3g2_pilot",
        population=PILOT_POPULATION,
        generations=PILOT_GENERATIONS,
    )

    environment_names = (
        "LLM_AUDIT_LOG",
        "LLM_AUDIT_INCLUDE_CONTENT",
        "EXPRESSION_ATTEMPT_AUDIT_LOG",
    )
    previous_environment = {
        name: os.environ.get(name) for name in environment_names
    }
    previous_bounds = (
        optimization_lane.DEFAULT_PARAM_BOUNDS,
        optimization_lane.POWER_EXPONENT_BOUNDS,
        optimization_lane.EXP_COEFFICIENT_BOUNDS,
    )
    bounds = policy.coefficient_bounds
    optimization_lane.DEFAULT_PARAM_BOUNDS = bounds["scale"]
    optimization_lane.POWER_EXPONENT_BOUNDS = bounds["power_exponent"]
    optimization_lane.EXP_COEFFICIENT_BOUNDS = bounds["exp_coefficient"]
    # absolute() keeps subst/junction short roots; resolve() expands to MAX_PATH.
    os.environ["LLM_AUDIT_LOG"] = str(audit_path.absolute())
    os.environ["LLM_AUDIT_INCLUDE_CONTENT"] = "true"
    os.environ["EXPRESSION_ATTEMPT_AUDIT_LOG"] = str(
        expression_audit_path.absolute()
    )

    started_utc = utc_now()
    started = time.perf_counter()
    try:
        with install_v10_candidate(
            df_train=train,
            targets=targets,
            lanes=lanes,
            lane_to_approach=lane_to_approach,
            intersection_id=args.intersection,
            policy=policy,
        ) as runtime:
            expression, thought, explanation, parameters, history = (
                evolve_universal_lane_expression(
                    df_train=train,
                    lanes=lanes,
                    lane_to_approach=lane_to_approach,
                    approach_targets=targets,
                    universal_features=["flow_lane", "GR_phase", "Cycle_Time"],
                    generations=PILOT_GENERATIONS,
                    pop_size=PILOT_POPULATION,
                    intersection_id=args.intersection,
                    score_mode="binary",
                    physics_weight=1.0,
                    prompt_knowledge=True,
                    prompt_style=policy.prompt_style,
                    seed=evolution_seed,
                    optimizer_restarts=policy.optimizer_restarts,
                    max_wall_seconds=None,
                    residual_guidance_mode="none",
                    structural_diversity_mode=policy.structural_diversity_mode,
                    use_feasible_archive=False,
                    targeted_physical_feedback=True,
                )
            )
            selected = runtime.evaluations.get(str(expression))
            if selected is None:
                raise RuntimeError("pilot winner is not a passing evaluation")
            epsilon = select_epsilon_parsimonious(runtime.evaluations)
            prefit_audit = list(runtime.prefit_audit)
            fitted_rejections = list(runtime.fitted_rejections)
            generation_audit = list(runtime.generation_audit)
            complexity_audit = list(runtime.complexity_audit)
            passing_evaluations = len(runtime.evaluations)
    finally:
        (
            optimization_lane.DEFAULT_PARAM_BOUNDS,
            optimization_lane.POWER_EXPONENT_BOUNDS,
            optimization_lane.EXP_COEFFICIENT_BOUNDS,
        ) = previous_bounds
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        guard.disable()

    budget = next(
        item
        for item in reversed(history)
        if item.get("event") == "search_budget_summary"
    )
    initialization = [
        item
        for item in history
        if item.get("event") == "evaluated" and item.get("generation") == 0
    ]
    if len(initialization) != PILOT_POPULATION:
        raise RuntimeError("final prompt pilot did not fill the P3 population")
    if int(budget["completed_generations"]) != PILOT_GENERATIONS:
        raise RuntimeError("final prompt pilot did not complete both generations")

    expression_attempts = read_jsonl(expression_audit_path)
    llm_attempts = read_jsonl(audit_path)
    expression_summary = summarize_expression_attempts(expression_attempts)
    llm_summary = summarize_llm_attempts(llm_attempts)
    prompt_audit = _prompt_audit(llm_attempts)
    accepted_counts = Counter(
        len(coefficient_names(str(item["expression"])))
        for item in expression_attempts
        if item.get("status") == "accepted" and item.get("expression")
    )
    postfit_denominator = passing_evaluations + len(fitted_rejections)
    passing_details = [
        item["evaluation_details"]
        for item in history
        if item.get("event") == "evaluated"
    ]
    rejected_details = [
        item.get("details", {}) for item in fitted_rejections
    ]
    fitted_details = passing_details + rejected_details
    efficiency_timing = {
        "enhanced_physics_wall_seconds_sum": float(
            sum(
                float(item.get("enhanced_physics_wall_seconds", 0.0))
                for item in fitted_details
            )
        ),
        "standard_fitted_physics_reuse_count": sum(
            bool(
                item.get("efficiency_reuse", {}).get(
                    "standard_fitted_physics", False
                )
            )
            for item in fitted_details
        ),
        "prefit_symbolic_endpoint_reuse_count": sum(
            bool(
                item.get("efficiency_reuse", {}).get(
                    "prefit_symbolic_endpoint", False
                )
            )
            for item in fitted_details
        ),
        "safe_fast_r9_certificate_count": sum(
            item.get("coefficient_robust_endpoint_audit", {}).get(
                "symbolic_r9_method"
            )
            == "fixed_inverse_green_finite_positive_h_certificate"
            for item in prefit_audit
        ),
        "fitted_family_prefit_rejection_count": sum(
            item.get("gate_id")
            == "fitted_rejected_family_cache_2026_08_11_v1"
            for item in prefit_audit
        ),
    }
    result = {
        "schema_version": 1,
        "status": "completed_v10_final_prompt_p3g2_pilot",
        "not_a_formal_result": True,
        "selection_and_evaluation_scope": "Training only",
        "intersection_id": args.intersection,
        "population": PILOT_POPULATION,
        "generations": PILOT_GENERATIONS,
        "optimizer": policy.optimizer,
        "optimizer_restarts": policy.optimizer_restarts,
        "prompt_contract_version": policy.prompt_contract_version,
        "execution_contract_version": policy.execution_contract_version,
        "efficiency_contract": {
            "reuse_standard_fitted_physics_audit": bool(
                policy.reuse_standard_fitted_physics_audit
            ),
            "reuse_prefit_symbolic_endpoint_audit": bool(
                policy.reuse_prefit_symbolic_endpoint_audit
            ),
            "reject_fitted_structural_families": bool(
                policy.reject_fitted_structural_families
            ),
            "rejected_structural_families": len(
                runtime.clean.rejected_family_feedback
            ),
        },
        "efficiency_timing": efficiency_timing,
        "evolution_selection": "unchanged paper Training Fitness",
        "structural_diversity_mode": policy.structural_diversity_mode,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "accessed_splits": sorted({item["split"] for item in guard.records}),
        "outer_validation_accessed": False,
        "test_file_opened": False,
        "llm_sampling": policy.llm_sampling,
        "llm_attempt_summary": llm_summary,
        "expression_attempt_summary": expression_summary,
        "prompt_audit": prompt_audit,
        "accepted_coefficient_count_distribution": {
            str(key): int(value) for key, value in sorted(accepted_counts.items())
        },
        "complexity_audit_summary": summarize_complexity_audit(complexity_audit),
        "epsilon_parsimony_diagnostic": epsilon,
        "prefit_gate_attempts": len(prefit_audit),
        "prefit_gate_passes": sum(bool(item.get("passed")) for item in prefit_audit),
        "fitted_physics_rejections": len(fitted_rejections),
        "physically_passing_evaluations": passing_evaluations,
        "postfit_pass_rate": (
            passing_evaluations / postfit_denominator
            if postfit_denominator
            else None
        ),
        "candidate_evaluations": int(budget["candidate_evaluations"]),
        "selected_expression": expression,
        "selected_coefficient_count": len(coefficient_names(expression)),
        "selected_parameters": parameters,
        "selected_training_fitness": selected["fitness"],
        "selected_training_metrics": selected["metrics"],
        "selected_enhanced_physics": selected["enhanced_physics"],
        "thought": thought,
        "explanation": explanation,
    }
    write_json(args.output / "result.json", result)
    write_json(args.output / "history.json", sanitize_formal_history(history))
    write_json(args.output / "generation_audit.json", generation_audit)
    write_json(args.output / "prefit_gate_audit.json", prefit_audit)
    write_json(args.output / "complexity_prompt_audit.json", complexity_audit)
    write_json(args.output / "fitted_rejections.json", fitted_rejections)
    print(
        f"I{args.intersection} final-prompt P{PILOT_POPULATION}/"
        f"G{PILOT_GENERATIONS}: attempts={expression_summary['records']}, "
        f"counts={dict(sorted(accepted_counts.items()))}, "
        f"fitness={selected['fitness']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
