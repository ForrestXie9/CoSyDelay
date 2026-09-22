"""Real-API P3/G2 Training-only accuracy-first parsimony pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]
if str(GMINI) not in sys.path:
    sys.path.insert(0, str(GMINI))

import optimization_lane  # noqa: E402
from constants import INTERSECTION_CONFIGS  # noqa: E402
from data_processing import load_dataset_flexible, preprocess_data_flexible  # noqa: E402
from expression_rules import coefficient_names  # noqa: E402
from llm_config import load_llm_config  # noqa: E402
from population_evolution_lane import evolve_universal_lane_expression  # noqa: E402
from methods.cosydelay.engine.data_protocol.integration import (  # noqa: E402
    install_clean_single_evolution,
)
from methods.cosydelay.engine.data_protocol.policy import CLEAN_POLICY  # noqa: E402
from methods.cosydelay.engine.data_protocol.run_formal_training_search import (  # noqa: E402
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
from methods.cosydelay.engine.data_protocol.source_manifest import (  # noqa: E402
    validate_source_manifest,
)
from methods.cosydelay.engine.optimizer_conditioning.complexity_prompt import (  # noqa: E402
    COMPLEXITY_CONTRACT_ID,
    install_accuracy_first_parsimony_contract,
    select_epsilon_parsimonious,
    summarize_complexity_audit,
)


PILOT_POPULATION = 3
PILOT_GENERATIONS = 2


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", type=int, choices=(1, 2), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite pilot: {args.output}")

    # This preflight intentionally occurs before output creation or data access.
    llm = load_llm_config()
    observed = {
        "model": llm.model,
        "temperature": llm.temperature,
        "top_p": llm.top_p,
        "max_tokens": llm.max_tokens,
        "seed": llm.seed,
    }
    if observed != CLEAN_POLICY.llm_sampling:
        raise RuntimeError(
            f"pilot LLM configuration is not frozen: {observed} != "
            f"{CLEAN_POLICY.llm_sampling}"
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
        guarded_roots=(GMINI.parent.parent,),
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
    evolution_seed = (
        CLEAN_POLICY.evolution_base_seed + args.intersection * 10_000
    )

    previous_environment = {
        name: os.environ.get(name)
        for name in (
            "LLM_AUDIT_LOG",
            "LLM_AUDIT_INCLUDE_CONTENT",
            "EXPRESSION_ATTEMPT_AUDIT_LOG",
        )
    }
    previous_bounds = {
        "scale": optimization_lane.DEFAULT_PARAM_BOUNDS,
        "power_exponent": optimization_lane.POWER_EXPONENT_BOUNDS,
        "exp_coefficient": optimization_lane.EXP_COEFFICIENT_BOUNDS,
    }
    bounds = CLEAN_POLICY.coefficient_bounds
    optimization_lane.DEFAULT_PARAM_BOUNDS = bounds["scale"]
    optimization_lane.POWER_EXPONENT_BOUNDS = bounds["power_exponent"]
    optimization_lane.EXP_COEFFICIENT_BOUNDS = bounds["exp_coefficient"]
    os.environ["LLM_AUDIT_LOG"] = str(audit_path.resolve())
    os.environ["LLM_AUDIT_INCLUDE_CONTENT"] = "true"
    os.environ["EXPRESSION_ATTEMPT_AUDIT_LOG"] = str(
        expression_audit_path.resolve()
    )

    started_utc = utc_now()
    started = time.perf_counter()
    try:
        with install_clean_single_evolution(
            df_train=train,
            targets=targets,
            lanes=lanes,
            lane_to_approach=lane_to_approach,
            intersection_id=args.intersection,
            policy=CLEAN_POLICY,
        ) as runtime:
            with install_accuracy_first_parsimony_contract() as complexity_audit:
                expression, thought, explanation, parameters, history = (
                    evolve_universal_lane_expression(
                        df_train=train,
                        lanes=lanes,
                        lane_to_approach=lane_to_approach,
                        approach_targets=targets,
                        universal_features=[
                            "flow_lane",
                            "GR_phase",
                            "Cycle_Time",
                        ],
                        generations=PILOT_GENERATIONS,
                        pop_size=PILOT_POPULATION,
                        intersection_id=args.intersection,
                        score_mode="binary",
                        physics_weight=1.0,
                        prompt_knowledge=True,
                        prompt_style=CLEAN_POLICY.prompt_style,
                        seed=evolution_seed,
                        optimizer_restarts=CLEAN_POLICY.optimizer_restarts,
                        max_wall_seconds=None,
                        residual_guidance_mode="none",
                        structural_diversity_mode="family_unique",
                        use_feasible_archive=False,
                        targeted_physical_feedback=True,
                    )
                )
            selected = runtime.evaluations.get(str(expression))
            if selected is None:
                raise RuntimeError(
                    "pilot winner is not a physically passing evaluation"
                )
            epsilon_parsimony = select_epsilon_parsimonious(runtime.evaluations)
            prefit_audit = list(runtime.prefit_audit)
            fitted_rejections = list(runtime.fitted_rejections)
            generation_audit = list(runtime.generation_audit)
            passing_evaluations = len(runtime.evaluations)
    finally:
        optimization_lane.DEFAULT_PARAM_BOUNDS = previous_bounds["scale"]
        optimization_lane.POWER_EXPONENT_BOUNDS = previous_bounds[
            "power_exponent"
        ]
        optimization_lane.EXP_COEFFICIENT_BOUNDS = previous_bounds[
            "exp_coefficient"
        ]
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
        raise RuntimeError("complexity pilot did not fill the P3 population")
    if int(budget["completed_generations"]) != PILOT_GENERATIONS:
        raise RuntimeError("complexity pilot did not complete both generations")

    expression_attempts = read_jsonl(expression_audit_path)
    llm_attempts = read_jsonl(audit_path)
    expression_summary = summarize_expression_attempts(expression_attempts)
    llm_summary = summarize_llm_attempts(llm_attempts)
    accepted_counts = Counter(
        len(coefficient_names(str(item["expression"])))
        for item in expression_attempts
        if item.get("status") == "accepted" and item.get("expression")
    )
    postfit_denominator = passing_evaluations + len(fitted_rejections)
    result = {
        "schema_version": 1,
        "status": "completed_accuracy_first_parsimony_prompt_pilot",
        "not_a_formal_result": True,
        "selection_and_evaluation_scope": "Training only",
        "intersection_id": args.intersection,
        "population": PILOT_POPULATION,
        "generations": PILOT_GENERATIONS,
        "optimizer": CLEAN_POLICY.optimizer,
        "optimizer_restarts": CLEAN_POLICY.optimizer_restarts,
        "base_prompt_contract_version": CLEAN_POLICY.prompt_contract_version,
        "complexity_contract_id": COMPLEXITY_CONTRACT_ID,
        "evolution_selection": "unchanged paper Training Fitness",
        "structural_diversity_mode": "family_unique",
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "accessed_splits": sorted({item["split"] for item in guard.records}),
        "outer_validation_accessed": False,
        "test_file_opened": False,
        "llm_sampling": CLEAN_POLICY.llm_sampling,
        "llm_attempt_summary": llm_summary,
        "expression_attempt_summary": expression_summary,
        "accepted_coefficient_count_distribution": {
            str(key): int(value) for key, value in sorted(accepted_counts.items())
        },
        "complexity_audit_summary": summarize_complexity_audit(complexity_audit),
        "epsilon_parsimony_diagnostic": epsilon_parsimony,
        "prefit_gate_attempts": len(prefit_audit),
        "prefit_gate_passes": sum(
            bool(item.get("passed")) for item in prefit_audit
        ),
        "prefit_gate_pass_rate": (
            sum(bool(item.get("passed")) for item in prefit_audit)
            / len(prefit_audit)
            if prefit_audit
            else None
        ),
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
        f"I{args.intersection} complexity pilot P{PILOT_POPULATION}/"
        f"G{PILOT_GENERATIONS}: attempts={expression_summary['records']}, "
        f"counts={dict(sorted(accepted_counts.items()))}, "
        f"prefit_pass={result['prefit_gate_pass_rate']:.3f}, "
        f"postfit_pass={result['postfit_pass_rate']:.3f}, "
        f"fitness={selected['fitness']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
