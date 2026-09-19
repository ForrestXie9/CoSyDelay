"""Exercise the real V10 candidate pipeline at P2/G1 without an API call."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from unittest.mock import patch

import expression_adaptation_lane
import optimization_lane
from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
from population_evolution_lane import evolve_universal_lane_expression
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import (
    DEFAULT_DATA_DIR,
    SplitAccessGuard,
    jsonable,
    lanes_for,
)

from .integration import install_v10_candidate
from .policy import V10_POLICY


EXPRESSION_BANK = (
    "Cycle_Time*flow_lane/GR_phase*a1",
    "Cycle_Time*flow_lane/GR_phase*(a1+a2*flow_lane)",
    (
        "Cycle_Time*flow_lane/GR_phase*"
        "(a1/(1+a2*GR_phase)+a3*log(1+a4*flow_lane))"
    ),
    (
        "Cycle_Time*flow_lane/GR_phase*"
        "(a1*(1+a2*flow_lane**a3)/(1+a4*GR_phase)+"
        "a5*exp(-a6*GR_phase))"
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    train_path = (DEFAULT_DATA_DIR / "Intersection_1_Train.jsonl").resolve()
    guard = SplitAccessGuard(train_path)
    guard.install()
    train = preprocess_data_flexible(
        load_dataset_flexible(str(train_path), 1), 1
    ).reset_index(drop=True)
    config = INTERSECTION_CONFIGS[1]
    targets = {
        approach: train[f"Delay_{approach}"].reset_index(drop=True)
        for approach in config["approaches"]
    }
    lanes, mapping = lanes_for(config)
    policy = replace(
        V10_POLICY,
        method_status="engineering_offline_integration_smoke",
        population=2,
        generations=1,
    )
    counter = {"value": 0}
    prompts = []

    def offline_llm(prompt, *args, **kwargs):
        prompts.append(str(prompt))
        expression = EXPRESSION_BANK[counter["value"] % len(EXPRESSION_BANK)]
        counter["value"] += 1
        return (
            "### Expression\n"
            f"y = {expression}\n"
            "### Explanation\n"
            "Offline deterministic V10 integration candidate."
        )

    bounds = policy.coefficient_bounds
    previous_bounds = (
        optimization_lane.DEFAULT_PARAM_BOUNDS,
        optimization_lane.POWER_EXPONENT_BOUNDS,
        optimization_lane.EXP_COEFFICIENT_BOUNDS,
    )
    optimization_lane.DEFAULT_PARAM_BOUNDS = bounds["scale"]
    optimization_lane.POWER_EXPONENT_BOUNDS = bounds["power_exponent"]
    optimization_lane.EXP_COEFFICIENT_BOUNDS = bounds["exp_coefficient"]
    started = time.perf_counter()
    try:
        with patch.object(expression_adaptation_lane, "run_llm", offline_llm):
            with install_v10_candidate(
                df_train=train,
                targets=targets,
                lanes=lanes,
                lane_to_approach=mapping,
                intersection_id=1,
                policy=policy,
            ) as runtime:
                expression, thought, explanation, parameters, history = (
                    evolve_universal_lane_expression(
                        df_train=train,
                        lanes=lanes,
                        lane_to_approach=mapping,
                        approach_targets=targets,
                        universal_features=[
                            "flow_lane",
                            "GR_phase",
                            "Cycle_Time",
                        ],
                        generations=1,
                        pop_size=2,
                        intersection_id=1,
                        score_mode="binary",
                        physics_weight=1.0,
                        prompt_knowledge=True,
                        prompt_style=policy.prompt_style,
                        seed=20270832,
                        optimizer_restarts=policy.optimizer_restarts,
                        residual_guidance_mode="none",
                        structural_diversity_mode=policy.structural_diversity_mode,
                        use_feasible_archive=False,
                        targeted_physical_feedback=True,
                    )
                )
                selected = runtime.evaluations[expression]
                evaluated = [
                    item for item in history if item.get("event") == "evaluated"
                ]
                parameterizations = {
                    item["evaluation_details"]["fit"]["parameterization"]
                    for item in evaluated
                }
                adapter_ids = {
                    item["evaluation_details"]["fit"]["formal_method_adapter"]
                    for item in evaluated
                }
                if parameterizations != {"all_positive_log"}:
                    raise RuntimeError(parameterizations)
                if adapter_ids != {
                    "cosydelay_v10_accuracy_first_parsimony_alllog.fitter"
                }:
                    raise RuntimeError(adapter_ids)
                initialization_prompts = [
                    prompt for prompt in prompts if "INITIALIZATION TASK:" in prompt
                ]
                mutation_prompts = [
                    prompt for prompt in prompts if "MUTATION TASK:" in prompt
                ]
                prompt_audit = {
                    "prompt_calls": len(prompts),
                    "initialization_prompt_calls": len(initialization_prompts),
                    "mutation_prompt_calls": len(mutation_prompts),
                    "maximum_prompt_characters": max(map(len, prompts)),
                    "coefficient_count_conflict_absent": all(
                        "3--6" not in prompt and "3-6" not in prompt
                        for prompt in prompts
                    ),
                    "single_physical_contract_per_prompt": all(
                        prompt.count("PHYSICAL AND STRUCTURAL REQUIREMENTS:") == 1
                        for prompt in prompts
                    ),
                    "single_output_schema_per_prompt": all(
                        prompt.count("### Expression") == 1 for prompt in prompts
                    ),
                    "mutation_has_parent": all(
                        prompt.count("PARENT EXPRESSION:") == 1
                        for prompt in mutation_prompts
                    ),
                    "mutation_has_training_feedback": all(
                        prompt.count(
                            "PARENT TRAINING EVALUATION (Training only):"
                        )
                        == 1
                        for prompt in mutation_prompts
                    ),
                    "validation_or_test_values_exposed": any(
                        "validation r2:" in prompt.lower()
                        or "test r2:" in prompt.lower()
                        for prompt in prompts
                    ),
                }
                if not all(
                    (
                        prompt_audit["coefficient_count_conflict_absent"],
                        prompt_audit["single_physical_contract_per_prompt"],
                        prompt_audit["single_output_schema_per_prompt"],
                        prompt_audit["mutation_has_parent"],
                        prompt_audit["mutation_has_training_feedback"],
                    )
                ) or prompt_audit["validation_or_test_values_exposed"]:
                    raise RuntimeError(prompt_audit)
                result = {
                    "status": "passed",
                    "role": "offline V10 engineering integration smoke only",
                    "not_formal_accuracy_evidence": True,
                    "intersection_id": 1,
                    "population": 2,
                    "generations": 1,
                    "optimizer_restarts": policy.optimizer_restarts,
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
                    "optimizer_parameterizations": sorted(parameterizations),
                    "formal_method_adapters": sorted(adapter_ids),
                    "structural_diversity_mode": policy.structural_diversity_mode,
                    "accessed_splits": sorted(
                        {item["split"] for item in guard.records}
                    ),
                    "test_file_opened": False,
                    "generation_audit": runtime.generation_audit,
                    "complexity_audit": runtime.complexity_audit,
                    "prefit_attempts": len(runtime.prefit_audit),
                    "fitted_rejections": len(runtime.fitted_rejections),
                    "expression": expression,
                    "parameters": parameters,
                    "metrics": selected["metrics"],
                    "enhanced_physics": selected["enhanced_physics"],
                    "history": history,
                    "wall_seconds": time.perf_counter() - started,
                    "offline_llm_responses": counter["value"],
                    "prompt_audit": prompt_audit,
                }
    finally:
        (
            optimization_lane.DEFAULT_PARAM_BOUNDS,
            optimization_lane.POWER_EXPONENT_BOUNDS,
            optimization_lane.EXP_COEFFICIENT_BOUNDS,
        ) = previous_bounds
        guard.disable()

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "result.json").write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"offline V10 smoke passed: expression={expression}, "
        f"R2={selected['metrics']['macro_raw_r2']:.6f}, "
        f"physics={selected['enhanced_physics']['joint_pass']}, "
        f"wall={result['wall_seconds']:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
