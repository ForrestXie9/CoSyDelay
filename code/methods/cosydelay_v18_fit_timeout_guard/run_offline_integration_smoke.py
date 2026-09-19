"""Exercise the real V18 P2/G1 pipeline without an API call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
import expression_adaptation_lane
import optimization_lane
from population_evolution_lane import evolve_universal_lane_expression
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import (
    DEFAULT_DATA_DIR,
    jsonable,
    lanes_for,
)

from .integration import get_last_fit_guard_audit, install_v18_candidate
from .policy import V18_POLICY
from .run_p10g10_training import validate_fit_guard_audit


EXPRESSIONS = (
    (
        "Cycle_Time*(a1+a2*flow_lane**a3*(1+a4*exp(-a5*GR_phase))"
        "+a6/(GR_phase**a7+a8))"
    ),
    "Cycle_Time*a1*flow_lane/GR_phase",
    "Cycle_Time*a1*flow_lane*(1+a2*flow_lane)/(GR_phase*(1+a3*flow_lane))",
    "Cycle_Time*a1*flow_lane*(1+a2*log(1+a3*flow_lane))/GR_phase",
    "Cycle_Time*flow_lane*(a1/GR_phase+a2*exp(a3*(1-GR_phase)))",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    train_path = (args.data_dir / "Intersection_1_Train.jsonl").resolve()
    train = preprocess_data_flexible(
        load_dataset_flexible(str(train_path), 1), 1
    ).reset_index(drop=True)
    config = INTERSECTION_CONFIGS[1]
    approaches = list(config["approaches"])
    targets = {
        approach: train[f"Delay_{approach}"].reset_index(drop=True)
        for approach in approaches
    }
    lanes, mapping = lanes_for(config)
    counter = {"value": 0}

    def offline_llm(prompt, *unused_args, **unused_kwargs):
        del prompt, unused_args, unused_kwargs
        expression = EXPRESSIONS[counter["value"] % len(EXPRESSIONS)]
        counter["value"] += 1
        return (
            "### Expression\n"
            f"y = {expression}\n"
            "### Explanation\n"
            "Deterministic offline V18 integration candidate."
        )

    bounds = V18_POLICY.coefficient_bounds
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
            with install_v18_candidate(
                df_train=train,
                targets=targets,
                lanes=lanes,
                lane_to_approach=mapping,
                intersection_id=1,
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
                        score_mode="principlewise",
                        physics_weight=1.0,
                        prompt_knowledge=True,
                        prompt_style="standard",
                        seed=20270831,
                        optimizer_restarts=10,
                        residual_guidance_mode="none",
                        structural_diversity_mode="canonical",
                        use_feasible_archive=False,
                        targeted_physical_feedback=False,
                    )
                )
                selected = runtime.evaluations[expression]
    finally:
        (
            optimization_lane.DEFAULT_PARAM_BOUNDS,
            optimization_lane.POWER_EXPONENT_BOUNDS,
            optimization_lane.EXP_COEFFICIENT_BOUNDS,
        ) = previous_bounds
    fit_guard_audit = get_last_fit_guard_audit()
    fit_guard_summary = validate_fit_guard_audit(
        fit_guard_audit, expected_candidates=4
    )
    if fit_guard_summary["prefit_r7_proven_finite_rejections"] < 1:
        raise RuntimeError("offline smoke did not exercise the V18 R7 pre-fit gate")
    budget = next(
        item for item in reversed(history) if item.get("event") == "search_budget_summary"
    )
    result = {
        "status": "pass",
        "role": "offline full-pipeline smoke; not formal accuracy evidence",
        "intersection_id": 1,
        "population": 2,
        "generations": 1,
        "candidate_evaluations": budget["candidate_evaluations"],
        "accessed_splits": ["train"],
        "test_file_opened": False,
        "expression": expression,
        "parameters": parameters,
        "selected_training_metrics": selected["metrics"],
        "selected_fitness": selected["fitness"],
        "selected_physics": selected["enhanced_physics"],
        "fit_guard_summary": fit_guard_summary,
        "fit_guard_audit": fit_guard_audit,
        "offline_llm_responses": counter["value"],
        "wall_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "offline V18 P2/G1 full-pipeline smoke passed: "
        f"candidates={budget['candidate_evaluations']}, "
        f"fit_timeouts={fit_guard_summary['fit_timeouts']}, "
        f"wall={result['wall_seconds']:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
