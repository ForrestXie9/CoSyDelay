"""Exercise the real clean pipeline at P2/G1 without an API call."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]
if str(GMINI) not in sys.path:
    sys.path.insert(0, str(GMINI))

import expression_adaptation_lane  # noqa: E402
import optimization_lane  # noqa: E402
from constants import INTERSECTION_CONFIGS  # noqa: E402
from data_processing import load_dataset_flexible, preprocess_data_flexible  # noqa: E402
from population_evolution_lane import evolve_universal_lane_expression  # noqa: E402
from methods.cosydelay.engine.data_protocol.integration import (  # noqa: E402
    install_clean_single_evolution,
)
from methods.cosydelay.engine.data_protocol.policy import CLEAN_POLICY  # noqa: E402
from methods.cosydelay.engine.data_protocol.run_formal_training_search import (  # noqa: E402
    DEFAULT_DATA_DIR,
    SplitAccessGuard,
    jsonable,
    lanes_for,
)


EXPRESSION_BANK = (
    "Cycle_Time*a1*flow_lane/GR_phase",
    "Cycle_Time*a1*flow_lane*exp(a2/GR_phase)",
    "Cycle_Time*a1*flow_lane/(GR_phase*(1+a2*GR_phase))",
    "Cycle_Time*a1*flow_lane*(1+a2*flow_lane)/(GR_phase*(1+a3*flow_lane))",
    "Cycle_Time*a1*flow_lane*(1+a2*log(1+a3*flow_lane))/GR_phase",
    "Cycle_Time*flow_lane*(a1/GR_phase+a2*exp(a3/GR_phase))",
    "Cycle_Time*a1*flow_lane*(1+a2*(flow_lane**a3))/GR_phase",
    "Cycle_Time*a1*flow_lane*exp(a2*(1-GR_phase)/GR_phase)",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            HERE
            / "experiments"
            / "offline_integration_smoke_i1_p2g1_clean_adapter_v3_20260809"
        ),
    )
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    train_path = (DEFAULT_DATA_DIR / "Intersection_1_Train.jsonl").resolve()
    guard = SplitAccessGuard(train_path)
    guard.install()
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
    policy = replace(
        CLEAN_POLICY,
        method_status="engineering_offline_integration_smoke",
        population=2,
        generations=1,
    )
    counter = {"value": 0}

    def offline_llm(prompt, *args, **kwargs):
        expression = EXPRESSION_BANK[counter["value"] % len(EXPRESSION_BANK)]
        counter["value"] += 1
        return (
            "### Expression\n"
            f"y = {expression}\n"
            "### Explanation\n"
            "Offline deterministic pipeline smoke candidate."
        )

    bounds = CLEAN_POLICY.coefficient_bounds
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
            with install_clean_single_evolution(
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
                        prompt_style="standard",
                        seed=20270831,
                        optimizer_restarts=10,
                        residual_guidance_mode="none",
                        structural_diversity_mode="canonical",
                        use_feasible_archive=False,
                        targeted_physical_feedback=True,
                    )
                )
                selected = runtime.evaluations[expression]
                result = {
                    "status": "passed",
                    "role": "offline engineering integration smoke only",
                    "not_formal_accuracy_evidence": True,
                    "intersection_id": 1,
                    "population": 2,
                    "generations": 1,
                    "optimizer_restarts": 10,
                    "accessed_splits": sorted(
                        {item["split"] for item in guard.records}
                    ),
                    "incumbent_injected": runtime.incumbent_injected,
                    "generation_audit": runtime.generation_audit,
                    "prefit_attempts": len(runtime.prefit_audit),
                    "fitted_rejections": len(runtime.fitted_rejections),
                    "expression": expression,
                    "parameters": parameters,
                    "metrics": selected["metrics"],
                    "enhanced_physics": selected["enhanced_physics"],
                    "history": history,
                    "wall_seconds": time.perf_counter() - started,
                    "offline_llm_responses": counter["value"],
                }
    finally:
        (
            optimization_lane.DEFAULT_PARAM_BOUNDS,
            optimization_lane.POWER_EXPONENT_BOUNDS,
            optimization_lane.EXP_COEFFICIENT_BOUNDS,
        ) = previous_bounds
        guard.disable()
    (output / "result.json").write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"offline clean smoke passed: expression={expression}, "
        f"R2={selected['metrics']['macro_raw_r2']:.6f}, "
        f"physics={selected['enhanced_physics']['joint_pass']}, "
        f"wall={result['wall_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
