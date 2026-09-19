import argparse
from pathlib import Path
import json
import sys

import numpy as np
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parent
GMINI = ROOT.parent
sys.path.insert(0, str(GMINI))

from data_processing import load_dataset_flexible, preprocess_data_flexible
from expression_rules import validate_candidate_expression
from expression_validation_lane import validate_fitted_lanes_batch
from main import get_lanes_and_mapping
from optimization_lane import (
    calculate_approach_delays_from_universal,
    fit_lane_parameters_to_approaches,
    prepare_optimization_context,
)


CANDIDATES = (
    "Cycle_Time*a1*(flow_lane/GR_phase)**a2",
    "Cycle_Time*(a1*(flow_lane/GR_phase)**a2+a3*flow_lane*log(1+a4/GR_phase))",
    "Cycle_Time*(a1*flow_lane/GR_phase+a2*(flow_lane/GR_phase)**a3+a4*flow_lane*log(1+a5/GR_phase))",
)
FALLBACK_OUTPUT = ROOT / "models" / "symbolic_lane_model_fallback.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit a deterministic development fallback; never a formal frozen model."
    )
    parser.add_argument("--output", type=Path, default=FALLBACK_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    formal_path = ROOT / "models" / "symbolic_lane_model.json"
    if output.resolve() == formal_path.resolve():
        raise ValueError(
            "Fallback preparation is forbidden from overwriting the formal frozen "
            f"model path: {formal_path}"
        )
    data_path = GMINI.parent.parent / "Final_cosy_delay" / "jsonl_files" / "Intersection_1_Train.jsonl"
    df = preprocess_data_flexible(load_dataset_flexible(str(data_path), 1), 1)
    lanes, mapping = get_lanes_and_mapping(1)
    targets = {approach: df[f"Delay_{approach}"] for approach in "SENW"}
    context = prepare_optimization_context(df, lanes, mapping, 1)

    best = None
    for candidate_index, expression in enumerate(CANDIDATES):
        structurally_valid, reason = validate_candidate_expression(expression)
        if not structurally_valid:
            print(f"skip: {reason}: {expression}")
            continue
        np.random.seed(20260712 + candidate_index)
        parameters = fit_lane_parameters_to_approaches(
            expression,
            df,
            lanes,
            mapping,
            targets,
            1,
            prepared_context=context,
        )
        physical_errors = validate_fitted_lanes_batch(
            expression, parameters, lanes, ["flow_lane", "GR_phase", "Cycle_Time"]
        )
        if physical_errors:
            print(f"skip physical errors ({len(physical_errors)} lanes): {expression}")
            continue
        predictions = calculate_approach_delays_from_universal(
            df,
            expression,
            parameters,
            lanes,
            mapping,
            1,
            prepared_context=context,
            strict=True,
        )
        score = float(np.mean([
            max(0.0, r2_score(targets[approach].values, predictions[approach]))
            for approach in targets
        ]))
        print(f"candidate R2={score:.4f}: {expression}")
        if best is None or score > best["train_r2"]:
            best = {
                "expression": expression,
                "lane_parameters": parameters,
                "train_r2": score,
                "source": "deterministic physically-valid fallback candidate selection",
            }

    if best is None:
        raise RuntimeError("No physically valid fallback symbolic model could be fitted")
    best.update({
        "schema_version": 1,
        "status": "fallback_not_for_formal_control",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Fallback output already exists: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(best, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
