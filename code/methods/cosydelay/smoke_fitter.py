"""Training-only runtime smoke test for the CoSyDelay coefficient fitter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import lanes_for
from methods.cosydelay_v16_manuscript_principlewise.evaluate_frozen_test import pooled_metrics
from optimization_lane import calculate_approach_delays_from_universal, prepare_optimization_context

from .fitter import StableRangeAcceleratedFitter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source = json.loads(args.source_result.resolve().read_text(encoding="utf-8"))
    if source.get("test_file_opened") is not False or source.get("outer_validation_accessed") is not False:
        raise RuntimeError("source expression must come from a Training-only result")
    intersection = int(source["intersection_id"])
    expression = str(source["selected_expression"])
    train_path = args.data_dir.resolve() / f"Intersection_{intersection}_Train.jsonl"
    train = preprocess_data_flexible(load_dataset_flexible(str(train_path), intersection), intersection).reset_index(drop=True)
    config = INTERSECTION_CONFIGS[intersection]
    lanes, mapping = lanes_for(config)
    targets = {item: train[f"Delay_{item}"].reset_index(drop=True) for item in config["approaches"]}
    prepared = prepare_optimization_context(train, lanes, mapping, intersection)
    diagnostics: dict = {}
    fitter = StableRangeAcceleratedFitter(parallel_workers=4, maxiter=200, maxfun=20_000)
    try:
        parameters = fitter.fit(
            universal_expr=expression, df=train, lanes=lanes, lane_to_approach=mapping,
            approach_targets=targets, intersection_id=intersection, prepared_context=prepared,
            rng=np.random.default_rng(20260930), n_restarts=10, diagnostics=diagnostics,
        )
    finally:
        fitter.close()
    prediction = calculate_approach_delays_from_universal(
        df=train, universal_expr=expression, lane_parameters=parameters,
        lanes=lanes, lane_to_approach=mapping, intersection_id=intersection, strict=True,
    )
    truth = {item: train[f"Delay_{item}"].to_numpy(dtype=float) for item in config["approaches"]}
    metrics = pooled_metrics(truth, prediction)
    if diagnostics.get("formal_method_adapter") != "cosydelay.fitter":
        raise RuntimeError(f"CoSyDelay fitter provenance missing: {diagnostics.get('formal_method_adapter')}")
    payload = {
        "status": "complete_training_only_cosydelay_fitter_smoke",
        "test_file_opened": False, "outer_validation_accessed": False,
        "intersection_id": intersection, "expression": expression,
        "metrics": metrics, "diagnostics": diagnostics,
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"r2": metrics["r2"], "range_profile": diagnostics["coefficient_range_profile"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
