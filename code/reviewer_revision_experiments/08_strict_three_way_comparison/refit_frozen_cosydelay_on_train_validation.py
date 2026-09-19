"""Refit frozen CoSyDelay structures on Training+Validation, then open Test."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

GMINI = Path(__file__).resolve().parents[2]
if str(GMINI) not in sys.path:
    sys.path.insert(0, str(GMINI))

import numpy as np
import pandas as pd

from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import lanes_for
from methods.cosydelay_v18_fit_timeout_guard.contract import V18_CONTRACT
from methods.cosydelay_v19_accelerated_equivalent.fitter import (
    PersistentAcceleratedEquivalentFitter,
)
from methods.cosydelay_v16_manuscript_principlewise.evaluate_frozen_test import (
    pooled_metrics,
)
from optimization_lane import (
    calculate_approach_delays_from_universal,
    prepare_optimization_context,
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_frame(path: Path, intersection: int) -> pd.DataFrame:
    return preprocess_data_flexible(
        load_dataset_flexible(str(path), intersection), intersection
    ).reset_index(drop=True)


def predict(frame, expression, parameters, lanes, mapping, intersection):
    return calculate_approach_delays_from_universal(
        df=frame,
        universal_expr=expression,
        lane_parameters=parameters,
        lanes=lanes,
        lane_to_approach=mapping,
        intersection_id=intersection,
        strict=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-selections", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restarts", type=int, default=10)
    args = parser.parse_args()
    output = args.output.resolve()
    data_dir = args.data_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    source = read_json(args.frozen_selections.resolve())
    if source.get("test_used_for_selection") is not False:
        raise RuntimeError("Frozen structure provenance is not Test-isolated")
    selected = sorted(source["selections"], key=lambda item: item["intersection_id"])
    selected_ids = [int(item["intersection_id"]) for item in selected]
    if (
        not selected_ids
        or any(intersection not in range(1, 7) for intersection in selected_ids)
        or len(set(selected_ids)) != len(selected_ids)
    ):
        raise RuntimeError("selections must be unique intersection ids from I1-I6")

    fitted = []
    # Refit phase: Test paths are neither constructed nor opened in this loop.
    for item in selected:
        intersection = int(item["intersection_id"])
        training = load_frame(
            data_dir / f"Intersection_{intersection}_Train.jsonl", intersection
        )
        validation = load_frame(
            data_dir / f"Intersection_{intersection}_Validation.jsonl", intersection
        )
        combined = pd.concat([training, validation], ignore_index=True)
        config = INTERSECTION_CONFIGS[intersection]
        approaches = list(config["approaches"])
        lanes, mapping = lanes_for(config)
        targets = {
            approach: combined[f"Delay_{approach}"].reset_index(drop=True)
            for approach in approaches
        }
        prepared = prepare_optimization_context(
            combined, lanes, mapping, intersection
        )
        fitter = PersistentAcceleratedEquivalentFitter(
            parallel_workers=V18_CONTRACT.approach_workers_cap,
            maxiter=V18_CONTRACT.optimizer_maxiter,
            maxfun=V18_CONTRACT.optimizer_maxfun,
        )
        diagnostics = {}
        try:
            parameters = fitter.fit(
                universal_expr=item["expression"],
                warm_parameters=None,
                df=combined,
                lanes=lanes,
                lane_to_approach=mapping,
                approach_targets=targets,
                intersection_id=intersection,
                prepared_context=prepared,
                rng=np.random.default_rng(
                    20260821
                    + intersection * 1000
                    + int(item["candidate_id"])
                ),
                n_restarts=args.restarts,
                diagnostics=diagnostics,
            )
        finally:
            fitter.close()
        fitted.append(
            {
                **item,
                "coefficient_refit_split": "training_plus_validation",
                "coefficient_refit_rows": len(combined),
                "coefficient_refit_restarts": args.restarts,
                "parameters": parameters,
                "fit_diagnostics": diagnostics,
            }
        )
        print(f"I{intersection}: Training+Validation coefficients fitted", flush=True)

    frozen = {
        "status": "all_structures_and_coefficients_frozen_before_test",
        "structure_selection_split": "validation",
        "coefficient_refit_split": "training_plus_validation",
        "test_used_for_structure_or_coefficient_fit": False,
        "source_frozen_selections": str(args.frozen_selections.resolve()),
        "selections": fitted,
    }
    write_json(output / "FROZEN_REFIT.json", frozen)

    rows = []
    per_intersection = []
    for item in fitted:
        intersection = int(item["intersection_id"])
        test = load_frame(
            data_dir / f"Intersection_{intersection}_Test.jsonl", intersection
        )
        config = INTERSECTION_CONFIGS[intersection]
        approaches = list(config["approaches"])
        lanes, mapping = lanes_for(config)
        truth = {
            approach: test[f"Delay_{approach}"].to_numpy(dtype=float)
            for approach in approaches
        }
        estimates = predict(
            test,
            item["expression"],
            item["parameters"],
            lanes,
            mapping,
            intersection,
        )
        values = pooled_metrics(truth, estimates)
        per_intersection.append({"intersection_id": intersection, **values})
        for approach in approaches:
            for y_true, y_pred in zip(truth[approach], estimates[approach]):
                rows.append(
                    {
                        "method": "CoSyDelay-V21-20restart-validation-refit",
                        "run_id": 1,
                        "intersection_id": intersection,
                        "approach": approach,
                        "split": "test",
                        "y_true": float(y_true),
                        "y_pred": float(y_pred),
                    }
                )
        print(f"I{intersection}: test R2={values['r2']:.6f}", flush=True)

    with (output / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {
        "status": "complete",
        "method": "CoSyDelay-V21-20restart-validation-refit",
        "test_used_for_structure_or_coefficient_fit": False,
        "mean_test_pooled_r2": float(
            np.mean([item["r2"] for item in per_intersection])
        ),
        "mean_test_pooled_rmse": float(
            np.mean([item["rmse"] for item in per_intersection])
        ),
        "mean_test_pooled_mae": float(
            np.mean([item["mae"] for item in per_intersection])
        ),
        "intersections": per_intersection,
    }
    write_json(output / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
