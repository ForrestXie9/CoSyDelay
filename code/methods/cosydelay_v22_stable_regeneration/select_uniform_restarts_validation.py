"""Freeze one V22 structure per intersection using Validation only.

For every uniform outer restart, this refits its Training top-k strict
structures on Training, evaluates them on Validation, and then selects the
single best Validation structure across all restarts.  No Test path is opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import lanes_for
from methods.cosydelay_v18_fit_timeout_guard.contract import V18_CONTRACT
from methods.cosydelay_v16_manuscript_principlewise.evaluate_frozen_test import pooled_metrics
from optimization_lane import calculate_approach_delays_from_universal, prepare_optimization_context
from .fitter import StableRangeAcceleratedFitter


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load(path: Path, intersection: int):
    return preprocess_data_flexible(load_dataset_flexible(str(path), intersection), intersection).reset_index(drop=True)


def _top_strict(history: list[dict], count: int) -> list[dict]:
    best: dict[str, dict] = {}
    for item in history:
        if item.get("event") != "evaluated" or not item.get("physical_joint_pass"):
            continue
        expression = str(item["expression"])
        if expression not in best or float(item["train_r2"]) > float(best[expression]["train_r2"]):
            best[expression] = item
    ranked = sorted(best.values(), key=lambda item: float(item["train_r2"]), reverse=True)
    if len(ranked) < count:
        raise RuntimeError(f"only {len(ranked)} strict unique candidates; require top-{count}")
    return ranked[:count]


def _predict(frame, expression, parameters, lanes, mapping, intersection):
    return calculate_approach_delays_from_universal(
        df=frame, universal_expr=expression, lane_parameters=parameters,
        lanes=lanes, lane_to_approach=mapping, intersection_id=intersection, strict=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.restarts < 1 or args.top_k < 1:
        raise ValueError("restarts and top-k must be positive")
    root, data, output = args.search_root.resolve(), args.data_dir.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    candidates_out: list[dict] = []
    selections: list[dict] = []

    # This complete loop constructs only Train and Validation paths.
    for intersection in range(1, 7):
        train = _load(data / f"Intersection_{intersection}_Train.jsonl", intersection)
        validation = _load(data / f"Intersection_{intersection}_Validation.jsonl", intersection)
        config = INTERSECTION_CONFIGS[intersection]
        approaches = list(config["approaches"])
        lanes, mapping = lanes_for(config)
        targets = {item: train[f"Delay_{item}"].reset_index(drop=True) for item in approaches}
        validation_targets = {item: validation[f"Delay_{item}"].to_numpy(dtype=float) for item in approaches}
        prepared = prepare_optimization_context(train, lanes, mapping, intersection)
        all_candidates: list[dict] = []
        # Validation selection must refit with the same V22 role-wise bounds
        # used during Training search.  Using the inherited V19 fitter here
        # would silently change the coefficient-range contract between splits.
        fitter = StableRangeAcceleratedFitter(
            parallel_workers=V18_CONTRACT.approach_workers_cap,
            maxiter=V18_CONTRACT.optimizer_maxiter,
            maxfun=V18_CONTRACT.optimizer_maxfun,
        )
        try:
            for restart in range(1, args.restarts + 1):
                run_dir = root / f"repeat_{restart:02d}" / "search" / f"intersection_{intersection:02d}" / "run_01"
                result = _read(run_dir / "result.json")
                if result.get("formal_version") != "V22-experimental":
                    raise RuntimeError(f"not a V22 result: {run_dir}")
                for rank, item in enumerate(_top_strict(_read(run_dir / "history.json"), args.top_k), 1):
                    diagnostics: dict = {}
                    parameters = fitter.fit(
                        universal_expr=str(item["expression"]), warm_parameters=None,
                        df=train, lanes=lanes, lane_to_approach=mapping,
                        approach_targets=targets, intersection_id=intersection,
                        prepared_context=prepared,
                        rng=np.random.default_rng(20260911 + restart * 10_000 + intersection * 100 + int(item["candidate_id"])),
                        n_restarts=10, diagnostics=diagnostics,
                    )
                    metrics = pooled_metrics(validation_targets, _predict(validation, str(item["expression"]), parameters, lanes, mapping, intersection))
                    record = {
                        "intersection_id": intersection, "restart": restart,
                        "training_rank": rank, "candidate_id": int(item["candidate_id"]),
                        "expression": str(item["expression"]),
                        "archived_training_r2": float(item["train_r2"]),
                        "validation_r2": float(metrics["r2"]),
                        "validation_rmse": float(metrics["rmse"]),
                        "validation_mae": float(metrics["mae"]),
                    }
                    all_candidates.append(record)
                    candidates_out.append(record)
                    print(f"I{intersection} restart={restart} rank={rank}: Validation R2={record['validation_r2']:.6f}", flush=True)
        finally:
            fitter.close()
        winner = max(all_candidates, key=lambda item: (item["validation_r2"], -item["restart"], -item["training_rank"]))
        selections.append(winner)
        print(f"I{intersection}: frozen restart={winner['restart']} rank={winner['training_rank']} Validation R2={winner['validation_r2']:.6f}", flush=True)

    frozen = {
        "status": "all_v22_structures_frozen_before_test",
        "method": "CoSyDelay-V22-uniform-restarts",
        "selection_split": "validation",
        "selection_metric": "pooled_r2",
        "search_restarts_per_intersection": args.restarts,
        "training_top_k_per_restart": args.top_k,
        "coefficient_fit_split": "training_only",
        "test_used_for_selection": False,
        "selections": selections,
    }
    (output / "FROZEN_SELECTIONS.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "candidate_validation_metrics.json").write_text(json.dumps(candidates_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
