"""Offline numerical-equivalence smoke for one real I1 candidate fit."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
from optimization_lane import prepare_optimization_context
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import (
    DEFAULT_DATA_DIR,
    jsonable,
    lanes_for,
)
from methods.cosydelay_v15_symbolic_r9.fitter import PersistentSymbolicR9Fitter
from methods.cosydelay_v16_manuscript_principlewise.fitter import (
    legacy_numeric_r9_probe_disabled,
    scrub_legacy_probe_fields,
)

from .contract import V18_CONTRACT
from .fit_timeout import GuardedCandidateFitter


EXPRESSION = "Cycle_Time*a1*flow_lane/GR_phase"
SEED = 20270831


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _fit_kwargs(train, lanes, mapping, targets, prepared, rng, diagnostics):
    return {
        "df": train,
        "lanes": lanes,
        "lane_to_approach": mapping,
        "approach_targets": targets,
        "intersection_id": 1,
        "prepared_context": prepared,
        "rng": rng,
        "n_restarts": 10,
        "diagnostics": diagnostics,
    }


def _maximum_parameter_difference(left: dict, right: dict) -> float:
    values = []
    if set(left) != set(right):
        return float("inf")
    for lane in left:
        if set(left[lane]) != set(right[lane]):
            return float("inf")
        values.extend(
            abs(float(left[lane][name]) - float(right[lane][name]))
            for name in left[lane]
        )
    return max(values, default=0.0)


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
    prepared = prepare_optimization_context(train, lanes, mapping, 1)
    static = {
        "df": train,
        "lanes": lanes,
        "lane_to_approach": mapping,
        "approach_targets": targets,
        "intersection_id": 1,
        "prepared_context": prepared,
    }
    direct_rng = np.random.default_rng(SEED)
    guarded_rng = np.random.default_rng(SEED)
    direct_diagnostics: dict = {}
    guarded_diagnostics: dict = {}
    direct_started = time.perf_counter()
    with legacy_numeric_r9_probe_disabled():
        with PersistentSymbolicR9Fitter(
            parallel_workers=V18_CONTRACT.approach_workers_cap,
            maxiter=V18_CONTRACT.optimizer_maxiter,
            maxfun=V18_CONTRACT.optimizer_maxfun,
        ) as direct_fitter:
            direct = direct_fitter.fit(
                universal_expr=EXPRESSION,
                warm_parameters=None,
                **_fit_kwargs(
                    train,
                    lanes,
                    mapping,
                    targets,
                    prepared,
                    direct_rng,
                    direct_diagnostics,
                ),
            )
    direct_wall = time.perf_counter() - direct_started
    guarded_started = time.perf_counter()
    with GuardedCandidateFitter(
        static_fit_kwargs=static,
        parallel_workers=V18_CONTRACT.approach_workers_cap,
        maxiter=V18_CONTRACT.optimizer_maxiter,
        maxfun=V18_CONTRACT.optimizer_maxfun,
        fit_timeout_seconds=V18_CONTRACT.candidate_fit_wall_timeout_seconds,
        startup_timeout_seconds=V18_CONTRACT.candidate_fit_startup_timeout_seconds,
        close_timeout_seconds=V18_CONTRACT.candidate_fit_close_timeout_seconds,
    ) as guarded_fitter:
        guarded = guarded_fitter.fit(
            universal_expr=EXPRESSION,
            warm_parameters=None,
            **_fit_kwargs(
                train,
                lanes,
                mapping,
                targets,
                prepared,
                guarded_rng,
                guarded_diagnostics,
            ),
        )
    guarded_wall = time.perf_counter() - guarded_started
    scrub_legacy_probe_fields(direct_diagnostics)
    maximum_difference = _maximum_parameter_difference(direct, guarded)
    rng_equal = direct_rng.bit_generator.state == guarded_rng.bit_generator.state
    result = {
        "status": "pass" if maximum_difference <= 1e-12 and rng_equal else "fail",
        "role": "offline implementation-equivalence smoke; not accuracy evidence",
        "intersection_id": 1,
        "accessed_splits": ["train"],
        "test_file_opened": False,
        "expression": EXPRESSION,
        "seed": SEED,
        "optimizer": V18_CONTRACT.optimizer,
        "optimizer_restarts": V18_CONTRACT.optimizer_restarts,
        "approach_workers": V18_CONTRACT.approach_workers_cap,
        "direct_wall_seconds": direct_wall,
        "guarded_wall_seconds": guarded_wall,
        "maximum_absolute_parameter_difference": maximum_difference,
        "optimizer_rng_states_equal": rng_equal,
        "direct_parameters": direct,
        "guarded_parameters": guarded,
        "direct_diagnostics": direct_diagnostics,
        "guarded_diagnostics": guarded_diagnostics,
        "fit_guard_audit": copy.deepcopy(guarded_fitter.audit),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"real I1 fit equivalence: {result['status']}; "
        f"max_abs_diff={maximum_difference:.3e}; rng_equal={rng_equal}; "
        f"direct={direct_wall:.2f}s; guarded={guarded_wall:.2f}s",
        flush=True,
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

