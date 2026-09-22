"""Drop-in L-BFGS-B coefficient fitter with approach-level parallelism.

Scientific behavior is intentionally frozen to ``optimization_lane.py``:

* the same summed approach-MSE objective;
* the same role-derived coefficient bounds;
* the same random-uniform initialization in [0.1, 1.0];
* the same deterministic RNG consumption order;
* the same per-approach lowest-fit-MSE restart selection; and
* the same lane-to-approach aggregation.

Only the three independent approach blocks are executed concurrently.  All
restart points are generated serially in the parent before dispatch, so using
three workers cannot alter the random sequence.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numexpr as ne
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from optimization_lane import (
    build_parameter_bounds,
    evaluate_lane_delay,
    extract_coefficients,
    prepare_optimization_context,
    unpack_lane_parameters,
)


OFFICIAL_RESTARTS = 10
OFFICIAL_MAXITER = 200
OFFICIAL_MAXFUN = 20_000
OFFICIAL_PARALLEL_WORKERS = 3


def _solve_approach(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Solve one independent approach using frozen, parent-generated starts."""
    # Each expression evaluation is only about 10^3 rows. One numexpr thread
    # per process avoids 3 x 16 nested threads while preserving elementwise
    # numerical results.
    ne.set_num_threads(1)
    expression = str(payload["expression"])
    coefficient_names = list(payload["coefficient_names"])
    block_lanes = list(payload["block_lanes"])
    lane_contexts = payload["lane_contexts"]
    weights = payload["weights"]
    target_values = np.asarray(payload["target_values"], dtype=float)
    valid = np.asarray(payload["valid"], dtype=bool)
    initials = np.asarray(payload["initials"], dtype=float)
    bounds = [tuple(item) for item in payload["bounds"]]
    maxiter = int(payload["maxiter"])
    maxfun = int(payload["maxfun"])

    def objective(values: np.ndarray) -> float:
        lane_parameters = unpack_lane_parameters(
            values, block_lanes, coefficient_names
        )
        prediction = np.zeros(len(target_values), dtype=float)
        try:
            for lane in block_lanes:
                prediction += (
                    np.asarray(weights[lane], dtype=float)
                    * evaluate_lane_delay(
                        expression,
                        lane_contexts[lane],
                        lane_parameters[lane],
                    )
                )
        except Exception:
            return 1e12
        if not np.any(valid):
            return 1e12
        residual = prediction[valid] - target_values[valid]
        if residual.size == 0 or np.any(~np.isfinite(residual)):
            return 1e12
        mse = np.mean(residual**2)
        return float(mse) if np.isfinite(mse) else 1e12

    records = []
    approach_started = time.perf_counter()
    for restart_index, initial in enumerate(initials):
        started = time.perf_counter()
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "maxfun": maxfun,
                "disp": False,
            },
        )
        records.append(
            {
                "index": int(restart_index),
                "values": np.asarray(result.x, dtype=float),
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "objective": float(result.fun),
                "iterations": int(getattr(result, "nit", 0)),
                "function_evaluations": int(getattr(result, "nfev", 0)),
                "wall_seconds": float(time.perf_counter() - started),
            }
        )
    chosen_index = int(
        np.argmin(
            [
                item["objective"]
                if np.isfinite(item["objective"])
                else np.inf
                for item in records
            ]
        )
    )
    chosen = records[chosen_index]
    return {
        "approach": str(payload["approach"]),
        "block_lanes": block_lanes,
        "coefficient_names": coefficient_names,
        "values": np.asarray(chosen["values"], dtype=float),
        "chosen_restart": chosen_index,
        "success": bool(chosen["success"]),
        "status": int(chosen["status"]),
        "message": str(chosen["message"]),
        "objective": float(chosen["objective"]),
        "iterations": int(chosen["iterations"]),
        "function_evaluations": int(chosen["function_evaluations"]),
        "chosen_wall_seconds": float(chosen["wall_seconds"]),
        "total_wall_seconds": float(time.perf_counter() - approach_started),
        "fitted_rows": int(np.count_nonzero(valid)),
        "zero_flow_rows_excluded": int(payload["zero_flow_rows_excluded"]),
        "zero_flow_policy": str(payload["zero_flow_policy"]),
        "total_function_evaluations": int(
            sum(item["function_evaluations"] for item in records)
        ),
        "restarts": [
            {key: value for key, value in item.items() if key != "values"}
            for item in records
        ],
    }


def _generate_payloads(
    *,
    universal_expr: str,
    lanes: Sequence[str],
    approach_targets: Mapping[str, pd.Series],
    prepared: Mapping[str, Any],
    rng: np.random.Generator,
    n_restarts: int,
    maxiter: int,
    maxfun: int,
    coefficient_bounds_override: Optional[
        Mapping[str, Tuple[float, float]]
    ],
) -> tuple[list[str], list[Dict[str, Any]]]:
    """Generate starts in exactly the production serial RNG order."""
    coefficient_names = extract_coefficients(universal_expr)
    n_coefficients = len(coefficient_names)
    n_params = len(lanes) * n_coefficients
    initial_params = rng.uniform(0.1, 1.0, n_params)
    initial_by_lane = unpack_lane_parameters(
        initial_params, list(lanes), coefficient_names
    )
    coefficient_bounds = build_parameter_bounds(
        universal_expr, coefficient_names
    )
    if coefficient_bounds_override:
        coefficient_bounds = [
            coefficient_bounds_override.get(name, bound)
            for name, bound in zip(coefficient_names, coefficient_bounds)
        ]
    lane_contexts = prepared["lane_contexts"]
    approach_weights = prepared["approach_weights"]
    payloads: list[Dict[str, Any]] = []

    # Preserve approach_targets insertion order and consume every random draw
    # before any worker is launched.
    for approach, target_delay in approach_targets.items():
        if approach not in approach_weights:
            continue
        weight_context = approach_weights[approach]
        block_lanes = list(weight_context["lanes"])
        block_initial = np.asarray(
            [
                initial_by_lane[lane][name]
                for lane in block_lanes
                for name in coefficient_names
            ],
            dtype=float,
        )
        block_bounds = coefficient_bounds * len(block_lanes)
        initials = [block_initial]
        for _ in range(1, n_restarts):
            lower = np.asarray(
                [max(bound[0], 0.1) for bound in block_bounds],
                dtype=float,
            )
            upper = np.asarray(
                [min(bound[1], 1.0) for bound in block_bounds],
                dtype=float,
            )
            upper = np.maximum(upper, lower)
            initials.append(rng.uniform(lower, upper))
        target_values = np.asarray(target_delay.values, dtype=float)
        valid = (
            np.asarray(weight_context["valid_mask"], dtype=bool)
            & np.isfinite(target_values)
        )
        payloads.append(
            {
                "approach": str(approach),
                "expression": universal_expr,
                "coefficient_names": coefficient_names,
                "block_lanes": block_lanes,
                "lane_contexts": {
                    lane: lane_contexts[lane] for lane in block_lanes
                },
                "weights": {
                    lane: weight_context["weights"][lane]
                    for lane in block_lanes
                },
                "target_values": target_values,
                "valid": valid,
                "bounds": block_bounds,
                "initials": np.asarray(initials, dtype=float),
                "maxiter": int(maxiter),
                "maxfun": int(maxfun),
                "fitted_rows": int(
                    np.count_nonzero(weight_context["valid_mask"])
                ),
                "zero_flow_rows_excluded": int(
                    weight_context["zero_flow_rows"]
                ),
                "zero_flow_policy": weight_context["zero_flow_policy"],
            }
        )
    return coefficient_names, payloads


def fit_lane_parameters_to_approaches_parallel(
    universal_expr: str,
    df: pd.DataFrame,
    lanes: Sequence[str],
    lane_to_approach: Mapping[str, str],
    approach_targets: Mapping[str, pd.Series],
    intersection_id: int,
    verbose: bool = False,
    prepared_context: Optional[Mapping[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
    n_restarts: int = OFFICIAL_RESTARTS,
    diagnostics: Optional[Dict[str, Any]] = None,
    coefficient_bounds_override: Optional[
        Mapping[str, Tuple[float, float]]
    ] = None,
    *,
    parallel_workers: int = OFFICIAL_PARALLEL_WORKERS,
    maxiter: Optional[int] = None,
    maxfun: Optional[int] = None,
    executor: Optional[ProcessPoolExecutor] = None,
) -> Dict[str, Dict[str, float]]:
    """Drop-in production fitter with parallel independent approaches."""
    if n_restarts < 1 or parallel_workers < 1:
        raise ValueError("n_restarts and parallel_workers must be positive")
    coefficient_names = extract_coefficients(universal_expr)
    if not coefficient_names:
        return {str(lane): {} for lane in lanes}
    rng = rng or np.random.default_rng()
    effective_maxiter = int(
        maxiter
        if maxiter is not None
        else os.environ.get("COSY_OPTIMIZER_MAXITER", OFFICIAL_MAXITER)
    )
    effective_maxfun = int(
        maxfun
        if maxfun is not None
        else os.environ.get("COSY_OPTIMIZER_MAXFUN", OFFICIAL_MAXFUN)
    )
    prepared = prepared_context or prepare_optimization_context(
        df,
        list(lanes),
        dict(lane_to_approach),
        int(intersection_id),
    )
    coefficient_names, payloads = _generate_payloads(
        universal_expr=universal_expr,
        lanes=list(lanes),
        approach_targets=approach_targets,
        prepared=prepared,
        rng=rng,
        n_restarts=int(n_restarts),
        maxiter=effective_maxiter,
        maxfun=effective_maxfun,
        coefficient_bounds_override=coefficient_bounds_override,
    )
    if not payloads:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "solver": "L-BFGS-B",
                    "n_restarts": int(n_restarts),
                    "maxiter": effective_maxiter,
                    "maxfun": effective_maxfun,
                    "parallel_workers": 0,
                    "rng_initialization_semantics": (
                        "identical_to_optimization_lane_serial"
                    ),
                    "objective_sum": 0.0,
                    "outer_wall_seconds": 0.0,
                    "serial_work_seconds": 0.0,
                    "parallel_efficiency_ratio": None,
                    "approaches": [],
                }
            )
        return {}
    started = time.perf_counter()
    workers = min(int(parallel_workers), len(payloads))
    owns_executor = executor is None
    active_executor = executor
    if workers > 1 and active_executor is None:
        active_executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
    try:
        if workers == 1:
            records = [_solve_approach(payload) for payload in payloads]
        else:
            assert active_executor is not None
            records = list(active_executor.map(_solve_approach, payloads))
    finally:
        if owns_executor and active_executor is not None:
            active_executor.shutdown(wait=True)
    outer_wall_seconds = time.perf_counter() - started
    fitted: Dict[str, Dict[str, float]] = {}
    fit_records = []
    for record in records:
        fitted.update(
            unpack_lane_parameters(
                np.asarray(record["values"], dtype=float),
                list(record["block_lanes"]),
                coefficient_names,
            )
        )
        fit_records.append(
            {
                "approach": record["approach"],
                "chosen_restart": record["chosen_restart"],
                "success": record["success"],
                "status": record["status"],
                "message": record["message"],
                "objective": record["objective"],
                "iterations": record["iterations"],
                "function_evaluations": record["function_evaluations"],
                "wall_seconds": record["chosen_wall_seconds"],
                "fitted_rows": record["fitted_rows"],
                "zero_flow_rows_excluded": record[
                    "zero_flow_rows_excluded"
                ],
                "zero_flow_policy": record["zero_flow_policy"],
                "total_wall_seconds": record["total_wall_seconds"],
                "total_function_evaluations": record[
                    "total_function_evaluations"
                ],
                "restarts": record["restarts"],
            }
        )
    objective_sum = float(
        sum(float(record["objective"]) for record in records)
    )
    if diagnostics is not None:
        serial_work_seconds = float(
            sum(float(record["total_wall_seconds"]) for record in records)
        )
        diagnostics.update(
            {
                "solver": "L-BFGS-B",
                "n_restarts": int(n_restarts),
                "maxiter": effective_maxiter,
                "maxfun": effective_maxfun,
                "parallel_workers": workers,
                "rng_initialization_semantics": (
                    "identical_to_optimization_lane_serial"
                ),
                "objective_sum": objective_sum,
                "outer_wall_seconds": float(outer_wall_seconds),
                "serial_work_seconds": serial_work_seconds,
                "parallel_efficiency_ratio": (
                    serial_work_seconds / outer_wall_seconds
                    if outer_wall_seconds > 0.0
                    else None
                ),
                "approaches": fit_records,
            }
        )
    if verbose:
        print(
            f"  Parallel L-BFGS-B: approaches={len(records)}, "
            f"restarts={n_restarts}, objective={objective_sum:.4f}, "
            f"wall={outer_wall_seconds:.2f}s"
        )
    return fitted


class ParallelApproachFitter:
    """Persistent drop-in callable for evolutionary searches.

    Reusing one process pool avoids process-start overhead for every candidate.
    Call :meth:`close` or use the class as a context manager.
    """

    def __init__(
        self,
        *,
        parallel_workers: int = OFFICIAL_PARALLEL_WORKERS,
        maxiter: int = OFFICIAL_MAXITER,
        maxfun: int = OFFICIAL_MAXFUN,
    ) -> None:
        if min(parallel_workers, maxiter, maxfun) < 1:
            raise ValueError("parallel_workers and budgets must be positive")
        self.parallel_workers = int(parallel_workers)
        self.maxiter = int(maxiter)
        self.maxfun = int(maxfun)
        self._executor: Optional[ProcessPoolExecutor] = ProcessPoolExecutor(
            max_workers=self.parallel_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )

    def __call__(
        self,
        universal_expr: str,
        df: pd.DataFrame,
        lanes: Sequence[str],
        lane_to_approach: Mapping[str, str],
        approach_targets: Mapping[str, pd.Series],
        intersection_id: int,
        verbose: bool = False,
        prepared_context: Optional[Mapping[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
        n_restarts: int = OFFICIAL_RESTARTS,
        diagnostics: Optional[Dict[str, Any]] = None,
        coefficient_bounds_override: Optional[
            Mapping[str, Tuple[float, float]]
        ] = None,
    ) -> Dict[str, Dict[str, float]]:
        if self._executor is None:
            raise RuntimeError(
                "ParallelApproachFitter is closed; create a new instance"
            )
        return fit_lane_parameters_to_approaches_parallel(
            universal_expr=universal_expr,
            df=df,
            lanes=lanes,
            lane_to_approach=lane_to_approach,
            approach_targets=approach_targets,
            intersection_id=intersection_id,
            verbose=verbose,
            prepared_context=prepared_context,
            rng=rng,
            n_restarts=n_restarts,
            diagnostics=diagnostics,
            coefficient_bounds_override=coefficient_bounds_override,
            parallel_workers=self.parallel_workers,
            maxiter=self.maxiter,
            maxfun=self.maxfun,
            executor=self._executor,
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "ParallelApproachFitter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
