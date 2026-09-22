"""All-positive-log-coordinate L-BFGS-B with the retained ten starts.

Only the optimizer coordinates change.  The expression, per-approach summed
MSE objective, analytic Jacobian, coefficient bounds, restart points, R9
feasibility rule, restart count, and approach-level parallelization are kept.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numexpr as ne
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from methods.cosydelay._internal.optimizer_parallel4_v3.fitter import (
    _mse_and_gradient_quiet,
    _r9_probe_restart,
    generate_mixed_restart_payloads,
)
from methods.cosydelay._internal.optimizer_jacobian.jacobian_fitter import (
    _symbolic_derivative_strings,
)
from optimization_lane import (
    extract_coefficients,
    prepare_optimization_context,
    unpack_lane_parameters,
)


START_STRATEGY_ID = "prospective_all10_log_coordinates_retained_raw_starts_v1"


def _to_log_space(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("all-log parameterization requires finite positive values")
    return np.log(values)


def _from_log_space(values: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(values, dtype=float))


def _mse_and_all_log_gradient(
    log_values: np.ndarray,
    payload: Mapping[str, Any],
) -> tuple[float, np.ndarray]:
    raw_values = _from_log_space(log_values)
    objective, raw_gradient = _mse_and_gradient_quiet(raw_values, payload)
    return float(objective), np.asarray(raw_gradient, dtype=float) * raw_values


def _solve_approach_all_log(payload: Mapping[str, Any]) -> Dict[str, Any]:
    ne.set_num_threads(1)
    initials = np.asarray(payload["initials"], dtype=float)
    raw_bounds = [tuple(map(float, item)) for item in payload["bounds"]]
    if any(lower <= 0.0 or upper <= lower for lower, upper in raw_bounds):
        raise ValueError("all-log parameterization requires ordered positive bounds")
    log_bounds = [
        (float(np.log(lower)), float(np.log(upper)))
        for lower, upper in raw_bounds
    ]
    records = []
    approach_started = time.perf_counter()
    restart_kinds = list(payload["restart_kinds"])
    for restart_index, initial in enumerate(initials):
        started = time.perf_counter()
        result = minimize(
            _mse_and_all_log_gradient,
            _to_log_space(initial),
            args=(payload,),
            method="L-BFGS-B",
            jac=True,
            bounds=log_bounds,
            options={
                "maxiter": int(payload["maxiter"]),
                "maxfun": int(payload["maxfun"]),
                "disp": False,
            },
        )
        raw_values = _from_log_space(result.x)
        r9_probe_pass, r9_probe_delays = _r9_probe_restart(raw_values, payload)
        records.append(
            {
                "index": int(restart_index),
                "parameterization": "all_positive_log",
                "start_kind": restart_kinds[restart_index],
                "values": raw_values,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "objective": float(result.fun),
                "iterations": int(getattr(result, "nit", 0)),
                "function_evaluations": int(getattr(result, "nfev", 0)),
                "gradient_evaluations": int(getattr(result, "njev", 0)),
                "gradient_inf_norm_optimization_space": float(
                    np.linalg.norm(
                        np.asarray(getattr(result, "jac", []), dtype=float),
                        ord=np.inf,
                    )
                ),
                "r9_probe_pass": bool(r9_probe_pass),
                "r9_probe_delays_seconds": r9_probe_delays,
                "wall_seconds": float(time.perf_counter() - started),
            }
        )

    r9_constrained = bool(payload.get("r9_constrained_restart_selection", True))
    feasible_indices = [
        index
        for index, item in enumerate(records)
        if item["r9_probe_pass"] and np.isfinite(item["objective"])
    ]
    selection_indices = (
        feasible_indices or list(range(len(records)))
        if r9_constrained
        else list(range(len(records)))
    )
    chosen_index = min(
        selection_indices,
        key=lambda index: (
            records[index]["objective"]
            if np.isfinite(records[index]["objective"])
            else np.inf
        ),
    )
    chosen = records[chosen_index]
    return {
        "approach": str(payload["approach"]),
        "block_lanes": list(payload["block_lanes"]),
        "coefficient_names": list(payload["coefficient_names"]),
        "values": np.asarray(chosen["values"], dtype=float),
        "chosen_restart": int(chosen_index),
        "success": bool(chosen["success"]),
        "status": int(chosen["status"]),
        "message": str(chosen["message"]),
        "objective": float(chosen["objective"]),
        "iterations": int(chosen["iterations"]),
        "function_evaluations": int(chosen["function_evaluations"]),
        "gradient_evaluations": int(chosen["gradient_evaluations"]),
        "gradient_inf_norm_optimization_space": float(
            chosen["gradient_inf_norm_optimization_space"]
        ),
        "chosen_wall_seconds": float(chosen["wall_seconds"]),
        "total_wall_seconds": float(time.perf_counter() - approach_started),
        "fitted_rows": int(payload["fitted_rows"]),
        "zero_flow_rows_excluded": int(payload["zero_flow_rows_excluded"]),
        "zero_flow_policy": str(payload["zero_flow_policy"]),
        "chosen_parameterization": "all_positive_log",
        "r9_constrained_restart_selection": r9_constrained,
        "r9_feasible_restart_count": int(len(feasible_indices)),
        "r9_feasible_solution_available": bool(feasible_indices),
        "total_function_evaluations": int(
            sum(item["function_evaluations"] for item in records)
        ),
        "total_gradient_evaluations": int(
            sum(item["gradient_evaluations"] for item in records)
        ),
        "restarts": [
            {key: value for key, value in item.items() if key != "values"}
            for item in records
        ],
        "restart_kinds": restart_kinds,
        "warm_values_inherited": int(payload["warm_values_inherited"]),
    }


def fit_lane_parameters_all_log_parallel(
    universal_expr: str,
    df: pd.DataFrame,
    lanes: Sequence[str],
    lane_to_approach: Mapping[str, str],
    approach_targets: Mapping[str, pd.Series],
    intersection_id: int,
    verbose: bool = False,
    prepared_context: Optional[Mapping[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
    n_restarts: int = 10,
    diagnostics: Optional[Dict[str, Any]] = None,
    coefficient_bounds_override: Optional[
        Mapping[str, Tuple[float, float]]
    ] = None,
    *,
    warm_parameters: Optional[Mapping[str, Mapping[str, float]]] = None,
    parallel_workers: int = 4,
    maxiter: int = 200,
    maxfun: int = 20_000,
    executor: Optional[ProcessPoolExecutor] = None,
    r9_constrained_restart_selection: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Fit one expression with the retained starts in all-log coordinates."""
    if min(n_restarts, parallel_workers, maxiter, maxfun) < 1:
        raise ValueError("restarts, workers, and budgets must be positive")
    coefficient_names = extract_coefficients(universal_expr)
    if not coefficient_names:
        return {str(lane): {} for lane in lanes}
    rng = rng or np.random.default_rng()
    prepared = prepared_context or prepare_optimization_context(
        df, list(lanes), dict(lane_to_approach), int(intersection_id)
    )

    build_started = time.perf_counter()
    derivative_expressions = _symbolic_derivative_strings(
        universal_expr, tuple(coefficient_names)
    )
    coefficient_names, payloads, base_seed = generate_mixed_restart_payloads(
        universal_expr=universal_expr,
        lanes=list(lanes),
        approach_targets=approach_targets,
        prepared=prepared,
        rng=rng,
        n_restarts=int(n_restarts),
        maxiter=int(maxiter),
        maxfun=int(maxfun),
        coefficient_bounds_override=coefficient_bounds_override,
        warm_parameters=warm_parameters,
    )
    for payload in payloads:
        payload["derivative_expressions"] = derivative_expressions
        payload["r9_constrained_restart_selection"] = bool(
            r9_constrained_restart_selection
        )
    symbolic_build_seconds = float(time.perf_counter() - build_started)
    if not payloads:
        return {}

    started = time.perf_counter()
    workers = min(int(parallel_workers), len(payloads))
    owns_executor = executor is None
    active_executor = executor
    if workers > 1 and active_executor is None:
        active_executor = ProcessPoolExecutor(max_workers=workers)
    try:
        if workers == 1:
            records = [_solve_approach_all_log(item) for item in payloads]
        else:
            assert active_executor is not None
            records = list(active_executor.map(_solve_approach_all_log, payloads))
    finally:
        if owns_executor and active_executor is not None:
            active_executor.shutdown(wait=True)
    outer_wall_seconds = float(time.perf_counter() - started)

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
                key: value
                for key, value in record.items()
                if key not in {"values", "block_lanes", "coefficient_names"}
            }
        )

    if diagnostics is not None:
        serial_work_seconds = float(
            sum(float(item["total_wall_seconds"]) for item in records)
        )
        diagnostics.update(
            {
                "solver": "L-BFGS-B",
                "jacobian": "analytic_sympy_numexpr",
                "parameterization": "all_positive_log",
                "start_strategy_id": START_STRATEGY_ID,
                "n_restarts": int(n_restarts),
                "maxiter": int(maxiter),
                "maxfun": int(maxfun),
                "parallel_workers": workers,
                "raw_restart_points_retained": True,
                "r9_constrained_restart_selection": bool(
                    r9_constrained_restart_selection
                ),
                "sobol_base_seed": base_seed,
                "parent_warm_start_available": warm_parameters is not None,
                "objective_sum": float(
                    sum(float(item["objective"]) for item in records)
                ),
                "outer_wall_seconds": outer_wall_seconds,
                "symbolic_build_seconds": symbolic_build_seconds,
                "serial_work_seconds": serial_work_seconds,
                "parallel_efficiency_ratio": (
                    serial_work_seconds / outer_wall_seconds
                    if outer_wall_seconds > 0.0
                    else None
                ),
                "total_function_evaluations": int(
                    sum(item["total_function_evaluations"] for item in records)
                ),
                "total_gradient_evaluations": int(
                    sum(item["total_gradient_evaluations"] for item in records)
                ),
                "approaches": fit_records,
                "prospective_only": True,
            }
        )
    if verbose:
        print(
            f"  all-log L-BFGS-B: approaches={len(records)}, "
            f"restarts={n_restarts}, wall={outer_wall_seconds:.2f}s"
        )
    return fitted
