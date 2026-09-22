"""Equivalent restart-level scheduling experiment for the frozen V18 fitter.

This module deliberately keeps the V18 objective, analytic Jacobian, bounds,
initial restart vectors and restart-selection rule unchanged.  It changes only
when independent L-BFGS-B calls are scheduled.  It is experimental and is not
imported by the frozen formal method.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from methods.cosydelay._internal.optimizer_parallel4_v3.fitter import (
    _r9_probe_restart,
    generate_mixed_restart_payloads,
)
from methods.cosydelay._internal.optimizer_jacobian.jacobian_fitter import (
    _symbolic_derivative_strings,
)
from methods.cosydelay._internal.optimizer_jacobian.jacobian_fitter import _penalty, _row_array
from methods.cosydelay._internal.optimizer_conditioning.all_log_fitter import (
    START_STRATEGY_ID,
    _from_log_space,
    _mse_and_all_log_gradient,
    _to_log_space,
)
from optimization_lane import extract_coefficients, prepare_optimization_context, unpack_lane_parameters
from scipy.optimize import minimize
import numexpr as ne


@lru_cache(maxsize=256)
def _compiled_programs(expression: str, derivatives: tuple[str, ...]):
    """Compile exactly the strings previously supplied to ``ne.evaluate``."""
    return ne.NumExpr(expression), tuple(ne.NumExpr(item) for item in derivatives)


def _mse_and_all_log_gradient_precompiled(log_values: np.ndarray, payload: Mapping[str, Any]):
    """Numerically identical objective, avoiding repeated expression parsing."""
    raw_values = _from_log_space(log_values)
    names = list(payload["coefficient_names"])
    lanes = list(payload["block_lanes"])
    lane_parameters = unpack_lane_parameters(raw_values, lanes, names)
    target = np.asarray(payload["target_values"], dtype=float)
    valid = np.asarray(payload["valid"], dtype=bool)
    n_rows = len(target)
    prediction = np.zeros(n_rows, dtype=float)
    sensitivities = np.zeros((len(raw_values), n_rows), dtype=float)
    expression, derivatives = _compiled_programs(str(payload["expression"]), tuple(payload["derivative_expressions"]))
    try:
        for lane_index, lane in enumerate(lanes):
            context = {**payload["lane_contexts"][lane], **lane_parameters[lane]}
            weight = np.asarray(payload["weights"][lane], dtype=float)
            raw = _row_array(expression(*[context[name] for name in expression.input_names]), n_rows)
            if np.any(~np.isfinite(raw)):
                return _penalty(log_values)
            active = raw > 0.0
            prediction += weight * np.maximum(raw, 0.0)
            start = lane_index * len(names)
            for coefficient_index, derivative_program in enumerate(derivatives):
                derivative = _row_array(derivative_program(*[context[name] for name in derivative_program.input_names]), n_rows)
                relevant = active & (np.abs(weight) > 0.0)
                if np.any((~np.isfinite(derivative)) & relevant):
                    return _penalty(log_values)
                derivative = np.where(active, np.nan_to_num(derivative, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
                sensitivities[start + coefficient_index] = weight * derivative
    except Exception:
        return _penalty(log_values)
    if not np.any(valid):
        return _penalty(log_values)
    residual = prediction[valid] - target[valid]
    if residual.size == 0 or np.any(~np.isfinite(residual)):
        return _penalty(log_values)
    objective = float(np.mean(residual**2))
    raw_gradient = 2.0 * np.mean(sensitivities[:, valid] * residual[np.newaxis, :], axis=1)
    if not np.isfinite(objective) or np.any(~np.isfinite(raw_gradient)):
        return _penalty(log_values)
    return objective, np.asarray(raw_gradient, dtype=float) * raw_values


def _solve_one_restart(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Run exactly one of the original restart calls in an isolated worker."""
    ne.set_num_threads(1)
    payload = task["payload"]
    index = int(task["restart_index"])
    raw_bounds = [tuple(map(float, item)) for item in payload["bounds"]]
    log_bounds = [(float(np.log(lo)), float(np.log(hi))) for lo, hi in raw_bounds]
    started = time.perf_counter()
    objective = _mse_and_all_log_gradient_precompiled if bool(task.get("precompile", False)) else _mse_and_all_log_gradient
    result = minimize(
        objective,
        _to_log_space(np.asarray(payload["initials"][index], dtype=float)),
        args=(payload,), method="L-BFGS-B", jac=True, bounds=log_bounds,
        options={"maxiter": int(payload["maxiter"]), "maxfun": int(payload["maxfun"]), "disp": False},
    )
    raw_values = _from_log_space(result.x)
    r9_probe_pass, r9_probe_delays = _r9_probe_restart(raw_values, payload)
    return {
        "approach": str(payload["approach"]), "index": index,
        "parameterization": "all_positive_log",
        "start_kind": list(payload["restart_kinds"])[index], "values": raw_values,
        "success": bool(result.success), "status": int(result.status),
        "message": str(result.message), "objective": float(result.fun),
        "iterations": int(getattr(result, "nit", 0)),
        "function_evaluations": int(getattr(result, "nfev", 0)),
        "gradient_evaluations": int(getattr(result, "njev", 0)),
        "gradient_inf_norm_optimization_space": float(np.linalg.norm(np.asarray(getattr(result, "jac", []), dtype=float), ord=np.inf)),
        "r9_probe_pass": bool(r9_probe_pass), "r9_probe_delays_seconds": r9_probe_delays,
        "wall_seconds": float(time.perf_counter() - started),
    }


def fit_lane_parameters_restart_parallel(
    universal_expr: str, df, lanes: Sequence[str], lane_to_approach: Mapping[str, str],
    approach_targets: Mapping[str, Any], intersection_id: int, verbose: bool = False,
    prepared_context: Optional[Mapping[str, Any]] = None, rng: Optional[np.random.Generator] = None,
    n_restarts: int = 10, diagnostics: Optional[Dict[str, Any]] = None,
    coefficient_bounds_override: Optional[Mapping[str, Tuple[float, float]]] = None,
    *, warm_parameters: Optional[Mapping[str, Mapping[str, float]]] = None,
    parallel_workers: int = 8, maxiter: int = 200, maxfun: int = 20_000,
    executor: Optional[ProcessPoolExecutor] = None, r9_constrained_restart_selection: bool = False, precompile_numexpr: bool = False,
) -> Dict[str, Dict[str, float]]:
    """V18-equivalent fit with approaches 閼?restarts scheduled independently."""
    if min(n_restarts, parallel_workers, maxiter, maxfun) < 1:
        raise ValueError("restarts, workers, and budgets must be positive")
    coefficient_names = extract_coefficients(universal_expr)
    if not coefficient_names:
        return {str(lane): {} for lane in lanes}
    rng = rng or np.random.default_rng()
    prepared = prepared_context or prepare_optimization_context(df, list(lanes), dict(lane_to_approach), int(intersection_id))
    build_started = time.perf_counter()
    derivatives = _symbolic_derivative_strings(universal_expr, tuple(coefficient_names))
    coefficient_names, payloads, base_seed = generate_mixed_restart_payloads(
        universal_expr=universal_expr, lanes=list(lanes), approach_targets=approach_targets,
        prepared=prepared, rng=rng, n_restarts=int(n_restarts), maxiter=int(maxiter), maxfun=int(maxfun),
        coefficient_bounds_override=coefficient_bounds_override, warm_parameters=warm_parameters,
    )
    for payload in payloads:
        payload["derivative_expressions"] = derivatives
        payload["r9_constrained_restart_selection"] = bool(r9_constrained_restart_selection)
    symbolic_build_seconds = float(time.perf_counter() - build_started)
    tasks = [{"payload": payload, "restart_index": index, "precompile": bool(precompile_numexpr)} for payload in payloads for index in range(len(payload["initials"]))]
    started = time.perf_counter()
    owns_executor = executor is None
    active = executor or ProcessPoolExecutor(max_workers=min(int(parallel_workers), len(tasks)))
    try:
        results = [_solve_one_restart(task) for task in tasks] if len(tasks) == 1 else list(active.map(_solve_one_restart, tasks))
    finally:
        if owns_executor:
            active.shutdown(wait=True)
    outer_wall_seconds = float(time.perf_counter() - started)
    by_approach: Dict[str, list[Dict[str, Any]]] = {str(payload["approach"]): [] for payload in payloads}
    for result in results:
        by_approach[result["approach"]].append(result)
    fitted: Dict[str, Dict[str, float]] = {}
    records = []
    for payload in payloads:
        items = sorted(by_approach[str(payload["approach"])], key=lambda item: item["index"])
        feasible = [i for i, item in enumerate(items) if item["r9_probe_pass"] and np.isfinite(item["objective"])]
        selection = (feasible or list(range(len(items)))) if r9_constrained_restart_selection else list(range(len(items)))
        chosen_index = min(selection, key=lambda i: items[i]["objective"] if np.isfinite(items[i]["objective"]) else np.inf)
        chosen = items[chosen_index]
        fitted.update(unpack_lane_parameters(np.asarray(chosen["values"], dtype=float), list(payload["block_lanes"]), coefficient_names))
        records.append({"approach": str(payload["approach"]), "chosen_restart": int(chosen_index), "objective": float(chosen["objective"]), "function_evaluations": int(chosen["function_evaluations"]), "total_function_evaluations": int(sum(x["function_evaluations"] for x in items)), "total_wall_seconds": float(sum(x["wall_seconds"] for x in items)), "restarts": [{k:v for k,v in x.items() if k not in {"values", "approach"}} for x in items]})
    if diagnostics is not None:
        serial_work_seconds = float(sum(item["total_wall_seconds"] for item in records))
        diagnostics.update({"solver":"L-BFGS-B", "jacobian":"analytic_sympy_numexpr", "parameterization":"all_positive_log", "start_strategy_id":START_STRATEGY_ID, "n_restarts":int(n_restarts), "maxiter":int(maxiter), "maxfun":int(maxfun), "parallel_workers":min(int(parallel_workers),len(tasks)), "parallelization":"independent_approach_restart_tasks", "numexpr_precompiled":bool(precompile_numexpr), "parent_warm_start_available":warm_parameters is not None, "raw_restart_points_retained":True, "r9_constrained_restart_selection":bool(r9_constrained_restart_selection), "sobol_base_seed":base_seed, "objective_sum":float(sum(item["objective"] for item in records)), "outer_wall_seconds":outer_wall_seconds, "symbolic_build_seconds":symbolic_build_seconds, "serial_work_seconds":serial_work_seconds, "parallel_efficiency_ratio":serial_work_seconds / outer_wall_seconds if outer_wall_seconds else None, "total_function_evaluations":int(sum(item["total_function_evaluations"] for item in records)), "approaches":records, "experimental_only":True})
    if verbose:
        print(f"restart-parallel L-BFGS-B: tasks={len(tasks)}, wall={outer_wall_seconds:.2f}s", flush=True)
    return fitted
