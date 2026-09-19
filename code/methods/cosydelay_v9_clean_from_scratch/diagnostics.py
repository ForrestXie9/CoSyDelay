"""Non-selective interpretability and compute diagnostics for formal results."""

from __future__ import annotations

import ast
from typing import Any, Mapping, Sequence

import numpy as np
import sympy as sp

from expression_rules import coefficient_names, parameter_roles, parse_symbolic_expression


def _coefficient_bound(
    name: str,
    roles: Mapping[str, set[str]],
    bounds: Mapping[str, tuple[float, float]],
) -> tuple[str, tuple[float, float]]:
    assigned = roles.get(name, {"scale"})
    if "power_exponent" in assigned:
        role = "power_exponent"
    elif "exp_coefficient" in assigned:
        role = "exp_coefficient"
    else:
        role = "scale"
    return role, tuple(bounds[role])


def audit_parameter_quality(
    expression: str,
    parameters: Mapping[str, Mapping[str, float]],
    bounds: Mapping[str, tuple[float, float]],
    *,
    near_boundary_fraction: float = 1e-4,
) -> dict[str, Any]:
    """Report boundary saturation and complexity; never change selection."""
    parsed, _ = parse_symbolic_expression(expression)
    roles = parameter_roles(expression)
    names = coefficient_names(expression)
    records = []
    for lane, values in parameters.items():
        for name in names:
            value = float(values[name])
            role, (lower, upper) = _coefficient_bound(name, roles, bounds)
            width = max(float(upper - lower), np.finfo(float).eps)
            lower_distance = (value - lower) / width
            upper_distance = (upper - value) / width
            records.append(
                {
                    "lane": str(lane),
                    "coefficient": name,
                    "role": role,
                    "value": value,
                    "lower": float(lower),
                    "upper": float(upper),
                    "near_lower_bound": bool(lower_distance <= near_boundary_fraction),
                    "near_upper_bound": bool(upper_distance <= near_boundary_fraction),
                }
            )
    boundary_hits = sum(
        item["near_lower_bound"] or item["near_upper_bound"] for item in records
    )
    ast_nodes = sum(1 for _ in ast.walk(ast.parse(expression, mode="eval")))
    return {
        "diagnostic_only_not_used_for_selection": True,
        "finite_coefficients": bool(
            records and all(np.isfinite(item["value"]) for item in records)
        ),
        "strictly_positive_coefficients": bool(
            records and all(item["value"] > 0.0 for item in records)
        ),
        "coefficient_names": names,
        "coefficients_per_lane": len(names),
        "fitted_coefficient_values": len(records),
        "near_boundary_fraction_of_span": float(near_boundary_fraction),
        "near_boundary_values": int(boundary_hits),
        "near_boundary_rate": (
            float(boundary_hits / len(records)) if records else None
        ),
        "boundary_records": [
            item
            for item in records
            if item["near_lower_bound"] or item["near_upper_bound"]
        ],
        "expression_characters": len(expression),
        "python_ast_nodes": int(ast_nodes),
        "sympy_operations": int(sp.count_ops(parsed, visual=False)),
        "parameter_identifiability_proven": False,
        "interpretation": (
            "Boundary saturation is an interpretability/optimization diagnostic, "
            "not a physical-rule failure. Identifiability requires a separate "
            "sensitivity or profile-likelihood analysis."
        ),
    }


def summarize_compute_efficiency(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fits = []
    for item in history:
        if item.get("event") != "evaluated":
            continue
        details = item.get("evaluation_details")
        if not isinstance(details, Mapping):
            continue
        fit = details.get("fit")
        fits.append((details, fit if isinstance(fit, Mapping) else {}))
    fit_walls = [float(details.get("fit_wall_seconds", 0.0)) for details, _ in fits]
    physics_walls = [
        float(details.get("physical_wall_seconds", 0.0)) for details, _ in fits
    ]
    function_evaluations = [
        int(fit.get("total_function_evaluations", 0)) for _, fit in fits
    ]
    gradient_evaluations = [
        int(fit.get("total_gradient_evaluations", 0)) for _, fit in fits
    ]
    return {
        "evaluated_candidates_with_timing": len(fits),
        "fit_wall_seconds_sum": float(sum(fit_walls)),
        "fit_wall_seconds_mean": float(np.mean(fit_walls)) if fit_walls else None,
        "physics_wall_seconds_sum": float(sum(physics_walls)),
        "physics_wall_seconds_mean": (
            float(np.mean(physics_walls)) if physics_walls else None
        ),
        "optimizer_function_evaluations_sum": int(sum(function_evaluations)),
        "optimizer_gradient_evaluations_sum": int(sum(gradient_evaluations)),
        "start_strategy_ids": sorted(
            {
                str(fit["start_strategy_id"])
                for _, fit in fits
                if fit.get("start_strategy_id")
            }
        ),
        "formal_method_adapters": sorted(
            {
                str(fit["formal_method_adapter"])
                for _, fit in fits
                if fit.get("formal_method_adapter")
            }
        ),
    }
