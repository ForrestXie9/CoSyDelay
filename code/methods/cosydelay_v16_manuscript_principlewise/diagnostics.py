"""V16 diagnostics that cannot affect fitting or evolutionary selection."""

from __future__ import annotations

import ast
from typing import Any, Mapping

import numpy as np
import sympy as sp

from expression_rules import (
    coefficient_names,
    parameter_roles,
    parse_symbolic_expression,
)


def _coefficient_role(
    name: str, roles: Mapping[str, set[str]]
) -> str:
    assigned = roles.get(name, {"scale"})
    if "power_exponent" in assigned:
        return "power_exponent"
    if "exp_coefficient" in assigned:
        return "exp_coefficient"
    return "scale"


def audit_parameter_quality_log_coordinates(
    expression: str,
    parameters: Mapping[str, Mapping[str, float]],
    bounds: Mapping[str, tuple[float, float]],
    *,
    near_boundary_fraction: float = 1e-4,
    exact_boundary_fraction: float = 1e-10,
) -> dict[str, Any]:
    """Audit fitted bounds in the coordinates actually used by V16.

    V16 optimizes every positive coefficient in log space.  Measuring distance
    over the raw linear interval can therefore label ordinary small positive
    values as boundary-saturated when a bound spans several orders of
    magnitude.  This audit uses the normalized log interval and is diagnostic
    only; it is called after the winner has already been selected.
    """
    if not 0.0 <= exact_boundary_fraction <= near_boundary_fraction < 0.5:
        raise ValueError("invalid boundary diagnostic fractions")
    parsed, _ = parse_symbolic_expression(expression)
    roles = parameter_roles(expression)
    names = coefficient_names(expression)
    records: list[dict[str, Any]] = []
    for component, values in parameters.items():
        for name in names:
            role = _coefficient_role(name, roles)
            lower, upper = map(float, bounds[role])
            value = float(values[name])
            if not (0.0 < lower < upper):
                raise ValueError(f"invalid positive bounds for {role}: {(lower, upper)}")
            if np.isfinite(value) and value > 0.0:
                log_lower = float(np.log(lower))
                log_upper = float(np.log(upper))
                log_position = float(
                    (np.log(value) - log_lower) / (log_upper - log_lower)
                )
                lower_distance = log_position
                upper_distance = 1.0 - log_position
            else:
                log_position = float("nan")
                lower_distance = float("inf")
                upper_distance = float("inf")
            records.append(
                {
                    "component": str(component),
                    "coefficient": name,
                    "role": role,
                    "value": value,
                    "lower": lower,
                    "upper": upper,
                    "normalized_log_position": log_position,
                    "near_lower_bound": bool(
                        lower_distance <= near_boundary_fraction
                    ),
                    "near_upper_bound": bool(
                        upper_distance <= near_boundary_fraction
                    ),
                    "at_lower_bound": bool(
                        lower_distance <= exact_boundary_fraction
                    ),
                    "at_upper_bound": bool(
                        upper_distance <= exact_boundary_fraction
                    ),
                }
            )

    near = [
        item
        for item in records
        if item["near_lower_bound"] or item["near_upper_bound"]
    ]
    exact = [
        item
        for item in records
        if item["at_lower_bound"] or item["at_upper_bound"]
    ]
    ast_nodes = sum(1 for _ in ast.walk(ast.parse(expression, mode="eval")))
    count = len(records)
    return {
        "diagnostic_only_not_used_for_selection": True,
        "coordinate_system": "normalized_natural_log_bound_interval",
        "finite_coefficients": bool(
            records and all(np.isfinite(item["value"]) for item in records)
        ),
        "strictly_positive_coefficients": bool(
            records and all(item["value"] > 0.0 for item in records)
        ),
        "coefficient_names": names,
        "coefficients_per_component": len(names),
        "fitted_coefficient_values": count,
        "near_boundary_fraction_of_log_span": float(near_boundary_fraction),
        "exact_boundary_fraction_of_log_span": float(exact_boundary_fraction),
        "near_boundary_values": len(near),
        "near_boundary_rate": float(len(near) / count) if count else None,
        "exact_boundary_values": len(exact),
        "exact_boundary_rate": float(len(exact) / count) if count else None,
        "boundary_records": near,
        "expression_characters": len(expression),
        "python_ast_nodes": int(ast_nodes),
        "sympy_operations": int(sp.count_ops(parsed, visual=False)),
        "parameter_identifiability_proven": False,
        "interpretation": (
            "Boundary saturation is diagnostic only and is measured in the "
            "same log coordinates used by the optimizer. It is neither a "
            "physical-rule failure nor evidence of parameter identifiability."
        ),
    }
