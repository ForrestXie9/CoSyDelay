"""Predeclared coefficient-role policy for the conditioning ablation.

The policy is intentionally structural and data independent.  It never reads
Validation or Test data and it does not infer bounds from achieved accuracy.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

from expression_rules import coefficient_names, parameter_roles


Bound = Tuple[float, float]

DEFAULT_ROLE_BOUNDS: Dict[str, Bound] = {
    "scale": (0.001, 1000.0),
    "power_exponent": (0.05, 5.0),
    "exp_coefficient": (0.0001, 1.0),
}

# This profile changes only the two nonlinear roles.  The broad positive scale
# range is retained for the first experiment so that any gain is not obtained
# by silently excluding a previously admissible scale coefficient.  The power
# cap limits very sharp demand responses over the declared operating range;
# the exp cap allows a modestly stronger response while remaining finite there.
CONDITIONED_ROLE_BOUNDS: Dict[str, Bound] = {
    "scale": (0.001, 1000.0),
    "power_exponent": (0.1, 3.0),
    "exp_coefficient": (0.0001, 2.0),
}


def nonlinear_role_conflicts(expression: str) -> Dict[str, list[str]]:
    """Return coefficients used in more than one nonlinear parameter role."""
    roles = parameter_roles(expression)
    conflicts: Dict[str, list[str]] = {}
    for name in coefficient_names(expression):
        nonlinear = sorted(
            role
            for role in roles.get(name, set())
            if role in {"power_exponent", "exp_coefficient"}
        )
        if len(nonlinear) > 1:
            conflicts[name] = nonlinear
    return conflicts


def coefficient_bounds_for_expression(
    expression: str,
    profile: Mapping[str, Bound],
    *,
    reject_nonlinear_role_conflicts: bool = True,
) -> Dict[str, Bound]:
    """Map each coefficient to one predeclared role-specific bound."""
    conflicts = nonlinear_role_conflicts(expression)
    if reject_nonlinear_role_conflicts and conflicts:
        detail = ", ".join(
            f"{name}={'+'.join(roles)}" for name, roles in conflicts.items()
        )
        raise ValueError(f"mixed nonlinear coefficient roles are forbidden: {detail}")

    roles = parameter_roles(expression)
    result: Dict[str, Bound] = {}
    for name in coefficient_names(expression):
        assigned = roles.get(name, {"scale"})
        if "power_exponent" in assigned:
            role = "power_exponent"
        elif "exp_coefficient" in assigned:
            role = "exp_coefficient"
        else:
            role = "scale"
        lower, upper = profile[role]
        if not (0.0 < float(lower) < float(upper)):
            raise ValueError(f"invalid positive bounds for {role}: {(lower, upper)}")
        result[name] = (float(lower), float(upper))
    return result
