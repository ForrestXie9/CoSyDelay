"""V21 denominator gate: reject demonstrated interior zeros, allow unknowns."""
from __future__ import annotations

from typing import Dict

import numpy as np
import sympy as sp

from expression_rules import coefficient_names


def has_proven_interior_denominator_zero(
    parsed: sp.Expr, symbols: Dict[str, sp.Symbol]
) -> bool:
    """Return true only with direct or constructive evidence of an interior zero.

    A sign change for one admissible positive coefficient assignment proves a
    continuous denominator has an interior root. Evaluation failure, overflow,
    or inability to establish a sign is classified as unknown and allowed to
    proceed to bounded fitting/evaluation.
    """
    flow = symbols["flow_lane"]
    green = symbols["GR_phase"]
    cycle = symbols["Cycle_Time"]
    coefficients = [symbols[name] for name in coefficient_names(str(parsed))]
    representative_sets = [
        {cycle: 120.0, **{symbol: scale for symbol in coefficients}}
        for scale in (0.1, 0.5, 1.0, 2.0, 10.0)
    ]
    flow_values = np.linspace(0.001, 2.0, 65)
    green_values = np.linspace(0.02, 0.95, 65)
    flow_grid, green_grid = np.meshgrid(flow_values, green_values, indexing="ij")

    for power in parsed.atoms(sp.Pow):
        if not power.exp.could_extract_minus_sign():
            continue
        base = power.base
        if not ({green, flow} & base.free_symbols):
            continue
        if base.is_zero is True:
            return True
        if base.is_positive is True or base.is_negative is True:
            continue
        for representatives in representative_sets:
            try:
                concrete = base.subs(representatives)
                function = sp.lambdify((flow, green), concrete, modules="numpy")
                with np.errstate(all="ignore"):
                    values = np.asarray(function(flow_grid, green_grid), dtype=float)
                values = np.broadcast_to(values, flow_grid.shape)
            except Exception:
                continue
            finite = values[np.isfinite(values)]
            # Unknown numerical behavior is not proof of a zero.
            if finite.size == 0 or finite.size != values.size:
                continue
            if np.any(np.abs(finite) <= 1e-10):
                return True
            # A continuous real-valued base with sampled values on both sides
            # of zero has a root between them (intermediate value theorem).
            if float(np.min(finite)) < 0.0 < float(np.max(finite)):
                return True
    return False
