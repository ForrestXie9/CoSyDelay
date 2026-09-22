"""Shared structural rules for generated lane-delay expressions."""
from __future__ import annotations

import ast
import math
import re
from functools import lru_cache
from typing import Dict, List, Set, Tuple

import numpy as np
import sympy as sp


FEATURE_NAMES = ("flow_lane", "GR_phase", "Cycle_Time")
ALLOWED_FUNCTION_NAMES = ("exp", "log")
MAX_EXPRESSION_NODES = 90
MAX_EXPRESSION_CHARACTERS = 2000
MAX_PYTHON_AST_NODES = 256
# Match the final CoSyDelay search: no artificial a1-a8 cap; only the
# expression-complexity bound.
MAX_COEFFICIENTS = MAX_PYTHON_AST_NODES
MAX_ABS_NUMERIC_LITERAL = 1_000_000.0
PREFIT_FLOW_POINTS = (0.0, 0.01, 0.5, 1.0, 2.0)
PREFIT_GREEN_POINTS = (0.02, 0.1, 0.5, 0.95)


def coefficient_names(expr: str) -> List[str]:
    return sorted(set(re.findall(r"\ba\d+\b", expr)), key=lambda name: int(name[1:]))


@lru_cache(maxsize=4096)
def parse_symbolic_expression(expr: str) -> Tuple[sp.Expr, Dict[str, sp.Symbol]]:
    """Parse the documented expression grammar without evaluating Python text.

    LLM output is untrusted input.  ``sympy.sympify`` ultimately relies on
    Python evaluation and therefore must not be the first parser applied to a
    generated string.  This small AST translator accepts only numeric literals,
    declared symbols, arithmetic operators, and one-argument ``exp``/``log``.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("expression must be a non-empty string")
    if len(expr) > MAX_EXPRESSION_CHARACTERS:
        raise ValueError(
            f"expression is too long ({len(expr)} > {MAX_EXPRESSION_CHARACTERS})"
        )
    names = coefficient_names(expr)
    invalid_coefficients = [
        name for name in names if not 1 <= int(name[1:]) <= MAX_COEFFICIENTS
    ]
    if invalid_coefficients:
        raise ValueError(
            "unsupported coefficient identifiers: "
            + ", ".join(invalid_coefficients)
            + f"; allowed identifiers are a1 through a{MAX_COEFFICIENTS}"
        )
    symbols = {
        "flow_lane": sp.Symbol("flow_lane", nonnegative=True),
        "GR_phase": sp.Symbol("GR_phase", positive=True),
        "Cycle_Time": sp.Symbol("Cycle_Time", positive=True),
    }
    symbols.update({name: sp.Symbol(name, positive=True) for name in names})
    allowed_names = set(symbols)
    try:
        syntax = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression syntax: {exc.msg}") from exc
    node_count = sum(1 for _ in ast.walk(syntax))
    if node_count > MAX_PYTHON_AST_NODES:
        raise ValueError(
            f"expression syntax is too complex ({node_count} > {MAX_PYTHON_AST_NODES})"
        )

    binary_operators = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.Pow: lambda left, right: left ** right,
    }

    def translate(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Expression):
            return translate(node.body)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("only real numeric literals are allowed")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("numeric literals must be finite")
            if abs(numeric) > MAX_ABS_NUMERIC_LITERAL:
                raise ValueError(
                    "numeric literal exceeds the documented safety bound"
                )
            return sp.Integer(value) if isinstance(value, int) else sp.Float(value)
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(f"unsupported symbol: {node.id}")
            return symbols[node.id]
        if isinstance(node, ast.BinOp):
            operation = binary_operators.get(type(node.op))
            if operation is None:
                raise ValueError(
                    f"unsupported binary operator: {type(node.op).__name__}"
                )
            return operation(translate(node.left), translate(node.right))
        if isinstance(node, ast.UnaryOp):
            operand = translate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError(
                f"unsupported unary operator: {type(node.op).__name__}"
            )
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in ALLOWED_FUNCTION_NAMES
            ):
                raise ValueError("only exp(...) and log(...) calls are allowed")
            if len(node.args) != 1 or node.keywords:
                raise ValueError(f"{node.func.id} requires exactly one argument")
            argument = translate(node.args[0])
            return sp.exp(argument) if node.func.id == "exp" else sp.log(argument)
        raise ValueError(f"unsupported syntax node: {type(node).__name__}")

    return translate(syntax), symbols


def parameter_roles(expr: str) -> Dict[str, Set[str]]:
    """Classify fitted coefficients by their structural role in the AST."""
    parsed, symbols = parse_symbolic_expression(expr)
    roles = {name: {"scale"} for name in coefficient_names(expr)}

    for power in parsed.atoms(sp.Pow):
        for name in roles:
            if symbols[name] in power.exp.free_symbols:
                roles[name].add("power_exponent")

    for exp_node in parsed.atoms(sp.exp):
        for name in roles:
            if symbols[name] in exp_node.args[0].free_symbols:
                roles[name].add("exp_coefficient")

    return roles


def canonical_expression(expr: str) -> str:
    """Return a stable structural key for algebraically reordered duplicates."""
    parsed, _ = parse_symbolic_expression(expr)
    return sp.srepr(parsed)


def structural_family_key(expr: str) -> str:
    """Return an operator/feature skeleton that ignores coefficient labels.

    ``canonical_expression`` distinguishes fitted parameter names.  That is
    appropriate for exact duplicate rejection, but two candidates can still
    consume population slots with the same functional skeleton after merely
    permuting ``a1``, ``a2``, ... .  This coarser key preserves the canonical
    SymPy tree, variables, constants, and operators while replacing every
    fitted-coefficient symbol by the same explicit token.
    """
    canonical = canonical_expression(expr)
    return re.sub(
        r"Symbol\('a\d+', positive=True\)",
        "FittedCoefficient()",
        canonical,
    )


def _has_interior_denominator_singularity(parsed: sp.Expr, symbols: Dict[str, sp.Symbol]) -> bool:
    flow = symbols["flow_lane"]
    green = symbols["GR_phase"]
    cycle = symbols["Cycle_Time"]
    coefficients = [symbols[name] for name in coefficient_names(str(parsed))]
    representatives = {
        cycle: 120.0,
        **{symbol: 1.0 for symbol in coefficients},
    }
    flow_values = np.linspace(0.001, 2.0, 65)
    green_values = np.linspace(0.02, 0.95, 65)
    flow_grid, green_grid = np.meshgrid(flow_values, green_values, indexing="ij")

    for power in parsed.atoms(sp.Pow):
        if not power.exp.could_extract_minus_sign():
            continue
        if not ({green, flow} & power.base.free_symbols):
            continue
        try:
            base = power.base.subs(representatives)
            function = sp.lambdify((flow, green), base, modules="numpy")
            with np.errstate(all="ignore"):
                values = np.asarray(function(flow_grid, green_grid), dtype=float)
            values = np.broadcast_to(values, flow_grid.shape)
        except Exception:
            continue
        finite = values[np.isfinite(values)]
        if finite.size != values.size or finite.size == 0:
            return True
        if np.any(np.abs(finite) <= 1e-10):
            return True
        if float(np.min(finite)) < 0.0 < float(np.max(finite)):
            return True
    return False


def _has_variable_power_domain_risk(parsed: sp.Expr, symbols: Dict[str, sp.Symbol]) -> bool:
    green = symbols["GR_phase"]
    flow = symbols["flow_lane"]
    coefficient_symbols = {symbols[name] for name in coefficient_names(str(parsed))}
    representatives = {symbol: 1.0 for symbol in coefficient_symbols}

    for power in parsed.atoms(sp.Pow):
        if not (power.exp.free_symbols & coefficient_symbols):
            continue
        base = power.base.subs(representatives)
        # A fitted real exponent is unsafe if its base is negative anywhere in
        # the universal traffic domain. Check deterministic boundary/interior points.
        for flow_value in PREFIT_FLOW_POINTS:
            for green_value in PREFIT_GREEN_POINTS:
                try:
                    value = float(base.subs({flow: flow_value, green: green_value}))
                except (TypeError, ValueError):
                    continue
                if value < 0.0:
                    return True
    return False


def _has_valid_symbolic_low_demand_limit(
    parsed: sp.Expr, symbols: Dict[str, sp.Symbol]
) -> bool:
    """Check finalized R5 before fitting under positive coefficient assumptions."""
    flow = symbols["flow_lane"]
    green = symbols["GR_phase"]
    cycle = symbols["Cycle_Time"]
    green_odds = sp.Symbol("_green_odds", nonnegative=True, finite=True)
    positive_cycle = sp.Symbol("_positive_cycle", positive=True, finite=True)

    def on_declared_domain(value: sp.Expr) -> sp.Expr:
        return sp.simplify(
            value.subs({green: 1 / (1 + green_odds), cycle: positive_cycle})
        )

    def established(value: sp.Expr) -> bool:
        return bool(
            not isinstance(value, sp.Limit)
            and not value.has(sp.Limit)
            and getattr(value, "is_finite", None) is True
            and getattr(value, "is_nonnegative", None) is True
        )

    try:
        # Fast path for right-continuous candidates such as x**a with a>0.
        # The fitted verifier still recomputes the analytic limit numerically.
        direct_value = on_declared_domain(parsed.subs(flow, 0))
        if established(direct_value):
            return True
        return established(on_declared_domain(sp.limit(parsed, flow, 0, dir="+")))
    except Exception:
        return False


def validate_candidate_legality(expr: str) -> Tuple[bool, str]:
    """Check parser safety and bounded grammar without enforcing physics.

    This validator is used by controlled prompt/verifier ablations. Physical
    principles must remain available to the external scorer instead of being
    silently enforced as a generation-time filter in every treatment arm.
    """
    errors: List[str] = []
    names = coefficient_names(expr)
    if not isinstance(expr, str) or not expr.strip():
        return False, "empty expression"
    if not names:
        errors.append("missing fitted coefficients")
    if len(names) > MAX_COEFFICIENTS:
        errors.append(f"too many coefficients ({len(names)} > {MAX_COEFFICIENTS})")
    if expr.count("(") != expr.count(")"):
        errors.append("unbalanced parentheses")

    try:
        parsed, symbols = parse_symbolic_expression(expr)
        unexpected = parsed.free_symbols - set(symbols.values())
        if unexpected:
            errors.append("unsupported symbols: " + ", ".join(sorted(map(str, unexpected))))
        if sum(1 for _ in sp.preorder_traversal(parsed)) > MAX_EXPRESSION_NODES:
            errors.append("expression is too complex")
        if _has_interior_denominator_singularity(parsed, symbols):
            errors.append("denominator can become zero within 0 < GR_phase <= 1")
        if _has_variable_power_domain_risk(parsed, symbols):
            errors.append("a fitted exponent has a base that can become negative")

        flow = symbols["flow_lane"]
        green = symbols["GR_phase"]
        for log_node in parsed.atoms(sp.log):
            argument = log_node.args[0]
            representative = argument.subs({symbols[name]: 1.0 for name in names})
            for flow_value in PREFIT_FLOW_POINTS:
                for green_value in PREFIT_GREEN_POINTS:
                    try:
                        value = float(
                            representative.subs({flow: flow_value, green: green_value})
                        )
                    except (TypeError, ValueError):
                        continue
                    if value <= 0.0:
                        errors.append("a logarithm argument can become non-positive")
                        break
                else:
                    continue
                break
    except Exception as exc:
        errors.append(f"expression parsing failed: {exc}")

    if errors:
        return False, "; ".join(dict.fromkeys(errors))
    return True, "Expression passed grammar and numerical-legality checks"


def validate_candidate_expression(expr: str) -> Tuple[bool, str]:
    """Fast, coefficient-independent checks run before any parameter fitting."""
    errors: List[str] = []
    names = coefficient_names(expr)

    if "flow_lane" not in expr:
        errors.append("missing required feature flow_lane")
    if "GR_phase" not in expr:
        errors.append("missing required feature GR_phase")
    if "Cycle_Time" not in expr:
        errors.append("missing required feature Cycle_Time")
    if not names:
        errors.append("missing fitted coefficients")
    if len(names) > MAX_COEFFICIENTS:
        errors.append(f"too many coefficients ({len(names)} > {MAX_COEFFICIENTS})")
    if expr.count("(") != expr.count(")"):
        errors.append("unbalanced parentheses")

    try:
        parsed, symbols = parse_symbolic_expression(expr)
        allowed = set(symbols.values())
        unexpected = parsed.free_symbols - allowed
        if unexpected:
            errors.append("unsupported symbols: " + ", ".join(sorted(map(str, unexpected))))
        if sum(1 for _ in sp.preorder_traversal(parsed)) > MAX_EXPRESSION_NODES:
            errors.append("expression is too complex")

        flow = symbols["flow_lane"]
        green = symbols["GR_phase"]
        cycle = symbols["Cycle_Time"]
        # With all other features dimensionless, a delay has units of seconds
        # exactly when scaling Cycle_Time by k scales the full expression by k.
        try:
            time_scale = sp.Symbol("_time_scale", positive=True)
            scaling_residual = sp.simplify(
                parsed.subs(cycle, time_scale * cycle) - time_scale * parsed
            )
            if scaling_residual != 0:
                errors.append("expression is not homogeneous with time dimension seconds")
        except Exception:
            errors.append("could not verify time-unit consistency")

        if _has_interior_denominator_singularity(parsed, symbols):
            errors.append("denominator can become zero within 0 < GR_phase <= 1")
        if _has_variable_power_domain_risk(parsed, symbols):
            errors.append("a fitted exponent has a base that can become negative")

        # Reject only direction violations that SymPy can prove immediately for
        # all positive coefficients and inputs.  Avoid factor/simplify here:
        # ambiguous signs belong to the bounded post-fit hybrid verifier.
        try:
            flow_derivative = sp.diff(parsed, flow)
            if flow_derivative != 0 and flow_derivative.is_nonpositive is True:
                errors.append("delay is structurally non-increasing with flow_lane")
            green_derivative = sp.diff(parsed, green)
            if green_derivative != 0 and green_derivative.is_nonnegative is True:
                errors.append("delay is structurally non-decreasing with GR_phase")
        except Exception:
            pass

        for log_node in parsed.atoms(sp.log):
            argument = log_node.args[0]
            representative = argument.subs({symbols[name]: 1.0 for name in names})
            for flow_value in PREFIT_FLOW_POINTS:
                for green_value in PREFIT_GREEN_POINTS:
                    try:
                        value = float(representative.subs({flow: flow_value, symbols["GR_phase"]: green_value}))
                    except (TypeError, ValueError):
                        continue
                    if value <= 0.0:
                        errors.append("a logarithm argument can become non-positive")
                        break
                else:
                    continue
                break
    except Exception as exc:
        errors.append(f"expression parsing failed: {exc}")

    if errors:
        return False, "; ".join(dict.fromkeys(errors))
    return True, "Expression passed pre-fit structural validation"
