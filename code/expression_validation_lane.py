# expression_validation_lane.py
"""
Validation for universal lane-level expressions
Maintains the same validation logic as the original approach-level validation
"""
import re
import sympy as sp
import time
import atexit
from dataclasses import asdict, dataclass, field
from typing import Any, List, Tuple, Dict, Optional
import numpy as np
from scipy.stats import qmc
import multiprocessing
import queue
from sympy import lambdify
from functools import lru_cache
from expression_rules import coefficient_names, parse_symbolic_expression


PHYSICAL_RULE_SCHEMA_ID = "operational_domain_binary_hybrid_2026_07_14_v12"
SYMBOLIC_LIMIT_TIMEOUT_SECONDS = 8.0
SYMBOLIC_LIMIT_PROOF_TIMEOUT_SECONDS = 2.0
OPERATIONAL_MAX_DELAY_SECONDS = 1e6
# Post-fit verifier bounds from traffic-engineering domain knowledge only.
TRAFFIC_FLOW_DOMAIN = (0.001, 1.2)
TRAFFIC_GREEN_DOMAIN = (0.04, 0.85)
TRAFFIC_CYCLE_DOMAIN_SECONDS = (60.0, 180.0)
ZERO_GREEN_LOW_PROBE_GREEN = 1e-6
# R9 absolute severity threshold at GR=1e-6 (probe flow=1, cycle=120).
# Singular forms like C*q/GR may yield O(1e4) rather than O(1e8) at the probe,
# while nonsingular offsets like 1/(GR+a2) stay O(1e2--1e3).
ZERO_GREEN_MIN_DELAY_SECONDS = 1e4

# This order is part of the serialized experiment protocol.  Operational-domain
# rules R1--R4 and R6--R7 are checked on the declared grid; endpoint rules
# R8--R9 add the Table-I zero-flow identity and zero-green severity check.
# Former R5 (domain magnitude cap |delay|<=1e6) is intentionally omitted:
# non-finiteness is still caught by R6, while absolute blow-up is left to R9.
PHYSICAL_RULE_NAMES = (
    "R1_required_variables",
    "R2_nondecreasing_flow",
    "R3_nonincreasing_green",
    "R4_time_dimension",
    "R6_nonnegative_delay",
    "R7_operational_responsiveness",
    "R8_zero_flow_boundary",
    "R9_zero_green_limit",
)

# Historical artifacts used the names below.  The mapping is provided only so
# readers can identify columns in those artifacts.  In particular, the old R2
# tested the stronger-but-different identity f(0, g, C) == 0, so its result must
# never be relabeled as a formal R5 result.  Formal postprocessing must require
# PHYSICAL_RULE_SCHEMA_ID and regenerate every rule vector.
LEGACY_PHYSICAL_RULE_SCHEMA_ID = "pre_table_i_zero_flow_identity_v0"
LEGACY_PHYSICAL_RULE_NAMES = (
    "R1_required_variables",
    "R2_zero_flow_boundary",
    "R3_time_dimension",
    "R4_nonnegative_delay",
    "R5_nondecreasing_flow",
    "R6_nonincreasing_green",
    "R7_zero_green_limit",
)
LEGACY_RULE_NAME_TO_CURRENT_ID = {
    "R1_required_variables": "R1_required_variables",
    "R2_zero_flow_boundary": "R8_zero_flow_boundary",
    "R3_time_dimension": "R4_time_dimension",
    "R4_nonnegative_delay": "R6_nonnegative_delay",
    "R5_nondecreasing_flow": "R2_nondecreasing_flow",
    "R6_nonincreasing_green": "R3_nonincreasing_green",
    "R7_zero_green_limit": "R9_zero_green_limit",
}


@dataclass(frozen=True)
class PhysicalVerifierConfig:
    """Pre-registered domain and tolerances for the eight-rule verifier."""

    n_grid_samples: int = 128
    lhs_seed: int = 42
    flow_domain: Tuple[float, float] = TRAFFIC_FLOW_DOMAIN
    green_domain: Tuple[float, float] = TRAFFIC_GREEN_DOMAIN
    cycle_domain_seconds: Tuple[float, float] = TRAFFIC_CYCLE_DOMAIN_SECONDS
    zero_tolerance: float = 1e-6
    derivative_tolerance: float = 1e-6
    symbolic_derivative_timeout_seconds: float = 0.25
    finite_difference_fraction: float = 1e-4
    response_tolerance_seconds: float = 1e-6
    max_delay_seconds: float = OPERATIONAL_MAX_DELAY_SECONDS
    symbolic_limit_timeout_seconds: float = SYMBOLIC_LIMIT_TIMEOUT_SECONDS
    zero_green_low_probe_green: float = ZERO_GREEN_LOW_PROBE_GREEN
    zero_green_min_delay_seconds: float = ZERO_GREEN_MIN_DELAY_SECONDS
    zero_green_numerical_probe_flow: float = 1.0
    zero_green_numerical_probe_cycle: float = 120.0
    rule_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    def __post_init__(self):
        if len(self.rule_weights) != len(PHYSICAL_RULE_NAMES):
            raise ValueError(
                f"rule_weights must contain exactly {len(PHYSICAL_RULE_NAMES)} values"
            )
        if any(weight < 0 for weight in self.rule_weights):
            raise ValueError("rule weights must be non-negative")
        if sum(self.rule_weights) <= 0:
            raise ValueError("at least one rule weight must be positive")
        if self.finite_difference_fraction <= 0:
            raise ValueError("finite_difference_fraction must be positive")
        if self.symbolic_derivative_timeout_seconds <= 0:
            raise ValueError("symbolic_derivative_timeout_seconds must be positive")
        if self.response_tolerance_seconds < 0:
            raise ValueError("response_tolerance_seconds must be non-negative")
        if self.max_delay_seconds <= 0:
            raise ValueError("max_delay_seconds must be positive")
        if self.symbolic_limit_timeout_seconds <= 0:
            raise ValueError("symbolic_limit_timeout_seconds must be positive")
        if self.zero_green_low_probe_green <= 0:
            raise ValueError("zero_green_low_probe_green must be positive")
        if self.zero_green_min_delay_seconds <= 0:
            raise ValueError("zero_green_min_delay_seconds must be positive")

@dataclass
class PhysicalScoreResult:
    """Auditable eight-rule score, including lane-level pass fractions."""

    score: float
    joint_pass: bool
    rule_scores: Dict[str, float]
    lane_rule_scores: Dict[str, Dict[str, float]]
    lane_errors: Dict[str, List[str]] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    config: PhysicalVerifierConfig = field(default_factory=PhysicalVerifierConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_schema_id": PHYSICAL_RULE_SCHEMA_ID,
            "rule_order": list(PHYSICAL_RULE_NAMES),
            "score": float(self.score),
            "joint_pass": bool(self.joint_pass),
            "rule_scores": {key: float(value) for key, value in self.rule_scores.items()},
            "lane_rule_scores": {
                lane: {key: float(value) for key, value in scores.items()}
                for lane, scores in self.lane_rule_scores.items()
            },
            "lane_errors": self.lane_errors,
            "diagnostics": self.diagnostics,
            "config": asdict(self.config),
        }


RULE_MATHEMATICAL_SUMMARY = {
    "R1_required_variables": (
        "Required variables: the expression must use flow_lane, GR_phase, "
        "and Cycle_Time."
    ),
    "R2_nondecreasing_flow": (
        "Flow monotonicity: at fixed GR_phase and Cycle_Time, y is "
        "nondecreasing in flow_lane."
    ),
    "R3_nonincreasing_green": (
        "Green monotonicity: at fixed flow_lane and Cycle_Time, y is "
        "nonincreasing in GR_phase."
    ),
    "R4_time_dimension": (
        "Time dimension: every additive term reduces to Cycle_Time times a "
        "dimensionless function."
    ),
    "R6_nonnegative_delay": (
        "Non-negativity and finiteness: y is finite, real, and >= 0 over the "
        "operational domain."
    ),
    "R7_operational_responsiveness": (
        "Non-degenerate response: y genuinely depends on both flow_lane and "
        "GR_phase (not constant in either variable when the other is fixed)."
    ),
    "R8_zero_flow_boundary": (
        "Zero-flow identity: y(0, GR_phase, Cycle_Time) = 0."
    ),
    "R9_zero_green_limit": (
        "Zero-green singularity: for fixed flow_lane > 0 and Cycle_Time > 0, "
        "lim_{GR_phase -> 0+} y = +infinity."
    ),
}


def format_physical_validation_feedback(
    physical_result: PhysicalScoreResult,
    *,
    prediction_reason: Optional[str] = None,
) -> str:
    """Build prompt-facing physical feedback from failed joint rules only."""
    if physical_result.joint_pass and not prediction_reason:
        return "All eight physical principles passed."

    lines: List[str] = []
    if prediction_reason:
        lines.append(f"Observed-data evaluation issue: {prediction_reason}")

    failed_rules = [
        name
        for name in PHYSICAL_RULE_NAMES
        if physical_result.rule_scores.get(name, 0.0) < 1.0 - 1e-12
    ]
    if failed_rules:
        lines.append("Failed joint rules:")
        for name in failed_rules:
            lines.append(f"- {RULE_MATHEMATICAL_SUMMARY.get(name, name)}")

    if not lines:
        return "One or more physical rules did not fully pass."

    return "\n".join(lines)


def safe_exp(x):
    """Prevent exp overflow while preserving the original safe-evaluation behavior."""
    x_clipped = np.clip(x, -100, 100)
    with np.errstate(all='ignore'):
        return np.exp(x_clipped)


def safe_power(base, exponent):
    """Prevent power overflow while preserving the original safe-evaluation behavior."""
    base = np.asarray(base, dtype=float)
    exponent = np.asarray(exponent, dtype=float)
    exponent = np.clip(exponent, -50, 50)
    base = np.maximum(np.abs(base), 1e-10)

    with np.errstate(all='ignore'):
        log_result = exponent * np.log(base)
        log_result = np.clip(log_result, -100, 100)
        result = np.exp(log_result)

    return np.clip(result, 1e-100, 1e100)


def safe_log(x):
    """Prevent invalid log values while preserving the original safe-evaluation behavior."""
    x_safe = np.maximum(x, 1e-300)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(x_safe)


def _symbolic_limit_worker(expression, variable, point, direction, result_queue):
    """Evaluate one SymPy limit in an isolated, terminable process."""
    try:
        result_queue.put(
            (True, sp.limit(expression, variable, point, dir=direction), None)
        )
    except Exception as exc:
        result_queue.put((False, None, f"{type(exc).__name__}: {exc}"))


def _symbolic_limit_with_timeout(
    expression: sp.Expr,
    variable: sp.Symbol,
    point,
    *,
    direction: str = "+",
    timeout: float = SYMBOLIC_LIMIT_TIMEOUT_SECONDS,
) -> sp.Expr:
    """Compute a symbolic limit without allowing SymPy to stall a run.

    Green-ratio exponentials such as ``exp(a / GR_phase)`` can make SymPy's
    heuristic limit engine consume CPU indefinitely.  A process boundary is
    required here because a Python thread timeout cannot stop symbolic code.
    """
    if timeout <= 0:
        raise ValueError("symbolic limit timeout must be positive")
    if multiprocessing.current_process().daemon:
        raise TimeoutError(
            "isolated symbolic limit evaluation is unavailable in a daemon process"
        )

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_symbolic_limit_worker,
        args=(expression, variable, point, direction, result_queue),
    )
    process.daemon = True
    process.start()
    process.join(timeout)

    try:
        if process.is_alive():
            process.terminate()
            process.join()
            raise TimeoutError(
                f"symbolic limit exceeded {timeout:.1f}s"
            )
        try:
            success, result, error = result_queue.get(timeout=0.5)
        except queue.Empty as exc:
            raise RuntimeError("symbolic limit worker returned no result") from exc
        if not success:
            raise RuntimeError(error)
        return result
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        result_queue.close()
        result_queue.join_thread()


_DERIVATIVE_WORKER_PROCESS = None
_DERIVATIVE_WORKER_REQUEST_QUEUE = None
_DERIVATIVE_WORKER_RESULT_QUEUE = None
_DERIVATIVE_WORKER_TASK_ID = 0


def _known_derivative_direction(
    derivative: sp.Expr, *, nonnegative: bool
) -> Optional[bool]:
    """Return a strict sign verdict without simplify/factor/solve."""
    if nonnegative:
        if derivative.is_nonnegative is True:
            return True
        if derivative.is_negative is True:
            return False
    else:
        if derivative.is_nonpositive is True:
            return True
        if derivative.is_positive is True:
            return False
    return None


def _derivative_worker_loop(request_queue, result_queue) -> None:
    """Serve bounded derivative checks in an independently killable process."""
    result_queue.put(("ready", None, True, None, None))
    while True:
        task = request_queue.get()
        if task is None:
            return
        task_id, payload = task
        started = time.perf_counter()
        try:
            expr = payload["expr"]
            lanes = payload["lanes"]
            lane_parameters = payload["lane_parameters"]
            flows = np.asarray(payload["flows"], dtype=float)
            greens = np.asarray(payload["greens"], dtype=float)
            cycles = np.asarray(payload["cycles"], dtype=float)
            tolerance = float(payload["derivative_tolerance"])

            parsed, symbols = parse_symbolic_expression(expr)
            flow = symbols["flow_lane"]
            green = symbols["GR_phase"]
            cycle = symbols["Cycle_Time"]
            names = coefficient_names(expr)
            ordered_symbols = [flow, green, cycle] + [
                symbols[name] for name in names
            ]
            coefficient_arrays = [
                np.asarray(
                    [lane_parameters[lane][name] for lane in lanes],
                    dtype=float,
                )[:, None]
                for name in names
            ]
            flow_grid = np.broadcast_to(flows[None, :], (len(lanes), flows.size))
            green_grid = np.broadcast_to(
                greens[None, :], (len(lanes), greens.size)
            )
            cycle_grid = np.broadcast_to(
                cycles[None, :], (len(lanes), cycles.size)
            )

            flow_derivative = sp.diff(parsed, flow)
            green_derivative = sp.diff(parsed, green)
            flow_monotone = np.zeros(flow_grid.shape, dtype=bool)
            green_monotone = np.zeros(green_grid.shape, dtype=bool)
            methods = {
                lane: {"flow_lane": "", "GR_phase": ""} for lane in lanes
            }
            unresolved_flow: List[int] = []
            unresolved_green: List[int] = []

            for index, lane in enumerate(lanes):
                substitutions = {
                    symbols[name]: float(lane_parameters[lane][name])
                    for name in names
                }
                flow_direction = _known_derivative_direction(
                    flow_derivative.subs(substitutions), nonnegative=True
                )
                green_direction = _known_derivative_direction(
                    green_derivative.subs(substitutions), nonnegative=False
                )
                if flow_direction is None:
                    unresolved_flow.append(index)
                else:
                    flow_monotone[index, :] = flow_direction
                    methods[lane]["flow_lane"] = "symbolic_derivative_sign"
                if green_direction is None:
                    unresolved_green.append(index)
                else:
                    green_monotone[index, :] = green_direction
                    methods[lane]["GR_phase"] = "symbolic_derivative_sign"

            def evaluate_derivative(derivative: sp.Expr) -> np.ndarray:
                function = sp.lambdify(
                    ordered_symbols, derivative, modules="numpy"
                )
                with np.errstate(all="ignore"):
                    values = np.asarray(
                        function(
                            flow_grid,
                            green_grid,
                            cycle_grid,
                            *coefficient_arrays,
                        ),
                        dtype=float,
                    )
                return np.broadcast_to(values, flow_grid.shape)

            if unresolved_flow:
                flow_values = evaluate_derivative(flow_derivative)
                sampled = np.isfinite(flow_values) & (flow_values >= -tolerance)
                for index in unresolved_flow:
                    lane = lanes[index]
                    flow_monotone[index, :] = sampled[index, :]
                    methods[lane]["flow_lane"] = "sampled_analytic_derivative"
            if unresolved_green:
                green_values = evaluate_derivative(green_derivative)
                sampled = np.isfinite(green_values) & (green_values <= tolerance)
                for index in unresolved_green:
                    lane = lanes[index]
                    green_monotone[index, :] = sampled[index, :]
                    methods[lane]["GR_phase"] = "sampled_analytic_derivative"

            result_queue.put(
                (
                    "result",
                    task_id,
                    True,
                    {
                        "flow_monotone": flow_monotone.tolist(),
                        "green_monotone": green_monotone.tolist(),
                        "methods": methods,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    None,
                )
            )
        except Exception as exc:
            result_queue.put(
                (
                    "result",
                    task_id,
                    False,
                    {"elapsed_seconds": time.perf_counter() - started},
                    f"{type(exc).__name__}: {exc}",
                )
            )


def _stop_derivative_worker() -> None:
    """Terminate and release the persistent derivative worker, if present."""
    global _DERIVATIVE_WORKER_PROCESS
    global _DERIVATIVE_WORKER_REQUEST_QUEUE
    global _DERIVATIVE_WORKER_RESULT_QUEUE

    process = _DERIVATIVE_WORKER_PROCESS
    request_queue = _DERIVATIVE_WORKER_REQUEST_QUEUE
    result_queue = _DERIVATIVE_WORKER_RESULT_QUEUE
    if process is not None and process.is_alive():
        process.terminate()
        process.join(timeout=2.0)
    for worker_queue in (request_queue, result_queue):
        if worker_queue is not None:
            try:
                worker_queue.cancel_join_thread()
                worker_queue.close()
            except Exception:
                pass
    _DERIVATIVE_WORKER_PROCESS = None
    _DERIVATIVE_WORKER_REQUEST_QUEUE = None
    _DERIVATIVE_WORKER_RESULT_QUEUE = None


atexit.register(_stop_derivative_worker)


def _ensure_derivative_worker() -> None:
    """Start one persistent worker; startup time is excluded from task timeout."""
    global _DERIVATIVE_WORKER_PROCESS
    global _DERIVATIVE_WORKER_REQUEST_QUEUE
    global _DERIVATIVE_WORKER_RESULT_QUEUE

    if (
        _DERIVATIVE_WORKER_PROCESS is not None
        and _DERIVATIVE_WORKER_PROCESS.is_alive()
    ):
        return
    _stop_derivative_worker()
    if multiprocessing.current_process().daemon:
        raise RuntimeError("a daemon process cannot start the derivative worker")
    context = multiprocessing.get_context("spawn")
    request_queue = context.Queue(maxsize=1)
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_derivative_worker_loop,
        args=(request_queue, result_queue),
        name="cosydelay-symbolic-derivative",
    )
    process.daemon = True
    process.start()
    try:
        message_type, _, success, _, error = result_queue.get(timeout=15.0)
        if message_type != "ready" or not success:
            raise RuntimeError(error or "derivative worker did not become ready")
    except Exception:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        request_queue.close()
        result_queue.close()
        raise
    _DERIVATIVE_WORKER_PROCESS = process
    _DERIVATIVE_WORKER_REQUEST_QUEUE = request_queue
    _DERIVATIVE_WORKER_RESULT_QUEUE = result_queue


def _hybrid_monotonicity_with_timeout(
    expr: str,
    lanes: List[str],
    lane_parameters: Dict[str, Dict[str, float]],
    flows: np.ndarray,
    greens: np.ndarray,
    cycles: np.ndarray,
    derivative_tolerance: float,
    timeout: float,
    flow_fallback: np.ndarray,
    green_fallback: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, str]], float, Optional[str]]:
    """Run derivative-first monotonicity with a hard process timeout."""
    global _DERIVATIVE_WORKER_TASK_ID
    fallback_methods = {
        lane: {
            "flow_lane": "finite_difference_fallback",
            "GR_phase": "finite_difference_fallback",
        }
        for lane in lanes
    }
    started = time.perf_counter()
    try:
        _ensure_derivative_worker()
        _DERIVATIVE_WORKER_TASK_ID += 1
        task_id = _DERIVATIVE_WORKER_TASK_ID
        payload = {
            "expr": expr,
            "lanes": lanes,
            "lane_parameters": {
                lane: dict(lane_parameters[lane]) for lane in lanes
            },
            "flows": np.asarray(flows, dtype=float).tolist(),
            "greens": np.asarray(greens, dtype=float).tolist(),
            "cycles": np.asarray(cycles, dtype=float).tolist(),
            "derivative_tolerance": float(derivative_tolerance),
        }
        _DERIVATIVE_WORKER_REQUEST_QUEUE.put((task_id, payload), timeout=timeout)
        message_type, result_task_id, success, result, error = (
            _DERIVATIVE_WORKER_RESULT_QUEUE.get(timeout=timeout)
        )
        if message_type != "result" or result_task_id != task_id:
            raise RuntimeError("derivative worker returned a mismatched result")
        if not success:
            raise RuntimeError(error or "derivative worker failed")
        return (
            np.asarray(result["flow_monotone"], dtype=bool),
            np.asarray(result["green_monotone"], dtype=bool),
            result["methods"],
            float(result["elapsed_seconds"]),
            None,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        if isinstance(exc, queue.Empty):
            reason = f"TimeoutError: analytic derivative exceeded {timeout:.3f}s"
        else:
            reason = f"{type(exc).__name__}: {exc}"
        _stop_derivative_worker()
        return (
            np.asarray(flow_fallback, dtype=bool),
            np.asarray(green_fallback, dtype=bool),
            fallback_methods,
            elapsed,
            reason,
        )


def _has_green_dependent_exponential(
    expression: sp.Expr,
    green: sp.Symbol,
) -> bool:
    """Identify zero-green limits that need process isolation."""
    return any(
        green in exp_node.args[0].free_symbols
        for exp_node in expression.atoms(sp.exp)
    )


def _known_nonnegative_sign(expression: sp.Expr) -> Optional[int]:
    """Return 1 for proven positive, 0 for proven nonnegative, else None.

    SymPy does not currently infer the common traffic saturation factor
    ``1 - exp(-positive)`` as positive. Recognizing it explicitly lets the R7
    verifier prove inverse-green exponential limits without invoking the slow
    general-purpose limit engine.
    """
    if getattr(expression, "is_positive", None) is True:
        return 1
    if getattr(expression, "is_zero", None) is True:
        return 0
    if getattr(expression, "is_nonnegative", None) is True:
        return 0

    positive = sp.Wild(
        "_positive_gap", properties=[lambda value: value.is_positive is True]
    )
    if expression.match(1 - sp.exp(-positive)) is not None:
        return 1
    if expression.match(sp.exp(positive) - 1) is not None:
        return 1
    factored = sp.factor(expression)
    if factored != expression:
        if factored.match(
            (sp.exp(positive) - 1) * sp.exp(-positive)
        ) is not None:
            return 1
        factored_sign = _known_nonnegative_sign(factored)
        if factored_sign is not None:
            return factored_sign

    if expression.is_Mul:
        signs = [_known_nonnegative_sign(arg) for arg in expression.args]
        if all(sign is not None for sign in signs):
            return 1 if all(sign == 1 for sign in signs) else 0
    if expression.is_Add:
        signs = [_known_nonnegative_sign(arg) for arg in expression.args]
        if all(sign is not None for sign in signs):
            return 1 if any(sign == 1 for sign in signs) else 0
    return None


def _proves_positive_infinite_green_exponential(
    expression: sp.Expr,
    green: sp.Symbol,
    *,
    timeout: float = SYMBOLIC_LIMIT_PROOF_TIMEOUT_SECONDS,
) -> bool:
    """Prove a sufficient class of ``GR_phase -> 0+`` exponential limits.

    The proof is deliberately conservative. It succeeds only when the full
    expression is a polynomial in a green-dependent exponential, that
    exponential's argument tends to +infinity, its leading coefficient is
    strictly positive, and every other coefficient is nonnegative.
    Timed-out argument limits are treated as inconclusive, never as proofs.
    """
    for exp_node in expression.atoms(sp.exp):
        if green not in exp_node.args[0].free_symbols:
            continue
        try:
            argument_limit = _symbolic_limit_with_timeout(
                exp_node.args[0],
                green,
                0,
                timeout=timeout,
            )
            if argument_limit != sp.oo:
                continue

            marker = sp.Dummy("green_exponential", positive=True)
            polynomial = sp.Poly(
                expression.xreplace({exp_node: marker}),
                marker,
                domain="EX",
            )
            if polynomial.degree() < 1:
                continue
            coefficient_signs = [
                _known_nonnegative_sign(coefficient)
                for coefficient in polynomial.all_coeffs()
            ]
            if coefficient_signs[0] == 1 and all(
                sign is not None for sign in coefficient_signs
            ):
                return True
        except TimeoutError:
            continue
        except Exception:
            continue
    return False


def _proves_positive_infinite_green_by_addends(
    expression: sp.Expr,
    green: sp.Symbol,
    *,
    timeout: float = SYMBOLIC_LIMIT_PROOF_TIMEOUT_SECONDS,
) -> bool:
    """Prove R9 for sums by limiting each addend separately.

    Full limits of long green exponentials can stall SymPy even when every
    addend already diverges to ``+oo``.  This check is conservative: at least
    one addend must tend to ``+oo`` and no addend may tend to ``-oo`` or a
    negative finite value.  Any timed-out addend aborts the proof as
    inconclusive.
    """
    expanded = sp.expand(expression)
    if not expanded.is_Add:
        return False

    has_positive_infinite = False
    for term in expanded.args:
        try:
            limit_value = _symbolic_limit_with_timeout(
                term,
                green,
                0,
                timeout=timeout,
            )
        except TimeoutError:
            return False
        except Exception:
            return False
        if limit_value == sp.oo:
            has_positive_infinite = True
        elif limit_value == -sp.oo:
            return False
        elif getattr(limit_value, "is_finite", None) is True:
            if getattr(limit_value, "is_negative", None) is True:
                return False
        elif isinstance(limit_value, sp.Limit) or limit_value.has(sp.Limit):
            return False
        else:
            return False
    return has_positive_infinite


@lru_cache(maxsize=4096)
def compile_numpy_function(expr_str: str, symbol_names: Tuple[str, ...]):
    """Cache lambdified expressions because physical validation evaluates many points."""
    symbols = tuple(sp.Symbol(name) for name in symbol_names)
    sp_expr = sp.sympify(expr_str, locals={name: symbol for name, symbol in zip(symbol_names, symbols)})
    return lambdify(symbols, sp_expr, modules=['numpy', {'exp': safe_exp, 'pow': safe_power, 'log': safe_log}])


def evaluate_with_timeout(sp_expr, point: Dict, timeout: float = 3.0) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    使用多进程实现超时评估（与原代码相同）
    默认超时时间：3秒
    返回: (成功标志, 计算值, 错误信息)
    """

    def _evaluate_expr(expr_str, point_dict, result_queue):
        """在子进程中评估表达式"""
        try:
            # 重新解析表达式
            symbols = {k: sp.Symbol(k) for k in point_dict.keys()}
            expr = sp.sympify(expr_str, locals=symbols)

            # 评估
            raw_result = expr.subs(point_dict)
            if getattr(raw_result, "is_real", None) is False:
                result_queue.put((False, None, "Complex-valued result"))
                return
            result = float(raw_result)

            # 检查结果
            if np.isnan(result):
                result_queue.put((False, None, "Result is NaN"))
            elif np.isinf(result):
                result_queue.put((True, float('inf'), None))
            elif abs(result) > 1e15:
                result_queue.put((False, None, f"Result too large: {result}"))
            else:
                result_queue.put((True, result, None))

        except (OverflowError, ValueError) as e:
            if "overflow" in str(e).lower():
                result_queue.put((True, float('inf'), None))
            else:
                result_queue.put((False, None, str(e)))
        except Exception as e:
            result_queue.put((False, None, f"Evaluation error: {str(e)}"))

    try:
        # 转换表达式为字符串
        expr_str = str(sp_expr)

        # 创建结果队列
        result_queue = multiprocessing.Queue()

        # 创建并启动子进程
        process = multiprocessing.Process(
            target=_evaluate_expr,
            args=(expr_str, point, result_queue)
        )
        process.start()

        # 等待结果或超时
        process.join(timeout=timeout)

        if process.is_alive():
            # 超时，终止进程
            process.terminate()
            process.join()
            return False, None, "Evaluation timed out"

        # 获取结果
        if not result_queue.empty():
            return result_queue.get()
        else:
            return False, None, "No result returned"

    except Exception as e:
        return False, None, f"Process error: {str(e)}"


def safe_evaluate_simple(sp_expr, point):
    """
    使用 numpy 进行快速数值计算（与原代码相同）
    """
    try:
        def safe_exp(x):
            """防止 exp 溢出的安全版本"""
            x_clipped = np.clip(x, -100, 100)
            with np.errstate(all='ignore'):
                result = np.exp(x_clipped)
            return result

        def safe_power(base, exponent):
            """防止幂运算溢出"""
            base = np.asarray(base, dtype=float)
            exponent = np.asarray(exponent, dtype=float)
            exponent = np.clip(exponent, -50, 50)
            base = np.maximum(np.abs(base), 1e-10)

            with np.errstate(all='ignore'):
                log_result = exponent * np.log(base)
                log_result = np.clip(log_result, -100, 100)
                result = np.exp(log_result)

            return np.clip(result, 1e-100, 1e100)

        def safe_log(x):
            """防止 log 无效值的安全版本"""
            x_safe = np.maximum(x, 1e-300)
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.log(x_safe)
            return result

        # 获取表达式中的所有符号
        symbol_names = tuple(sorted(str(sym) for sym in sp_expr.free_symbols))

        # 将 sympy 表达式转换为 numpy 函数
        func = compile_numpy_function(str(sp_expr), symbol_names)

        # 准备参数
        args = [point.get(name, point.get(sp.Symbol(name), 0)) for name in symbol_names]

        # NumPy may warn for overflow or division by zero while testing a
        # candidate expression. The result checks below decide validity.
        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
            under="ignore",
        ):
            result = func(*args)

        # 处理 numpy 数组返回值
        if np.iscomplexobj(result):
            return False, None, "Complex-valued result"
        if isinstance(result, np.ndarray):
            result = float(result.item())
        else:
            result = float(result)

        # Preserve the original sentinel-value behavior. Physical validation
        # decides whether these values are acceptable.
        if np.isnan(result):
            result = 1e7
        elif np.isinf(result):
            result = 1e7 if result > 0 else -1e7

        return True, result, None

    except (ValueError, TypeError, ZeroDivisionError) as e:
        return False, None, f"Evaluation error: {str(e)}"
    except OverflowError:
        return True, 1e7, None
    except Exception as e:
        # 回退到原方法
        try:
            with np.errstate(all='ignore'):
                result = float(sp_expr.subs(point).evalf(n=10, maxn=1000))

            if np.isnan(result):
                result = 1e7
            elif np.isinf(result):
                result = 1e7 if result > 0 else -1e7

            result = np.clip(result, -1e10, 1e10)
            return True, result, None
        except:
            return False, None, f"Unexpected error: {str(e)}"


def safe_evaluate(sp_expr, point: Dict, timeout: float = 3.0) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    智能安全评估：先尝试快速评估，失败则使用超时版本（与原代码相同）
    """
    # 先尝试简单评估
    success, result, error = safe_evaluate_simple(sp_expr, point)

    if success or "too large" not in str(error).lower():
        return success, result, error

    # 如果简单评估因为值太大失败，使用带超时的版本
    return evaluate_with_timeout(sp_expr, point, timeout)


@lru_cache(maxsize=32)
def generate_universal_test_points(
    n_samples: int = 100,
    seed: int = 42,
    flow_domain: Tuple[float, float] = TRAFFIC_FLOW_DOMAIN,
    green_domain: Tuple[float, float] = TRAFFIC_GREEN_DOMAIN,
    cycle_domain: Tuple[float, float] = TRAFFIC_CYCLE_DOMAIN_SECONDS,
) -> Tuple[Dict[str, float], ...]:
    """
    Generate test points using Latin Hypercube Sampling (与原代码逻辑相同)
    For universal lane features: flow_lane, GR_phase, Cycle_Time
    """
    n_dims = 3
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    samples = sampler.random(n=n_samples)

    test_points = []
    for sample in samples:
        test_point = {
            'flow_lane': flow_domain[0] + sample[0] * (flow_domain[1] - flow_domain[0]),
            'GR_phase': green_domain[0] + sample[1] * (green_domain[1] - green_domain[0]),
            'Cycle_Time': cycle_domain[0] + sample[2] * (cycle_domain[1] - cycle_domain[0]),
        }
        test_points.append(test_point)

    # Add critical boundary test points
    boundary_points = []

    # Boundary 1: Zero flow
    zero_flow_point = {
        'flow_lane': 0.0,
        'GR_phase': 0.5,
        'Cycle_Time': 120
    }
    boundary_points.append(zero_flow_point)

    # Boundary 2: Small green ratio with high flow
    small_gr_point = {
        'flow_lane': flow_domain[1],
        'GR_phase': green_domain[0],
        'Cycle_Time': cycle_domain[1]
    }
    boundary_points.append(small_gr_point)

    # Boundary 3: Oversaturation
    high_flow_point = {
        'flow_lane': flow_domain[1],
        'GR_phase': 0.2,
        'Cycle_Time': 120
    }
    boundary_points.append(high_flow_point)

    # Boundary 4: Small positive flow at maximum green and minimum cycle.
    low_flow_point = {
        'flow_lane': flow_domain[0],
        'GR_phase': green_domain[1],
        'Cycle_Time': cycle_domain[0]
    }
    boundary_points.append(low_flow_point)

    return tuple(boundary_points + test_points)


def check_syntax(expr: str) -> Tuple[bool, Optional[str]]:
    """Basic syntax validation"""
    if expr.count('(') != expr.count(')'):
        return False, "Unbalanced parentheses"
    return True, None


def check_zero_flow_zero_delay(sp_expr, zero_flow_point: Dict) -> Tuple[bool, Optional[str]]:
    """Check R8: zero flow must give exactly zero delay.

    Used by the pre-fit expression validator and by :func:`_evaluate_zero_flow_boundary`.
    """
    success, delay, error = safe_evaluate(sp_expr, zero_flow_point)

    if not success:
        return False, f"Cannot evaluate expression at zero flow: {error}"

    if delay is not None and abs(delay) > 1e-6:
        return False, f"Zero flow does not yield zero delay: got {delay:.4f}"

    return True, None


def check_finite_nonnegative_low_demand_limit(
    fitted_expression: sp.Expr,
    flow: sp.Symbol,
    green: sp.Symbol,
    cycle: sp.Symbol,
) -> Tuple[bool, Optional[sp.Expr], Optional[str]]:
    """Establish finalized R5 symbolically over 0 < green <= 1 and cycle > 0.

    ``green = 1 / (1 + odds)`` with ``odds >= 0`` represents the complete
    mathematical green-ratio domain.  A numerical probe is intentionally not a
    fallback: Table I assigns zero when the required analytic condition is
    undecidable.
    """
    try:
        green_odds = sp.Symbol("_green_odds", nonnegative=True, finite=True)
        positive_cycle = sp.Symbol("_positive_cycle", positive=True, finite=True)
        limit_value = sp.limit(fitted_expression, flow, 0, dir="+")
        limit_value = sp.simplify(
            limit_value.subs(
                {
                    green: 1 / (1 + green_odds),
                    cycle: positive_cycle,
                }
            )
        )
        if isinstance(limit_value, sp.Limit) or limit_value.has(sp.Limit):
            return False, limit_value, "low-demand limit remained unevaluated"
        finite = getattr(limit_value, "is_finite", None) is True
        nonnegative = getattr(limit_value, "is_nonnegative", None) is True
        if finite and nonnegative:
            return True, limit_value, None
        properties = (
            f"finite={getattr(limit_value, 'is_finite', None)}, "
            f"nonnegative={getattr(limit_value, 'is_nonnegative', None)}"
        )
        return (
            False,
            limit_value,
            f"low-demand limit is not established as finite and non-negative ({properties})",
        )
    except Exception as exc:
        return False, None, f"low-demand limit undecidable: {exc}"


def _positive_parameter_expression(
    parsed: sp.Expr,
    symbols: Dict[str, sp.Symbol],
    names: List[str],
) -> sp.Expr:
    """Replace fitted coefficients by positive symbols for a shared proof.

    Every optimizer bound in :mod:`optimization_lane` is strictly positive.
    Proving a limit with positive symbolic coefficients is therefore valid for
    every fitted lane and, importantly, avoids asking SymPy to manipulate the
    optimizer's 15--17 digit floating-point values.  Those long decimals can
    otherwise be rationalized into enormous integers during ``limit`` and
    ``simplify`` and make a small expression take minutes per lane.
    """
    positive_coefficients = {
        symbols[name]: sp.Symbol(
            f"_positive_{name}", positive=True, finite=True
        )
        for name in names
    }
    return parsed.subs(positive_coefficients)


def _compact_symbolic_number(value: float) -> sp.Rational:
    """Return a bounded-complexity copy of a fitted value for symbolic fallback.

    This copy is used only when a property depends on relative coefficient
    magnitudes and the shared positive-parameter proof is inconclusive.  Model
    predictions and serialized fitted parameters continue to use full-precision
    floats.  Two significant digits keep exponent denominators small enough for
    SymPy while preserving coefficient sign and practical scale.
    """
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"non-finite fitted coefficient: {value!r}")
    return sp.Rational(format(numeric, ".2g"))


def check_small_green_large_delay(sp_expr, small_gr_point: Dict) -> Tuple[bool, Optional[str]]:
    """
    Small green ratio WITH high flow should give large delay
    使用3秒超时来处理极端值（与原代码相同）
    """
    try:
        limit_expr = sp_expr.subs({
            sp.Symbol('flow_lane'): 1.0,
            sp.Symbol('Cycle_Time'): 90.0
        })
        limit_value = sp.limit(limit_expr, sp.Symbol('GR_phase'), 0, dir='+')
        if limit_value in (sp.oo, sp.zoo):
            return True, None
        if limit_value == -sp.oo:
            return False, "Expression tends to negative infinity as GR_phase -> 0"
    except Exception:
        pass

    limit_point = small_gr_point.copy()

    # 测试绿灯比趋近于0的情况
    test_grs = [0.1, 0.01, 0.001, 1e-06]
    any_large_delay = False

    for gr in test_grs:
        test_point = limit_point.copy()
        test_point['GR_phase'] = gr
        test_point['flow_lane'] = 10  # 使用中等流量

        success, delay, error = safe_evaluate(sp_expr, test_point, timeout=3.0)

        if not success:
            if "timed out" in str(error).lower():
                any_large_delay = True
                break
            continue

        if delay is not None:
            if np.isinf(delay):
                any_large_delay = True
                break

            # 在最小绿灯比时，延迟应该非常大
            if gr == 1e-06 and delay >= 1e6:
                any_large_delay = True

    if not any_large_delay:
        return False, f"Expression does not produce sufficiently large delay as GR_phase → 0"

    return True, None


def check_non_negative_delay(sp_expr, test_points: List[Dict],
                             start_idx: int = 3, end_idx: int = 33) -> Tuple[bool, Optional[str]]:
    """Expression should yield non-negative delay"""
    for i, test_point in enumerate(test_points[start_idx:end_idx], start_idx):
        success, delay_val, error = safe_evaluate(sp_expr, test_point, timeout=2.0)

        if not success:
            continue

        if delay_val is not None and delay_val < -1e-6:
            return False, f"Expression yields negative delay"

    return True, None


def check_monotonicity(derivatives: Dict, test_points: List[Dict],
                       expr: str, start_idx: int = 3, end_idx: int = 33) -> Tuple[bool, Optional[str]]:
    """First derivative w.r.t. flow_lane should be non-negative"""
    if 'flow_lane' in expr and "flow_lane_d1" in derivatives:
        for i, test_point in enumerate(test_points[start_idx:end_idx], start_idx):
            success, deriv1_val, error = safe_evaluate(derivatives["flow_lane_d1"], test_point, timeout=2.0)

            if not success:
                continue

            if deriv1_val is not None and deriv1_val < -1e-6:
                return False, f"Monotonicity violated for flow_lane"

    return True, None


def check_inverse_green_ratio(derivatives: Dict, test_points: List[Dict],
                              expr: str, start_idx: int = 3, end_idx: int = 33) -> Tuple[bool, Optional[str]]:
    """Derivative w.r.t. GR_phase should be non-positive"""
    if 'GR_phase' in expr and "GR_phase_d1" in derivatives:
        for i, test_point in enumerate(test_points[start_idx:end_idx], start_idx):
            success, deriv_val, error = safe_evaluate(derivatives["GR_phase_d1"], test_point, timeout=2.0)

            if not success:
                continue

            if deriv_val is not None and deriv_val > 1e-6:
                return False, f"Inverse relationship violated for GR_phase"

    return True, None


def check_unit_consistency(expr: str, features: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Unit consistency validation (与原代码逻辑相同)

    Universal lane features:
    - flow_lane: dimensionless (flow ratio)
    - GR_phase: dimensionless (green ratio)
    - Cycle_Time: seconds [s]

    Result must have dimension [s] (seconds)
    """
    units = {
        'Cycle_Time': 's',
        'flow_lane': '',  # dimensionless
        'GR_phase': ''  # dimensionless
    }

    def _extract_power(exp):
        if isinstance(exp, (int, float, sp.Integer, sp.Rational, sp.Float)):
            return float(exp)
        return None

    try:
        # Replace coefficients with 1.0 for unit analysis
        temp_expr = re.sub(r'\ba\d+\b', '1.0', expr)

        sp_expr = sp.sympify(temp_expr, locals={f: sp.Symbol(f) for f in features})
        terms = sp.Add.make_args(sp_expr) if isinstance(sp_expr, sp.Add) else [sp_expr]

        for term in terms:
            num, denom = sp.fraction(term)
            term_units = {}

            # Process numerator
            for arg in sp.Mul.make_args(num):
                if isinstance(arg, sp.Symbol) and str(arg) in features:
                    term_units[str(arg)] = term_units.get(str(arg), 0) + 1
                elif isinstance(arg, sp.Pow) and str(arg.base) in features:
                    power = _extract_power(arg.exp)
                    if power is None:
                        return False, f"Unit validation failed: non-numeric exponent for {arg.base} in term {term}"
                    term_units[str(arg.base)] = term_units.get(str(arg.base), 0) + power
                elif isinstance(arg, (sp.exp, sp.log)):
                    # Arguments of exp() and log() must be dimensionless
                    arg_content = arg.args[0]
                    sub_term_units = {}
                    for sub_arg in sp.Mul.make_args(arg_content):
                        if isinstance(sub_arg, sp.Symbol) and str(sub_arg) in features:
                            sub_term_units[str(sub_arg)] = sub_term_units.get(str(sub_arg), 0) + 1
                        elif isinstance(sub_arg, sp.Pow) and str(sub_arg.base) in features:
                            power = _extract_power(sub_arg.exp)
                            if power is None:
                                return False, f"Unit validation failed: non-numeric exponent in {arg} in term {term}"
                            sub_term_units[str(sub_arg.base)] = sub_term_units.get(str(sub_arg.base), 0) + power

                    # Calculate unit powers for argument
                    sub_unit_powers = {}
                    for f, p in sub_term_units.items():
                        feature_unit = units.get(f, '')
                        if feature_unit:
                            sub_unit_powers[feature_unit] = sub_unit_powers.get(feature_unit, 0) + p

                    if sub_unit_powers:
                        return False, f"Unit validation failed: {arg} argument must be dimensionless, got {sub_unit_powers} in term {term}"

            # Process denominator
            for arg in sp.Mul.make_args(denom):
                if isinstance(arg, sp.Symbol) and str(arg) in features:
                    term_units[str(arg)] = term_units.get(str(arg), 0) - 1
                elif isinstance(arg, sp.Pow) and str(arg.base) in features:
                    power = _extract_power(arg.exp)
                    if power is None:
                        return False, f"Unit validation failed: non-numeric exponent for {arg.base} in term {term}"
                    term_units[str(arg.base)] = term_units.get(str(arg.base), 0) - power
                elif isinstance(arg, (sp.exp, sp.log)):
                    arg_content = arg.args[0]
                    sub_term_units = {}
                    for sub_arg in sp.Mul.make_args(arg_content):
                        if isinstance(sub_arg, sp.Symbol) and str(sub_arg) in features:
                            sub_term_units[str(sub_arg)] = sub_term_units.get(str(sub_arg), 0) + 1
                        elif isinstance(sub_arg, sp.Pow) and str(sub_arg.base) in features:
                            power = _extract_power(sub_arg.exp)
                            if power is None:
                                return False, f"Unit validation failed: non-numeric exponent in {arg} in term {term}"
                            sub_term_units[str(sub_arg.base)] = sub_term_units.get(str(sub_arg.base), 0) + power

                    sub_unit_powers = {}
                    for f, p in sub_term_units.items():
                        feature_unit = units.get(f, '')
                        if feature_unit:
                            sub_unit_powers[feature_unit] = sub_unit_powers.get(feature_unit, 0) + p

                    if sub_unit_powers:
                        return False, f"Unit validation failed: {arg} argument must be dimensionless, got {sub_unit_powers} in term {term}"

            # Calculate final unit powers for this term
            unit_powers = {}
            for feature, power in term_units.items():
                feature_unit = units.get(feature, '')
                if feature_unit:
                    unit_powers[feature_unit] = unit_powers.get(feature_unit, 0) + power
            unit_powers = {u: round(p, 6) for u, p in unit_powers.items() if abs(p) > 1e-6}

            # Check if term has correct time dimension [s]
            if unit_powers != {'s': 1}:
                return False, f"Unit inconsistency in term {term}: expected {{'s': 1}}, got {unit_powers or 'dimensionless'}"

    except Exception as e:
        return False, f"Unit validation failed: {str(e)}"

    return True, None


def validate_universal_lane_expression(expr: str, universal_features: List[str],
                                       intersection_id: int = 1) -> Tuple[bool, str]:
    """
    Legacy pre-fit validation function retained for artifact compatibility.

    It includes the historical exact-zero-flow check and must not be used as a
    substitute for ``score_fitted_lanes_principlewise`` under the finalized
    Table-I schema.

    Collects ALL errors before returning (不会提前返回)

    Returns False if any check fails, along with all error messages
    """
    errors = []
    passed = True

    try:
        # Check 1: Required features - flow_lane
        if 'flow_lane' not in expr:
            passed = False
            errors.append("Required features check failed: Missing flow_lane")

        # Check 2: Required features - GR_phase
        if 'GR_phase' not in expr:
            passed = False
            errors.append("Required features check failed: Missing GR_phase")

        # Check 3: No negative coefficient for flow_lane
        negative_pattern = re.compile(r'-\s*\d*\.?\d+\s*\*\s*flow_lane')
        if negative_pattern.search(expr):
            passed = False
            errors.append("Negative coefficients check failed: Negative coefficient for flow_lane")

        # Check 4: Syntax
        check_passed, error = check_syntax(expr)
        if not check_passed:
            passed = False
            errors.append(f"Syntax check failed: {error}")

        # Parse expression (replace coefficients with 1.0 for validation)
        sp_expr = None
        try:
            temp_expr = re.sub(r'\ba\d+\b', '1.0', expr)

            sp_expr = sp.sympify(temp_expr, locals={
                'flow_lane': sp.Symbol('flow_lane'),
                'GR_phase': sp.Symbol('GR_phase'),
                'Cycle_Time': sp.Symbol('Cycle_Time')
            })
        except Exception as e:
            passed = False
            errors.append(f"Expression parsing failed: {str(e)}")
            # 继续收集其他错误，但跳过需要sp_expr的检查

        # 只有成功解析才继续后续检查
        if sp_expr is not None:
            # Generate test points
            test_points = generate_universal_test_points(n_samples=50)

            # Check 5: Zero flow zero delay
            check_passed, error = check_zero_flow_zero_delay(sp_expr, test_points[0])
            if not check_passed:
                passed = False
                errors.append(f"Zero flow zero delay check failed: {error}")

            # Prepare derivatives
            derivatives = {}
            try:
                if 'flow_lane' in expr:
                    derivatives["flow_lane_d1"] = sp.diff(sp_expr, sp.Symbol('flow_lane'))
                    derivatives["flow_lane_d2"] = sp.diff(derivatives["flow_lane_d1"], sp.Symbol('flow_lane'))

                if 'GR_phase' in expr:
                    derivatives["GR_phase_d1"] = sp.diff(sp_expr, sp.Symbol('GR_phase'))
            except Exception as e:
                passed = False
                errors.append(f"Derivative calculation failed: {str(e)}")

            # Check 7: Non-negative delay
            check_passed, error = check_non_negative_delay(sp_expr, test_points)
            if not check_passed:
                passed = False
                errors.append(f"Non-negative delay check failed: {error}")

            # Check 8: Monotonicity
            check_passed, error = check_monotonicity(derivatives, test_points, expr)
            if not check_passed:
                passed = False
                errors.append(f"Monotonicity check failed: {error}")

            # Check 9: Inverse green ratio
            check_passed, error = check_inverse_green_ratio(derivatives, test_points, expr)
            if not check_passed:
                passed = False
                errors.append(f"Inverse green ratio check failed: {error}")

        # Check 11: Unit consistency (can be checked independently)
        check_passed, error = check_unit_consistency(expr, universal_features)
        if not check_passed:
            passed = False
            errors.append(f"Unit consistency check failed: {error}")

        # 返回所有收集到的错误
        if errors:
            return False, "\n".join(errors)

        return True, f"Universal lane expression is physically consistent for intersection {intersection_id}"

    except Exception as e:
        errors.append(f"Unexpected validation error: {str(e)}")
        return False, "\n".join(errors)


def _empty_physical_score(
    lanes: List[str],
    config: PhysicalVerifierConfig,
    message: str,
) -> PhysicalScoreResult:
    rule_scores = {name: 0.0 for name in PHYSICAL_RULE_NAMES}
    lane_rule_scores = {
        lane: {name: 0.0 for name in PHYSICAL_RULE_NAMES}
        for lane in lanes
    }
    return PhysicalScoreResult(
        score=0.0,
        joint_pass=False,
        rule_scores=rule_scores,
        lane_rule_scores=lane_rule_scores,
        lane_errors={lane: [message] for lane in lanes},
        diagnostics={"fatal_error": message},
        config=config,
    )


def _score_fitted_lanes_legacy_table_i(
    expr: str,
    lane_parameters: Dict[str, Dict[str, float]],
    lanes: List[str],
    universal_features: List[str],
    config: Optional[PhysicalVerifierConfig] = None,
) -> PhysicalScoreResult:
    """Score seven declared traffic-delay principles after coefficient fitting.

    R1, R4, R5, and R7 are binary structural/unit/analytic checks. R2, R3, and
    R6 are the proportions of a deterministic Latin-hypercube grid that pass
    the declared inequalities. A non-finite value counts as a violation; grid
    passing is diagnostic evidence over the declared domain, not a global proof.
    """
    config = config or PhysicalVerifierConfig()
    lane_errors: Dict[str, List[str]] = {lane: [] for lane in lanes}
    lane_rule_scores: Dict[str, Dict[str, float]] = {
        lane: {name: 0.0 for name in PHYSICAL_RULE_NAMES}
        for lane in lanes
    }
    low_demand_limits: Dict[str, Optional[str]] = {}
    zero_green_limits: Dict[str, Optional[str]] = {}
    active_lanes = [lane for lane in lanes if lane in lane_parameters]
    for lane in lanes:
        if lane not in lane_parameters:
            lane_errors[lane].append("missing fitted parameters")
    if not active_lanes:
        return _empty_physical_score(lanes, config, "no fitted lane parameters")

    try:
        parsed, symbols = parse_symbolic_expression(expr)
    except Exception as exc:
        return _empty_physical_score(lanes, config, f"expression parsing failed: {exc}")

    flow = symbols["flow_lane"]
    green = symbols["GR_phase"]
    cycle = symbols["Cycle_Time"]
    required_ok = {flow, green, cycle}.issubset(parsed.free_symbols)
    unit_ok, unit_error = check_unit_consistency(expr, universal_features)

    for lane in active_lanes:
        lane_rule_scores[lane]["R1_required_variables"] = float(required_ok)
        lane_rule_scores[lane]["R4_time_dimension"] = float(unit_ok)
        if not required_ok:
            lane_errors[lane].append("R1: expression must use flow_lane, GR_phase, and Cycle_Time")
        if not unit_ok:
            lane_errors[lane].append(f"R4: {unit_error}")

    names = coefficient_names(expr)
    try:
        missing_coefficients = {
            lane: [name for name in names if name not in lane_parameters[lane]]
            for lane in active_lanes
        }
        bad_lanes = [lane for lane, missing in missing_coefficients.items() if missing]
        if bad_lanes:
            for lane in bad_lanes:
                lane_errors[lane].append(
                    "missing coefficients: " + ", ".join(missing_coefficients[lane])
                )
            return _empty_physical_score(
                lanes,
                config,
                "one or more lanes have incomplete fitted coefficients",
            )

        ordered_symbols = [flow, green, cycle] + [symbols[name] for name in names]
        value_func = sp.lambdify(ordered_symbols, parsed, modules="numpy")
        flow_derivative_func = sp.lambdify(
            ordered_symbols, sp.diff(parsed, flow), modules="numpy"
        )
        green_derivative_func = sp.lambdify(
            ordered_symbols, sp.diff(parsed, green), modules="numpy"
        )

        # R5 and R7 are symbolic properties.  First prove them once under the
        # optimizer's shared assumption that every coefficient is positive.
        # A successful proof applies to all lanes and replaces 24 nearly
        # identical, potentially pathological SymPy limit calculations.
        positive_parameter_expression = _positive_parameter_expression(
            parsed, symbols, names
        )
        generic_low_ok, generic_low_limit, generic_low_error = (
            check_finite_nonnegative_low_demand_limit(
                positive_parameter_expression,
                flow,
                green,
                cycle,
            )
        )
        generic_low_decisive = bool(
            generic_low_ok
            or (
                generic_low_limit is not None
                and (
                    getattr(generic_low_limit, "is_finite", None) is False
                    or getattr(generic_low_limit, "is_nonnegative", None) is False
                )
            )
        )

        positive_flow = sp.Symbol("_positive_flow", positive=True)
        positive_cycle = sp.Symbol("_positive_cycle", positive=True)
        try:
            generic_zero_green_expression = positive_parameter_expression.subs(
                {flow: positive_flow, cycle: positive_cycle}
            )
            if _proves_positive_infinite_green_exponential(
                generic_zero_green_expression, green
            ):
                generic_zero_green_limit = sp.oo
            else:
                generic_zero_green_limit = _symbolic_limit_with_timeout(
                    generic_zero_green_expression,
                    green,
                    0,
                    timeout=config.symbolic_limit_timeout_seconds,
                )
            generic_zero_green_ok = bool(
                generic_zero_green_limit == sp.oo
                or (
                    getattr(generic_zero_green_limit, "is_infinite", False) is True
                    and getattr(generic_zero_green_limit, "is_positive", False) is True
                )
            )
            generic_zero_green_decisive = bool(
                generic_zero_green_ok
                or getattr(generic_zero_green_limit, "is_finite", None) is True
                or generic_zero_green_limit == -sp.oo
            )
            generic_zero_green_error = None
        except Exception as exc:
            generic_zero_green_limit = None
            generic_zero_green_ok = False
            # A timed-out shared proof fails closed for every lane. Retrying the
            # same pathological structure 12 times would only multiply the
            # timeout without adding evidence.
            generic_zero_green_decisive = isinstance(exc, TimeoutError)
            generic_zero_green_error = str(exc)

        coefficient_arrays = [
            np.asarray(
                [lane_parameters[lane][name] for lane in active_lanes], dtype=float
            )[:, None]
            for name in names
        ]

        def evaluate_grid(func, flow_values, green_values, cycle_values):
            def as_lane_grid(values):
                array = np.asarray(values, dtype=float)
                if array.ndim == 0:
                    array = array.reshape(1, 1)
                elif array.ndim == 1:
                    array = array[None, :]
                if array.shape[0] == 1:
                    array = np.broadcast_to(
                        array, (len(active_lanes), array.shape[1])
                    )
                if array.shape[0] != len(active_lanes):
                    raise ValueError("validation grid lane dimension mismatch")
                return array

            flow_array = as_lane_grid(flow_values)
            green_array = as_lane_grid(green_values)
            cycle_array = as_lane_grid(cycle_values)
            with np.errstate(all="ignore"):
                values = np.asarray(
                    func(
                        flow_array,
                        green_array,
                        cycle_array,
                        *coefficient_arrays,
                    ),
                    dtype=float,
                )
            return np.broadcast_to(values, flow_array.shape)

        points = generate_universal_test_points(
            config.n_grid_samples,
            config.lhs_seed,
            config.flow_domain,
            config.green_domain,
            config.cycle_domain_seconds,
        )[1:]
        test_flows = [point["flow_lane"] for point in points]
        test_greens = [point["GR_phase"] for point in points]
        test_cycles = [point["Cycle_Time"] for point in points]
        delay_values = evaluate_grid(value_func, test_flows, test_greens, test_cycles)
        flow_derivatives = evaluate_grid(
            flow_derivative_func, test_flows, test_greens, test_cycles
        )
        green_derivatives = evaluate_grid(
            green_derivative_func, test_flows, test_greens, test_cycles
        )

        for index, lane in enumerate(active_lanes):
            fitted_expression = None
            if generic_low_decisive:
                low_demand_ok = generic_low_ok
                low_demand_limit = generic_low_limit
                low_demand_error = generic_low_error
            else:
                substitutions = {
                    symbols[name]: _compact_symbolic_number(
                        lane_parameters[lane][name]
                    )
                    for name in names
                }
                fitted_expression = parsed.subs(substitutions)
                low_demand_ok, low_demand_limit, low_demand_error = (
                    check_finite_nonnegative_low_demand_limit(
                        fitted_expression,
                        flow,
                        green,
                        cycle,
                    )
                )
            low_demand_limits[lane] = (
                None if low_demand_limit is None else str(low_demand_limit)
            )
            nonnegative_mask = np.isfinite(delay_values[index]) & (
                delay_values[index] >= -config.zero_tolerance
            )
            flow_mask = np.isfinite(flow_derivatives[index]) & (
                flow_derivatives[index] >= -config.derivative_tolerance
            )
            green_mask = np.isfinite(green_derivatives[index]) & (
                green_derivatives[index] <= config.derivative_tolerance
            )

            lane_rule_scores[lane]["R2_nondecreasing_flow"] = float(
                np.mean(flow_mask)
            )
            lane_rule_scores[lane]["R3_nonincreasing_green"] = float(
                np.mean(green_mask)
            )
            lane_rule_scores[lane]["R5_finite_nonnegative_low_demand_limit"] = float(
                low_demand_ok
            )
            lane_rule_scores[lane]["R6_nonnegative_delay"] = float(
                np.mean(nonnegative_mask)
            )

            try:
                if generic_zero_green_decisive:
                    limit_value = generic_zero_green_limit
                else:
                    if fitted_expression is None:
                        substitutions = {
                            symbols[name]: _compact_symbolic_number(
                                lane_parameters[lane][name]
                            )
                            for name in names
                        }
                        fitted_expression = parsed.subs(substitutions)
                    limit_expression = fitted_expression.subs(
                        {flow: positive_flow, cycle: positive_cycle}
                    )
                    if _proves_positive_infinite_green_exponential(
                        limit_expression, green
                    ):
                        limit_value = sp.oo
                    else:
                        limit_value = _symbolic_limit_with_timeout(
                            limit_expression,
                            green,
                            0,
                            timeout=config.symbolic_limit_timeout_seconds,
                        )
                zero_green_limits[lane] = str(limit_value)
                limit_ok = bool(
                    limit_value == sp.oo
                    or (
                        getattr(limit_value, "is_infinite", False) is True
                        and getattr(limit_value, "is_positive", False) is True
                    )
                )
                if not limit_ok and generic_zero_green_error:
                    lane_errors[lane].append(
                        "R7: zero-green limit undecidable: "
                        f"{generic_zero_green_error}"
                    )
            except Exception as exc:
                limit_ok = False
                zero_green_limits[lane] = None
                detail = generic_zero_green_error or str(exc)
                lane_errors[lane].append(
                    f"R7: zero-green limit undecidable: {detail}"
                )
            lane_rule_scores[lane]["R7_zero_green_limit"] = float(limit_ok)

            if not low_demand_ok:
                lane_errors[lane].append(f"R5: {low_demand_error}")
            for rule_name, mask in (
                ("R2: delay decreases with flow", flow_mask),
                ("R3: delay increases with green ratio", green_mask),
                ("R6: negative/non-finite delay", nonnegative_mask),
            ):
                failures = int(mask.size - np.count_nonzero(mask))
                if failures:
                    lane_errors[lane].append(
                        f"{rule_name} at {failures}/{mask.size} grid points"
                    )
            if not limit_ok and not any(
                message.startswith("R7:") for message in lane_errors[lane]
            ):
                lane_errors[lane].append(
                    "R7: delay does not tend to positive infinity as GR_phase -> 0+"
                )
    except Exception as exc:
        return _empty_physical_score(lanes, config, f"batch verifier failed: {exc}")

    def lane_min(rule_name: str) -> float:
        return float(min(lane_rule_scores[lane][rule_name] for lane in active_lanes))

    def lane_mean(rule_name: str) -> float:
        return float(np.mean([lane_rule_scores[lane][rule_name] for lane in active_lanes]))

    rule_scores = {
        "R1_required_variables": float(required_ok),
        "R2_nondecreasing_flow": lane_mean("R2_nondecreasing_flow"),
        "R3_nonincreasing_green": lane_mean("R3_nonincreasing_green"),
        "R4_time_dimension": float(unit_ok),
        "R5_finite_nonnegative_low_demand_limit": lane_min(
            "R5_finite_nonnegative_low_demand_limit"
        ),
        "R6_nonnegative_delay": lane_mean("R6_nonnegative_delay"),
        "R7_zero_green_limit": lane_min("R7_zero_green_limit"),
    }
    weights = np.asarray(config.rule_weights, dtype=float)
    values = np.asarray([rule_scores[name] for name in PHYSICAL_RULE_NAMES], dtype=float)
    score = float(np.dot(weights, values) / weights.sum())
    joint_pass = bool(np.all(values >= 1.0 - 1e-12))
    return PhysicalScoreResult(
        score=score,
        joint_pass=joint_pass,
        rule_scores=rule_scores,
        lane_rule_scores=lane_rule_scores,
        lane_errors={
            lane: list(dict.fromkeys(messages))
            for lane, messages in lane_errors.items()
            if messages
        },
        diagnostics={
            "rule_schema_id": PHYSICAL_RULE_SCHEMA_ID,
            "rule_order": list(PHYSICAL_RULE_NAMES),
            "grid_points_per_lane": len(points),
            "active_lanes": len(active_lanes),
            "low_demand_limits_by_lane": low_demand_limits,
            "zero_green_limits_by_lane": zero_green_limits,
            "symbolic_limit_evaluation": {
                "positive_parameter_proof": True,
                "r5_shared_proof_decisive": generic_low_decisive,
                "r7_shared_proof_decisive": generic_zero_green_decisive,
                "fallback_coefficient_significant_digits": 2,
                "note": (
                    "Full-precision coefficients are retained for predictions; "
                    "compact copies are used only for an inconclusive symbolic "
                    "limit fallback."
                ),
            },
            "symbolic_rules": {
                "R5": (
                    "limit as flow_lane -> 0+ exists, is finite, and is "
                    "non-negative for arbitrary 0 < GR_phase <= 1 and Cycle_Time > 0"
                ),
                "R7": "limit as GR_phase -> 0+ for symbolic positive flow and cycle is +infinity",
            },
            "note": "Grid pass fractions are diagnostics, not global proofs.",
        },
        config=config,
    )


def _zero_flow_boundary_test_points(
    config: PhysicalVerifierConfig,
) -> Tuple[Dict[str, float], ...]:
    """Representative (green, cycle) pairs at flow_lane=0 for R8."""
    return (
        {"flow_lane": 0.0, "GR_phase": 0.5, "Cycle_Time": 120.0},
        {
            "flow_lane": 0.0,
            "GR_phase": config.green_domain[0],
            "Cycle_Time": config.cycle_domain_seconds[0],
        },
        {
            "flow_lane": 0.0,
            "GR_phase": config.green_domain[1],
            "Cycle_Time": config.cycle_domain_seconds[1],
        },
    )


def _delay_within_operational_bounds(
    values: np.ndarray,
    config: PhysicalVerifierConfig,
) -> np.ndarray:
    """Finite, real delays within the operational magnitude cap used at prediction."""
    return np.isfinite(values) & (np.abs(values) <= config.max_delay_seconds)


def _is_positive_infinite_limit(limit_value) -> bool:
    return bool(
        limit_value == sp.oo
        or (
            getattr(limit_value, "is_infinite", False) is True
            and getattr(limit_value, "is_positive", False) is True
        )
    )


def _evaluate_zero_green_limit_numerical(
    evaluate_grid,
    active_lanes: List[str],
    config: PhysicalVerifierConfig,
) -> Tuple[Dict[str, bool], Dict[str, Optional[str]], Dict[str, List[str]], Dict[str, Any]]:
    """Evaluate R9 by substituting a near-zero green probe.

    At fixed positive flow and cycle, evaluate delay at
    ``zero_green_low_probe_green``. The lane passes when that value is
    positive infinite or strictly greater than ``zero_green_min_delay_seconds``.
    """
    probe_flow = config.zero_green_numerical_probe_flow
    probe_cycle = config.zero_green_numerical_probe_cycle
    low_green = config.zero_green_low_probe_green
    min_delay = config.zero_green_min_delay_seconds
    passed_by_lane = {lane: False for lane in active_lanes}
    limits_by_lane: Dict[str, Optional[str]] = {lane: None for lane in active_lanes}
    errors_by_lane: Dict[str, List[str]] = {lane: [] for lane in active_lanes}
    diagnostics: Dict[str, Any] = {
        "verification_type": "near_zero_green_absolute_delay",
        "probe_flow": probe_flow,
        "probe_cycle": probe_cycle,
        "low_green": low_green,
        "min_delay_seconds": min_delay,
        "delays_by_lane": {},
    }

    low_delays = evaluate_grid(
        [probe_flow], [low_green], [probe_cycle]
    )[:, 0]

    for index, lane in enumerate(active_lanes):
        delay_low = float(low_delays[index])
        diagnostics["delays_by_lane"][lane] = {
            "low_green_delay_seconds": delay_low,
        }

        if np.isposinf(delay_low) or (not np.isfinite(delay_low) and delay_low > 0):
            passed_by_lane[lane] = True
            limits_by_lane[lane] = "+infinity (numerical_probe)"
            continue

        if not np.isfinite(delay_low):
            errors_by_lane[lane].append(
                "R9: non-finite delay during zero-green numerical probe"
            )
            continue

        if delay_low > min_delay:
            passed_by_lane[lane] = True
            limits_by_lane[lane] = (
                f"{delay_low:.6g}s at GR={low_green:g} "
                f"(>{min_delay:g}s)"
            )
            continue

        errors_by_lane[lane].append(
            "R9: delay at near-zero green is not severe enough "
            f"({delay_low:.6g}s at GR={low_green:g}, need > {min_delay:g}s)"
        )

    return passed_by_lane, limits_by_lane, errors_by_lane, diagnostics


def _limit_remained_unevaluated(limit_value) -> bool:
    return bool(
        limit_value is None
        or isinstance(limit_value, sp.Limit)
        or (hasattr(limit_value, "has") and limit_value.has(sp.Limit))
    )


def _evaluate_zero_green_limit_symbolic_lane(
    fitted_expression: sp.Expr,
    flow: sp.Symbol,
    green: sp.Symbol,
    cycle: sp.Symbol,
    config: PhysicalVerifierConfig,
) -> Tuple[Optional[bool], Optional[str], str, Optional[str]]:
    """Return symbolic R9 decision for one lane with fitted coefficients.

    ``passed`` is ``True``/``False`` when the symbolic check is decisive, and
    ``None`` when the caller should fall back to the numerical probe.
    """
    positive_flow = sp.Symbol("_positive_flow", positive=True)
    positive_cycle = sp.Symbol("_positive_cycle", positive=True)
    limit_expression = fitted_expression.subs(
        {flow: positive_flow, cycle: positive_cycle}
    )

    if _proves_positive_infinite_green_exponential(
        limit_expression,
        green,
        timeout=min(
            SYMBOLIC_LIMIT_PROOF_TIMEOUT_SECONDS,
            config.symbolic_limit_timeout_seconds,
        ),
    ):
        return True, "+infinity", "green_exponential_polynomial_proof", None
    if _proves_positive_infinite_green_by_addends(
        limit_expression,
        green,
        timeout=min(
            SYMBOLIC_LIMIT_PROOF_TIMEOUT_SECONDS,
            config.symbolic_limit_timeout_seconds,
        ),
    ):
        return True, "+infinity", "addend_limit_proof", None

    try:
        limit_value = _symbolic_limit_with_timeout(
            limit_expression,
            green,
            0,
            timeout=config.symbolic_limit_timeout_seconds,
        )
        method = (
            "isolated_symbolic_limit"
            if _has_green_dependent_exponential(limit_expression, green)
            else "direct_symbolic_limit"
        )

        if _limit_remained_unevaluated(limit_value):
            return None, None, method, "symbolic limit remained unevaluated"

        if _is_positive_infinite_limit(limit_value):
            return True, str(limit_value), method, None
        if limit_value == -sp.oo:
            return (
                False,
                str(limit_value),
                method,
                "R9: delay tends to negative infinity as GR_phase -> 0+",
            )
        if getattr(limit_value, "is_finite", None) is True:
            return (
                False,
                str(limit_value),
                method,
                (
                    "R9: delay has a finite limit as GR_phase -> 0+ "
                    f"({limit_value})"
                ),
            )
        return None, str(limit_value), method, "symbolic limit sign was inconclusive"
    except TimeoutError as exc:
        return None, None, "symbolic_limit_timeout", str(exc)
    except Exception as exc:
        return None, None, "symbolic_limit_error", str(exc)


def _evaluate_zero_flow_boundary(
    evaluate_grid,
    active_lanes: List[str],
    config: PhysicalVerifierConfig,
) -> Tuple[Dict[str, bool], Dict[str, float], Dict[str, List[str]]]:
    """Evaluate R8 numerically with fitted lane coefficients."""
    passed_by_lane = {lane: True for lane in active_lanes}
    representative_delay: Dict[str, float] = {}
    errors_by_lane: Dict[str, List[str]] = {lane: [] for lane in active_lanes}

    for point in _zero_flow_boundary_test_points(config):
        delays = evaluate_grid(
            [point["flow_lane"]],
            [point["GR_phase"]],
            [point["Cycle_Time"]],
        )[:, 0]
        for index, lane in enumerate(active_lanes):
            delay = float(delays[index])
            if lane not in representative_delay:
                representative_delay[lane] = delay
            if not (
                np.isfinite(delay) and abs(delay) <= config.zero_tolerance
            ):
                passed_by_lane[lane] = False
                errors_by_lane[lane].append(
                    "R8: zero flow does not yield zero delay at "
                    f"GR_phase={point['GR_phase']}, "
                    f"Cycle_Time={point['Cycle_Time']}: got {delay:.6g}"
                )
    return passed_by_lane, representative_delay, errors_by_lane


def _evaluate_zero_green_limit(
    evaluate_grid,
    active_lanes: List[str],
    config: PhysicalVerifierConfig,
) -> Tuple[
    Dict[str, bool],
    Dict[str, Optional[str]],
    Dict[str, List[str]],
    Dict[str, Any],
]:
    """Evaluate R9 by absolute delay at a near-zero green probe.

    Substitute ``GR_phase = zero_green_low_probe_green`` at fixed positive flow
    and cycle; pass when the resulting delay exceeds
    ``zero_green_min_delay_seconds`` (default: 1e4 seconds).
    Symbolic limits are not used.
    """
    (
        passed_by_lane,
        limits_by_lane,
        errors_by_lane,
        numerical_diagnostics,
    ) = _evaluate_zero_green_limit_numerical(
        evaluate_grid,
        active_lanes,
        config,
    )
    methods_by_lane = {
        lane: "numerical_near_zero_green_magnitude" for lane in active_lanes
    }
    diagnostics: Dict[str, Any] = {
        "verification_type": "numerical_near_zero_green_magnitude",
        "numerical_probe": numerical_diagnostics,
        "methods_by_lane": methods_by_lane,
    }
    return passed_by_lane, limits_by_lane, errors_by_lane, diagnostics


def score_fitted_lanes_principlewise(
    expr: str,
    lane_parameters: Dict[str, Dict[str, float]],
    lanes: List[str],
    universal_features: List[str],
    config: Optional[PhysicalVerifierConfig] = None,
) -> PhysicalScoreResult:
    """Verify fitted delay expressions over the declared operational domain.

    This protocol combines operational-domain grid checks (R1--R4, R6--R7)
    with the Table-I endpoint checks for zero flow (R8) and a numerical-only
    zero-green severity check (R9): evaluate delay at a near-zero green probe
    and require it to exceed ``zero_green_min_delay_seconds``.
    Former R5 (domain magnitude cap) is omitted. Monotonicity uses a bounded
    hybrid hierarchy: direct derivative-sign inference first, deterministic
    sampling of the analytic derivatives when the sign is undecidable, and
    function finite differences only if derivative generation or
    numericalization reaches its hard timeout or fails.  The remaining
    numerical rules are strict binary checks over a deterministic
    operational-domain grid.  ``score`` remains the eight-rule diagnostic
    average; ``joint_pass`` is the all-or-nothing result used by the main
    experiment's binary fitness.
    """
    config = config or PhysicalVerifierConfig()
    lane_errors: Dict[str, List[str]] = {lane: [] for lane in lanes}
    lane_rule_scores: Dict[str, Dict[str, float]] = {
        lane: {name: 0.0 for name in PHYSICAL_RULE_NAMES}
        for lane in lanes
    }
    active_lanes = [lane for lane in lanes if lane in lane_parameters]
    for lane in lanes:
        if lane not in lane_parameters:
            lane_errors[lane].append("missing fitted parameters")
    if not active_lanes:
        return _empty_physical_score(lanes, config, "no fitted lane parameters")

    try:
        parsed, symbols = parse_symbolic_expression(expr)
    except Exception as exc:
        return _empty_physical_score(
            lanes, config, f"expression parsing failed: {exc}"
        )

    flow = symbols["flow_lane"]
    green = symbols["GR_phase"]
    cycle = symbols["Cycle_Time"]
    required_ok = {flow, green, cycle}.issubset(parsed.free_symbols)
    unit_ok, unit_error = check_unit_consistency(expr, universal_features)
    for lane in active_lanes:
        lane_rule_scores[lane]["R1_required_variables"] = float(required_ok)
        lane_rule_scores[lane]["R4_time_dimension"] = float(unit_ok)
        if not required_ok:
            lane_errors[lane].append(
                "R1: expression must use flow_lane, GR_phase, and Cycle_Time"
            )
        if not unit_ok:
            lane_errors[lane].append(f"R4: {unit_error}")

    names = coefficient_names(expr)
    missing_coefficients = {
        lane: [name for name in names if name not in lane_parameters[lane]]
        for lane in active_lanes
    }
    if any(missing_coefficients.values()):
        for lane, missing in missing_coefficients.items():
            if missing:
                lane_errors[lane].append(
                    "missing coefficients: " + ", ".join(missing)
                )
        return _empty_physical_score(
            lanes, config, "one or more lanes have incomplete fitted coefficients"
        )

    try:
        ordered_symbols = [flow, green, cycle] + [symbols[name] for name in names]
        value_func = sp.lambdify(ordered_symbols, parsed, modules="numpy")
        coefficient_arrays = [
            np.asarray(
                [lane_parameters[lane][name] for lane in active_lanes],
                dtype=float,
            )[:, None]
            for name in names
        ]

        def evaluate_function_grid(function, flow_values, green_values, cycle_values):
            def as_lane_grid(values):
                array = np.asarray(values, dtype=float)
                if array.ndim == 0:
                    array = array.reshape(1, 1)
                elif array.ndim == 1:
                    array = array[None, :]
                if array.shape[0] == 1:
                    array = np.broadcast_to(
                        array, (len(active_lanes), array.shape[1])
                    )
                if array.shape[0] != len(active_lanes):
                    raise ValueError("verification grid lane dimension mismatch")
                return array

            flow_array = as_lane_grid(flow_values)
            green_array = as_lane_grid(green_values)
            cycle_array = as_lane_grid(cycle_values)
            with np.errstate(all="ignore"):
                values = np.asarray(
                    function(
                        flow_array,
                        green_array,
                        cycle_array,
                        *coefficient_arrays,
                    ),
                    dtype=float,
                )
            return np.broadcast_to(values, flow_array.shape)

        def evaluate_grid(flow_values, green_values, cycle_values):
            return evaluate_function_grid(
                value_func, flow_values, green_values, cycle_values
            )

        # Exclude the historical exact-zero point.  Every retained point lies
        # inside the declared operational domain, including its boundaries.
        points = generate_universal_test_points(
            config.n_grid_samples,
            config.lhs_seed,
            config.flow_domain,
            config.green_domain,
            config.cycle_domain_seconds,
        )[1:]
        flows = np.asarray([point["flow_lane"] for point in points], dtype=float)
        greens = np.asarray([point["GR_phase"] for point in points], dtype=float)
        cycles = np.asarray([point["Cycle_Time"] for point in points], dtype=float)

        flow_span = config.flow_domain[1] - config.flow_domain[0]
        green_span = config.green_domain[1] - config.green_domain[0]
        flow_delta = flow_span * config.finite_difference_fraction
        green_delta = green_span * config.finite_difference_fraction
        flow_lower = np.maximum(flows - flow_delta / 2.0, config.flow_domain[0])
        flow_upper = np.minimum(flows + flow_delta / 2.0, config.flow_domain[1])
        green_lower = np.maximum(
            greens - green_delta / 2.0, config.green_domain[0]
        )
        green_upper = np.minimum(
            greens + green_delta / 2.0, config.green_domain[1]
        )

        base_values = evaluate_grid(flows, greens, cycles)
        flow_lower_values = evaluate_grid(flow_lower, greens, cycles)
        flow_upper_values = evaluate_grid(flow_upper, greens, cycles)
        green_lower_values = evaluate_grid(flows, green_lower, cycles)
        green_upper_values = evaluate_grid(flows, green_upper, cycles)
        flow_endpoint_low = evaluate_grid(
            np.full_like(flows, config.flow_domain[0]), greens, cycles
        )
        flow_endpoint_high = evaluate_grid(
            np.full_like(flows, config.flow_domain[1]), greens, cycles
        )
        green_endpoint_low = evaluate_grid(
            flows, np.full_like(greens, config.green_domain[0]), cycles
        )
        green_endpoint_high = evaluate_grid(
            flows, np.full_like(greens, config.green_domain[1]), cycles
        )

        evaluated_arrays = (
            base_values,
            flow_lower_values,
            flow_upper_values,
            green_lower_values,
            green_upper_values,
            flow_endpoint_low,
            flow_endpoint_high,
            green_endpoint_low,
            green_endpoint_high,
        )
        all_nonnegative = np.logical_and.reduce(
            [
                np.isfinite(values)
                & (values >= -config.zero_tolerance)
                for values in evaluated_arrays
            ]
        )
        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
            flow_local_difference = flow_upper_values - flow_lower_values
            green_local_difference = green_upper_values - green_lower_values
            flow_endpoint_difference = flow_endpoint_high - flow_endpoint_low
            green_endpoint_difference = green_endpoint_low - green_endpoint_high
        flow_monotone_finite_difference = (
            np.isfinite(flow_lower_values)
            & np.isfinite(flow_upper_values)
            & (flow_local_difference >= -config.derivative_tolerance)
        )
        green_monotone_finite_difference = (
            np.isfinite(green_lower_values)
            & np.isfinite(green_upper_values)
            & (green_local_difference <= config.derivative_tolerance)
        )

        # R2/R3: all SymPy derivative/sign work runs in a persistent child
        # process. The parent can terminate it at the hard deadline and already
        # has deterministic finite differences ready as the fallback.
        (
            flow_monotone,
            green_monotone,
            monotonicity_methods,
            derivative_elapsed,
            derivative_fallback_reason,
        ) = _hybrid_monotonicity_with_timeout(
            expr,
            active_lanes,
            lane_parameters,
            flows,
            greens,
            cycles,
            config.derivative_tolerance,
            config.symbolic_derivative_timeout_seconds,
            flow_monotone_finite_difference,
            green_monotone_finite_difference,
        )
        flow_responsive = (
            np.isfinite(flow_endpoint_low)
            & np.isfinite(flow_endpoint_high)
            & (flow_endpoint_difference > config.response_tolerance_seconds)
        )
        green_responsive = (
            np.isfinite(green_endpoint_low)
            & np.isfinite(green_endpoint_high)
            & (green_endpoint_difference > config.response_tolerance_seconds)
        )

        zero_flow_passed, zero_flow_delays, zero_flow_errors = (
            _evaluate_zero_flow_boundary(evaluate_grid, active_lanes, config)
        )
        (
            zero_green_passed,
            zero_green_limits,
            zero_green_errors,
            zero_green_diagnostics,
        ) = _evaluate_zero_green_limit(
            evaluate_grid,
            active_lanes,
            config,
        )

        response_diagnostics: Dict[str, Dict[str, float]] = {}
        for index, lane in enumerate(active_lanes):
            checks = {
                "R2_nondecreasing_flow": bool(np.all(flow_monotone[index])),
                "R3_nonincreasing_green": bool(np.all(green_monotone[index])),
                "R6_nonnegative_delay": bool(np.all(all_nonnegative[index])),
                "R7_operational_responsiveness": bool(
                    np.all(flow_responsive[index])
                    and np.all(green_responsive[index])
                ),
                "R8_zero_flow_boundary": bool(zero_flow_passed[lane]),
                "R9_zero_green_limit": bool(zero_green_passed[lane]),
            }
            for rule_name, passed in checks.items():
                lane_rule_scores[lane][rule_name] = float(passed)

            masks = {
                "R2: delay decreases as flow increases": flow_monotone[index],
                "R3: delay increases as green ratio increases": green_monotone[index],
                "R6: negative delay in the operational domain": all_nonnegative[index],
                "R7: insufficient flow response": flow_responsive[index],
                "R7: insufficient green-ratio response": green_responsive[index],
            }
            for message, mask in masks.items():
                failures = int(mask.size - np.count_nonzero(mask))
                if failures:
                    lane_errors[lane].append(
                        f"{message} at {failures}/{mask.size} grid points"
                    )
            for message in zero_flow_errors[lane]:
                lane_errors[lane].append(message)
            for message in zero_green_errors[lane]:
                lane_errors[lane].append(message)

            def finite_minimum(values) -> float:
                finite_values = np.asarray(values)[np.isfinite(values)]
                return (
                    float(np.min(finite_values))
                    if finite_values.size
                    else float("nan")
                )

            response_diagnostics[lane] = {
                "minimum_flow_endpoint_increase_seconds": finite_minimum(
                    flow_endpoint_difference[index]
                ),
                "minimum_green_endpoint_decrease_seconds": finite_minimum(
                    green_endpoint_difference[index]
                ),
            }
    except Exception as exc:
        return _empty_physical_score(
            lanes, config, f"operational-domain verifier failed: {exc}"
        )

    def lane_min(rule_name: str) -> float:
        return float(
            min(lane_rule_scores[lane][rule_name] for lane in active_lanes)
        )

    rule_scores = {
        rule_name: lane_min(rule_name)
        for rule_name in PHYSICAL_RULE_NAMES
    }
    weights = np.asarray(config.rule_weights, dtype=float)
    values = np.asarray(
        [rule_scores[name] for name in PHYSICAL_RULE_NAMES], dtype=float
    )
    score = float(np.dot(weights, values) / weights.sum())
    joint_pass = bool(np.all(values >= 1.0 - 1e-12))
    return PhysicalScoreResult(
        score=score,
        joint_pass=joint_pass,
        rule_scores=rule_scores,
        lane_rule_scores=lane_rule_scores,
        lane_errors={
            lane: list(dict.fromkeys(messages))
            for lane, messages in lane_errors.items()
            if messages
        },
        diagnostics={
            "rule_schema_id": PHYSICAL_RULE_SCHEMA_ID,
            "rule_order": list(PHYSICAL_RULE_NAMES),
            "verification_type": "hybrid_derivative_operational_domain_grid",
            "grid_points_per_lane": len(points),
            "active_lanes": len(active_lanes),
            "domain": {
                "flow_lane": list(config.flow_domain),
                "GR_phase": list(config.green_domain),
                "Cycle_Time_seconds": list(config.cycle_domain_seconds),
                "max_delay_seconds": config.max_delay_seconds,
            },
            "finite_difference_fraction": config.finite_difference_fraction,
            "symbolic_derivative_timeout_seconds": (
                config.symbolic_derivative_timeout_seconds
            ),
            "symbolic_derivative_elapsed_seconds": derivative_elapsed,
            "derivative_fallback_reason": derivative_fallback_reason,
            "monotonicity_methods_by_lane": monotonicity_methods,
            "response_tolerance_seconds": config.response_tolerance_seconds,
            "response_diagnostics_by_lane": response_diagnostics,
            "endpoint_limits_evaluated": True,
            "zero_flow_delay_by_lane": zero_flow_delays,
            "zero_green_limits_by_lane": zero_green_limits,
            "zero_green_symbolic_proof": zero_green_diagnostics,
            "zero_green_methods_by_lane": zero_green_diagnostics.get(
                "methods_by_lane", {}
            ),
            "symbolic_rules": {
                "R6": (
                    "delay must be finite and non-negative throughout the "
                    "operational domain grid"
                ),
                "R8": "delay equals zero when flow_lane=0 for representative green/cycle pairs",
                "R9": (
                    "at fixed positive flow and cycle, delay at "
                    "zero_green_low_probe_green must exceed "
                    "zero_green_min_delay_seconds (absolute magnitude check)"
                ),
            },
            "note": (
                "Monotonicity uses derivative-sign inference, sampled analytic "
                "derivatives, then finite differences as a failure/timeout "
                "fallback. Operational rules R1--R4 and R6--R7 remain binary "
                "over the declared domain (R5 magnitude cap removed); R8 is "
                "exact at flow_lane=0; R9 evaluates absolute delay at a "
                "near-zero green probe."
            ),
        },
        config=config,
    )


def validate_fitted_lanes_batch_legacy(
    expr: str,
    lane_parameters: Dict[str, Dict[str, float]],
    lanes: List[str],
    universal_features: List[str],
) -> Dict[str, str]:
    """Reproduce the historical pre-Table-I all-or-nothing checks.

    This compatibility function retains the former exact-zero-flow behavior for
    the explicitly named ``legacy_binary`` arm only.  Its result is not evidence
    under :data:`PHYSICAL_RULE_SCHEMA_ID`.
    """
    errors: Dict[str, List[str]] = {lane: [] for lane in lanes}
    missing = [lane for lane in lanes if lane not in lane_parameters]
    for lane in missing:
        errors[lane].append("missing fitted parameters")
    active_lanes = [lane for lane in lanes if lane in lane_parameters]
    if not active_lanes:
        return {lane: "; ".join(messages) for lane, messages in errors.items() if messages}

    unit_ok, unit_error = check_unit_consistency(expr, universal_features)
    if not unit_ok:
        for lane in active_lanes:
            errors[lane].append(f"Unit consistency check failed: {unit_error}")

    try:
        parsed, symbols = parse_symbolic_expression(expr)
        flow = symbols["flow_lane"]
        green = symbols["GR_phase"]
        cycle = symbols["Cycle_Time"]
        names = coefficient_names(expr)
        ordered_symbols = [flow, green, cycle] + [symbols[name] for name in names]
        value_func = sp.lambdify(ordered_symbols, parsed, modules="numpy")
        flow_derivative_func = sp.lambdify(
            ordered_symbols, sp.diff(parsed, flow), modules="numpy"
        )
        green_derivative_func = sp.lambdify(
            ordered_symbols, sp.diff(parsed, green), modules="numpy"
        )

        coefficient_arrays = [
            np.asarray([lane_parameters[lane][name] for lane in active_lanes], dtype=float)[:, None]
            for name in names
        ]

        def evaluate_grid(func, flow_values, green_values, cycle_values):
            def as_lane_grid(values):
                array = np.asarray(values, dtype=float)
                if array.ndim == 1:
                    array = array[None, :]
                if array.shape[0] == 1:
                    array = np.broadcast_to(array, (len(active_lanes), array.shape[1]))
                if array.shape[0] != len(active_lanes):
                    raise ValueError("validation grid lane dimension mismatch")
                return array

            flow_array = as_lane_grid(flow_values)
            green_array = as_lane_grid(green_values)
            cycle_array = as_lane_grid(cycle_values)
            with np.errstate(all="ignore"):
                values = np.asarray(
                    func(flow_array, green_array, cycle_array, *coefficient_arrays),
                    dtype=float,
                )
            return np.broadcast_to(values, flow_array.shape)

        zero_values = evaluate_grid(value_func, [0.0], [0.5], [90.0])[:, 0]
        # Exclude the exact zero-flow point from derivative checks; it is
        # validated above. Include physical boundary cases plus the full LHS grid.
        points = generate_universal_test_points(n_samples=128)[1:]
        test_flows = [point["flow_lane"] for point in points]
        test_greens = [point["GR_phase"] for point in points]
        test_cycles = [point["Cycle_Time"] for point in points]
        delay_values = evaluate_grid(value_func, test_flows, test_greens, test_cycles)
        flow_derivatives = evaluate_grid(
            flow_derivative_func, test_flows, test_greens, test_cycles
        )
        green_derivatives = evaluate_grid(
            green_derivative_func, test_flows, test_greens, test_cycles
        )

        for index, lane in enumerate(active_lanes):
            if not np.isfinite(zero_values[index]) or abs(zero_values[index]) > 1e-6:
                errors[lane].append("zero flow does not yield zero delay")

            if not np.all(np.isfinite(delay_values[index])):
                errors[lane].append("delay is non-finite in the validation domain")
            elif np.any(delay_values[index] < -1e-6):
                errors[lane].append("delay becomes negative")

            if not np.all(np.isfinite(flow_derivatives[index])):
                errors[lane].append("flow derivative is non-finite")
            elif np.any(flow_derivatives[index] < -1e-6):
                errors[lane].append("delay decreases as flow_lane increases")

            if not np.all(np.isfinite(green_derivatives[index])):
                errors[lane].append("green-ratio derivative is non-finite")
            elif np.any(green_derivatives[index] > 1e-6):
                errors[lane].append("delay increases as GR_phase increases")
    except Exception as exc:
        for lane in active_lanes:
            errors[lane].append(f"batch physical validation failed: {exc}")

    return {
        lane: "; ".join(dict.fromkeys(messages))
        for lane, messages in errors.items()
        if messages
    }


def validate_fitted_lanes_batch(
    expr: str,
    lane_parameters: Dict[str, Dict[str, float]],
    lanes: List[str],
    universal_features: List[str],
    config: Optional[PhysicalVerifierConfig] = None,
) -> Dict[str, str]:
    """Return strict finalized-Table-I joint-pass errors by lane.

    Kept under the original public function name for callers that request the
    current verifier.  Historical reproduction must opt in to
    :func:`validate_fitted_lanes_batch_legacy` explicitly.
    """
    result = score_fitted_lanes_principlewise(
        expr,
        lane_parameters,
        lanes,
        universal_features,
        config=config,
    )
    return {
        lane: "; ".join(dict.fromkeys(messages))
        for lane, messages in result.lane_errors.items()
        if messages
    }
