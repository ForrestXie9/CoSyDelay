"""Isolated seven-rule verifier for the recovered 2026-07-11 manuscript.

The shared runtime currently exposes an eight-rule operational verifier.  It
also retains the earlier seven-rule implementation as a private function, but
that function now reads the eight-rule global key order and raises a KeyError.
This module supplies the missing schema boundary without modifying the shared
module while the V15 confirmation is running.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
import multiprocessing
import queue
import threading
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import expression_validation_lane as shared
import numpy as np
import sympy as sp
from expression_rules import coefficient_names, parse_symbolic_expression


MANUSCRIPT_RULE_SCHEMA_ID = (
    "manuscript_table_i_2026_07_11_v5_fitted_limit_recheck"
)
MANUSCRIPT_RULE_NAMES = (
    "R1_required_variables",
    "R2_nondecreasing_flow",
    "R3_nonincreasing_green",
    "R4_time_dimension",
    "R5_finite_nonnegative_low_demand_limit",
    "R6_nonnegative_delay",
    "R7_zero_green_limit",
)


@dataclass(frozen=True)
class ManuscriptVerifierConfig:
    """Predeclared verifier domain and tolerances for the seven rules."""

    n_grid_samples: int = 128
    lhs_seed: int = 42
    flow_domain: Tuple[float, float] = (0.001, 2.0)
    green_domain: Tuple[float, float] = (0.02, 0.95)
    cycle_domain_seconds: Tuple[float, float] = (30.0, 240.0)
    zero_tolerance: float = 1e-6
    derivative_tolerance: float = 1e-6
    symbolic_limit_timeout_seconds: float = 8.0
    rule_weights: Tuple[float, ...] = (1.0,) * 7
    include_all_domain_corners: bool = True

    def __post_init__(self) -> None:
        if len(self.rule_weights) != len(MANUSCRIPT_RULE_NAMES):
            raise ValueError("rule_weights must contain exactly seven values")
        if any(float(weight) < 0.0 for weight in self.rule_weights):
            raise ValueError("rule weights must be non-negative")
        if sum(float(weight) for weight in self.rule_weights) <= 0.0:
            raise ValueError("at least one rule weight must be positive")
        for name, bounds in (
            ("flow_domain", self.flow_domain),
            ("green_domain", self.green_domain),
            ("cycle_domain_seconds", self.cycle_domain_seconds),
        ):
            if len(bounds) != 2 or float(bounds[0]) >= float(bounds[1]):
                raise ValueError(f"{name} must be an increasing pair")
        if self.n_grid_samples < 1:
            raise ValueError("n_grid_samples must be positive")
        if self.symbolic_limit_timeout_seconds <= 0.0:
            raise ValueError("symbolic_limit_timeout_seconds must be positive")
        if not self.include_all_domain_corners:
            raise ValueError("V16 requires all eight domain-corner anchors")


@dataclass
class ManuscriptPhysicalScoreResult:
    """Seven-component score with a separate strict joint-pass diagnostic."""

    score: float
    joint_pass: bool
    rule_scores: Dict[str, float]
    lane_rule_scores: Dict[str, Dict[str, float]]
    lane_errors: Dict[str, List[str]] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    config: ManuscriptVerifierConfig = field(
        default_factory=ManuscriptVerifierConfig
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_schema_id": MANUSCRIPT_RULE_SCHEMA_ID,
            "rule_order": list(MANUSCRIPT_RULE_NAMES),
            "score": float(self.score),
            "joint_pass": bool(self.joint_pass),
            "rule_scores": {
                name: float(self.rule_scores[name])
                for name in MANUSCRIPT_RULE_NAMES
            },
            "lane_rule_scores": {
                str(lane): {
                    name: float(scores[name]) for name in MANUSCRIPT_RULE_NAMES
                }
                for lane, scores in self.lane_rule_scores.items()
            },
            "lane_errors": {
                str(lane): list(messages)
                for lane, messages in self.lane_errors.items()
            },
            "diagnostics": dict(self.diagnostics),
            "config": asdict(self.config),
        }


_SHARED_SCHEMA_LOCK = threading.RLock()
_BASE_TEST_POINT_GENERATOR = shared.generate_universal_test_points


def _compatibility_r5_placeholder(*args, **kwargs):
    """Skip the retained generic R5 calculation that V16 never scores."""
    return True, sp.Integer(0), None


def _compatibility_r7_placeholder(*args, **kwargs) -> bool:
    """Skip the retained generic R7 calculation that V16 never scores."""
    return True


def _generate_v16_test_points(
    n_samples: int,
    seed: int,
    flow_domain: Tuple[float, float],
    green_domain: Tuple[float, float],
    cycle_domain: Tuple[float, float],
) -> Tuple[Dict[str, float], ...]:
    """Retain the LHS grid and add every joint domain corner exactly once."""
    points = list(
        _BASE_TEST_POINT_GENERATOR(
            n_samples,
            seed,
            flow_domain,
            green_domain,
            cycle_domain,
        )
    )
    keys = {
        (
            float(point["flow_lane"]),
            float(point["GR_phase"]),
            float(point["Cycle_Time"]),
        )
        for point in points
    }
    for flow_value, green_value, cycle_value in product(
        tuple(map(float, flow_domain)),
        tuple(map(float, green_domain)),
        tuple(map(float, cycle_domain)),
    ):
        key = (flow_value, green_value, cycle_value)
        if key in keys:
            continue
        points.append(
            {
                "flow_lane": flow_value,
                "GR_phase": green_value,
                "Cycle_Time": cycle_value,
            }
        )
        keys.add(key)
    return tuple(points)


def _check_exact_time_dimension(expression: str) -> Tuple[bool, str | None]:
    """Prove degree-one time homogeneity without replacing coefficients.

    The retained Table-I helper substitutes every coefficient by one before
    checking units.  That can incorrectly accept ``Cycle_Time**a1``.  V16
    keeps every coefficient symbolic and dimensionless, forbids Cycle_Time in
    an exp/log argument, and verifies ``f(k*C) = k*f(C)`` exactly.
    """
    try:
        parsed, symbols = parse_symbolic_expression(str(expression))
        cycle = symbols["Cycle_Time"]
        for function in tuple(parsed.atoms(sp.exp)) + tuple(parsed.atoms(sp.log)):
            if cycle in function.args[0].free_symbols:
                return False, "Cycle_Time occurs in an exp/log argument"
        scale = sp.Symbol("_time_scale", positive=True, finite=True)
        residual = sp.simplify(
            parsed.subs(cycle, scale * cycle) - scale * parsed
        )
        if residual == 0:
            return True, None
        return False, "expression is not homogeneous of degree one in Cycle_Time"
    except Exception as exc:
        return False, f"exact time-dimension check was undecidable: {exc}"


def _exact_fitted_expression(
    parsed: sp.Expr,
    symbols: Mapping[str, sp.Symbol],
    parameters: Mapping[str, float],
    names: Sequence[str],
) -> sp.Expr:
    """Substitute exact decimal rationals, never rounded coefficient copies."""
    substitutions = {
        symbols[name]: sp.Rational(str(float(parameters[name]))) for name in names
    }
    return parsed.subs(substitutions)


def _exact_fitted_analytic_fallback(
    fitted_expression: sp.Expr,
    flow: sp.Symbol,
    green: sp.Symbol,
    cycle: sp.Symbol,
    *,
    rule: str,
    timeout: float,
) -> Tuple[bool, str | None, str | None]:
    """Evaluate R5/R7 for one component with exact fitted coefficients.

    A proof under independent positive coefficient symbols is a sufficient
    pass certificate, but a failure of that generic expression is not a proof
    about every fitted coefficient tuple.  In particular, fitted equalities
    can change a limit by cancelling a denominator term.  This routine is
    therefore used whenever the generic certificate does not pass, including
    decisive generic failures and timeouts.
    """
    try:
        if rule == "R5":
            green_odds = sp.Symbol(
                "_green_odds", nonnegative=True, finite=True
            )
            positive_cycle = sp.Symbol(
                "_positive_cycle", positive=True, finite=True
            )
            expression = fitted_expression.subs(
                {
                    green: 1 / (1 + green_odds),
                    cycle: positive_cycle,
                }
            )
            limit_value = shared._symbolic_limit_with_timeout(
                expression, flow, 0, direction="+", timeout=timeout
            )
            passed = bool(
                not isinstance(limit_value, sp.Limit)
                and not limit_value.has(sp.Limit)
                and getattr(limit_value, "is_finite", None) is True
                and getattr(limit_value, "is_nonnegative", None) is True
            )
            error = None if passed else (
                "exact fitted low-demand limit is not established as finite "
                "and non-negative"
            )
        elif rule == "R7":
            positive_flow = sp.Symbol("_positive_flow", positive=True, finite=True)
            positive_cycle = sp.Symbol(
                "_positive_cycle", positive=True, finite=True
            )
            expression = fitted_expression.subs(
                {flow: positive_flow, cycle: positive_cycle}
            )
            limit_value = shared._symbolic_limit_with_timeout(
                expression, green, 0, direction="+", timeout=timeout
            )
            passed = bool(
                limit_value == sp.oo
                or (
                    getattr(limit_value, "is_infinite", False) is True
                    and getattr(limit_value, "is_positive", False) is True
                )
            )
            error = None if passed else (
                "exact fitted zero-green limit is not positive infinity"
            )
        else:
            raise ValueError(f"unknown analytic rule {rule}")
        return passed, str(limit_value), error
    except Exception as exc:
        return False, None, f"exact fitted {rule} limit failed closed: {exc}"


def _persistent_symbolic_limit_worker_loop(request_queue, result_queue) -> None:
    """Serve the same SymPy limit operation without per-candidate startup."""
    result_queue.put(("ready", None, True, None, None))
    while True:
        task = request_queue.get()
        if task is None:
            return
        task_id, expression, variable, point, direction = task
        try:
            result = sp.limit(expression, variable, point, dir=direction)
            result_queue.put(("result", task_id, True, result, None))
        except Exception as exc:
            result_queue.put(
                (
                    "result",
                    task_id,
                    False,
                    None,
                    f"{type(exc).__name__}: {exc}",
                )
            )


class PersistentSymbolicLimitEvaluator:
    """Reusable, independently terminable worker for bounded SymPy limits.

    This preserves the retained verifier's operation and timeout semantics.
    It changes only process lifetime: a healthy worker is reused, while a
    timed-out or failed worker is terminated and recreated for the next call.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process = None
        self._request_queue = None
        self._result_queue = None
        self._task_id = 0
        self._closed = False
        self._requests = 0
        self._completed = 0
        self._timeouts = 0
        self._worker_starts = 0
        self._startup_wall_seconds = 0.0
        self._request_wall_seconds = 0.0

    def _stop_unlocked(self) -> None:
        process = self._process
        request_queue = self._request_queue
        result_queue = self._result_queue
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
        self._process = None
        self._request_queue = None
        self._result_queue = None

    def _ensure_unlocked(self) -> None:
        if self._closed:
            raise RuntimeError("persistent symbolic limit evaluator is closed")
        if self._process is not None and self._process.is_alive():
            return
        self._stop_unlocked()
        if multiprocessing.current_process().daemon:
            raise TimeoutError(
                "isolated symbolic limit evaluation is unavailable in a daemon process"
            )
        context = multiprocessing.get_context("spawn")
        request_queue = context.Queue(maxsize=1)
        result_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=_persistent_symbolic_limit_worker_loop,
            args=(request_queue, result_queue),
            name="cosydelay-v16-symbolic-limit",
        )
        process.daemon = True
        started = time.perf_counter()
        process.start()
        try:
            message_type, _, success, _, error = result_queue.get(timeout=30.0)
            if message_type != "ready" or not success:
                raise RuntimeError(error or "symbolic limit worker did not become ready")
        except Exception:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            request_queue.close()
            result_queue.close()
            raise
        self._process = process
        self._request_queue = request_queue
        self._result_queue = result_queue
        self._worker_starts += 1
        self._startup_wall_seconds += time.perf_counter() - started

    def limit(
        self,
        expression: sp.Expr,
        variable: sp.Symbol,
        point,
        *,
        direction: str = "+",
        timeout: float = 8.0,
    ) -> sp.Expr:
        """Match ``_symbolic_limit_with_timeout`` with worker reuse."""
        if timeout <= 0.0:
            raise ValueError("symbolic limit timeout must be positive")
        with self._lock:
            self._ensure_unlocked()
            self._task_id += 1
            task_id = self._task_id
            self._requests += 1
            started = time.perf_counter()
            try:
                self._request_queue.put(
                    (task_id, expression, variable, point, direction),
                    timeout=timeout,
                )
                try:
                    message_type, result_task_id, success, result, error = (
                        self._result_queue.get(timeout=timeout)
                    )
                except queue.Empty as exc:
                    self._timeouts += 1
                    self._stop_unlocked()
                    raise TimeoutError(
                        f"symbolic limit exceeded {timeout:.1f}s"
                    ) from exc
                if message_type != "result" or result_task_id != task_id:
                    self._stop_unlocked()
                    raise RuntimeError(
                        "symbolic limit worker returned a mismatched result"
                    )
                if not success:
                    raise RuntimeError(error or "symbolic limit worker failed")
                self._completed += 1
                return result
            finally:
                self._request_wall_seconds += time.perf_counter() - started

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "implementation": "persistent_spawn_worker_same_sympy_limit",
                "requests": int(self._requests),
                "completed": int(self._completed),
                "timeouts": int(self._timeouts),
                "worker_starts": int(self._worker_starts),
                "startup_wall_seconds": float(self._startup_wall_seconds),
                "request_wall_seconds": float(self._request_wall_seconds),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop_unlocked()


def _shared_config(config: ManuscriptVerifierConfig):
    """Build the retained implementation's config under the seven-key schema."""
    return shared.PhysicalVerifierConfig(
        n_grid_samples=config.n_grid_samples,
        lhs_seed=config.lhs_seed,
        flow_domain=config.flow_domain,
        green_domain=config.green_domain,
        cycle_domain_seconds=config.cycle_domain_seconds,
        zero_tolerance=config.zero_tolerance,
        derivative_tolerance=config.derivative_tolerance,
        symbolic_limit_timeout_seconds=config.symbolic_limit_timeout_seconds,
        rule_weights=config.rule_weights,
    )


def score_fitted_lanes_manuscript_principlewise(
    expr: str,
    lane_parameters: Mapping[str, Mapping[str, float]],
    lanes: Sequence[str],
    universal_features: Sequence[str],
    config: ManuscriptVerifierConfig | None = None,
    limit_evaluator: PersistentSymbolicLimitEvaluator | None = None,
) -> ManuscriptPhysicalScoreResult:
    """Evaluate the manuscript's R1--R7 vector and equal-weight score.

    The compatibility lock is intentional.  The retained seven-rule
    implementation was coupled to module globals by an intervening eight-rule
    refactor.  We restore those globals before returning and copy the result
    into a schema-stable object, so serialization cannot later change when the
    shared module is used by V15.
    """
    config = config or ManuscriptVerifierConfig()
    with _SHARED_SCHEMA_LOCK:
        previous_names = shared.PHYSICAL_RULE_NAMES
        previous_schema = shared.PHYSICAL_RULE_SCHEMA_ID
        previous_limit_evaluator = shared._symbolic_limit_with_timeout
        previous_test_point_generator = shared.generate_universal_test_points
        previous_r5_checker = shared.check_finite_nonnegative_low_demand_limit
        previous_r7_prover = shared._proves_positive_infinite_green_exponential
        shared.PHYSICAL_RULE_NAMES = MANUSCRIPT_RULE_NAMES
        shared.PHYSICAL_RULE_SCHEMA_ID = MANUSCRIPT_RULE_SCHEMA_ID
        shared.generate_universal_test_points = _generate_v16_test_points
        # V16 scores fitted R5/R7 below.  Avoid redundant generic calculations
        # in the retained numerical compatibility helper, including a possible
        # extra 8-second timeout that cannot affect the V16 result.
        shared.check_finite_nonnegative_low_demand_limit = (
            _compatibility_r5_placeholder
        )
        shared._proves_positive_infinite_green_exponential = (
            _compatibility_r7_placeholder
        )
        if limit_evaluator is not None:
            shared._symbolic_limit_with_timeout = limit_evaluator.limit
        try:
            compatibility_config = _shared_config(config)
            observed = shared._score_fitted_lanes_legacy_table_i(
                str(expr),
                {
                    str(lane): {
                        str(name): float(value)
                        for name, value in parameters.items()
                    }
                    for lane, parameters in lane_parameters.items()
                },
                [str(lane) for lane in lanes],
                [str(feature) for feature in universal_features],
                config=compatibility_config,
            )
            rule_scores = {
                name: float(observed.rule_scores[name])
                for name in MANUSCRIPT_RULE_NAMES
            }
            lane_rule_scores = {
                str(lane): {
                    name: float(scores[name])
                    for name in MANUSCRIPT_RULE_NAMES
                }
                for lane, scores in observed.lane_rule_scores.items()
            }
            exact_time_ok, exact_time_error = _check_exact_time_dimension(
                str(expr)
            )
            for lane in lane_rule_scores:
                lane_rule_scores[lane]["R4_time_dimension"] = float(
                    exact_time_ok
                )
            analytic_fallback_errors: Dict[str, List[str]] = {
                str(lane): [] for lane in lane_rule_scores
            }
            analytic_fallback_limits: Dict[str, Dict[str, str | None]] = {
                str(lane): {} for lane in lane_rule_scores
            }
            # R5/R7 are properties of each fitted component, not of a generic
            # expression whose coefficient symbols are merely positive.  Even
            # a generic pass can be invalid on an exceptional fitted equality
            # (for example a1 == a2 can remove a denominator term).  Therefore
            # the generic results are diagnostics only and never supply the
            # V16 score.
            parsed, symbols = parse_symbolic_expression(str(expr))
            names = coefficient_names(str(expr))
            flow = symbols["flow_lane"]
            green = symbols["GR_phase"]
            cycle = symbols["Cycle_Time"]
            for lane in lane_rule_scores:
                fitted_expression = _exact_fitted_expression(
                    parsed, symbols, lane_parameters[lane], names
                )
                for rule, score_name in (
                    (
                        "R5",
                        "R5_finite_nonnegative_low_demand_limit",
                    ),
                    ("R7", "R7_zero_green_limit"),
                ):
                    passed, limit_value, error = (
                        _exact_fitted_analytic_fallback(
                            fitted_expression,
                            flow,
                            green,
                            cycle,
                            rule=rule,
                            timeout=config.symbolic_limit_timeout_seconds,
                        )
                    )
                    lane_rule_scores[lane][score_name] = float(passed)
                    analytic_fallback_limits[lane][rule] = limit_value
                    if error:
                        analytic_fallback_errors[lane].append(
                            f"{rule}: {error}"
                        )
            lane_errors = {
                str(lane): [
                    message
                    for message in messages
                    if not str(message).startswith("R4:")
                    and not str(message).startswith("R5:")
                    and not str(message).startswith("R7:")
                ]
                for lane, messages in observed.lane_errors.items()
            }
            if not exact_time_ok:
                for lane in lane_rule_scores:
                    lane_errors.setdefault(lane, []).append(
                        f"R4: {exact_time_error}"
                    )
            for lane, messages in analytic_fallback_errors.items():
                lane_errors.setdefault(lane, []).extend(messages)
            lane_errors = {
                lane: messages for lane, messages in lane_errors.items() if messages
            }
            # The generalized skeleton is instantiated once per declared
            # movement.  Preserve every movement/rule contribution instead of
            # collapsing an analytic rule to the worst movement before taking
            # the seven-rule mean.  With equal rule weights this is exactly the
            # mean of the per-movement seven-component scores.
            declared_lanes = [str(lane) for lane in lanes]
            weights = np.asarray(config.rule_weights, dtype=float)
            rule_scores = {
                name: float(
                    np.mean(
                        [lane_rule_scores[lane][name] for lane in declared_lanes]
                    )
                )
                for name in MANUSCRIPT_RULE_NAMES
            }
            movement_physical_scores = {
                lane: float(
                    np.dot(
                        weights,
                        np.asarray(
                            [
                                lane_rule_scores[lane][name]
                                for name in MANUSCRIPT_RULE_NAMES
                            ],
                            dtype=float,
                        ),
                    )
                    / weights.sum()
                )
                for lane in declared_lanes
            }
            physical_score = float(
                np.mean(list(movement_physical_scores.values()))
            )
            strict_joint_pass = bool(
                all(
                    lane_rule_scores[lane][name] >= 1.0 - 1e-12
                    for lane in declared_lanes
                    for name in MANUSCRIPT_RULE_NAMES
                )
            )
            diagnostics = dict(observed.diagnostics)
            # The retained compatibility helper saw deliberate no-op R5/R7
            # placeholders, so its legacy top-level display fields contain
            # placeholder values.  Replace them with the exact fitted limits
            # that actually supply the V16 scores; otherwise a passing positive
            # intercept could be misleadingly serialized as a zero limit.
            diagnostics["low_demand_limits_by_lane"] = {
                lane: analytic_fallback_limits[lane].get("R5")
                for lane in declared_lanes
            }
            diagnostics["zero_green_limits_by_lane"] = {
                lane: analytic_fallback_limits[lane].get("R7")
                for lane in declared_lanes
            }
            diagnostics["symbolic_limit_evaluation"] = {
                "legacy_generic_positive_parameter_checks_executed": False,
                "fitted_component_recheck": (
                    "exact_decimal_rational_full_fitted_coefficients"
                ),
                "fitted_component_limit_checks_executed": True,
                "rounded_coefficient_result_used_for_scoring": False,
                "timeout_or_undecidable_policy": "affected_rule_scores_zero",
            }
            diagnostics.update(
                {
                    "rule_schema_id": MANUSCRIPT_RULE_SCHEMA_ID,
                    "rule_order": list(MANUSCRIPT_RULE_NAMES),
                    "compatibility_source": (
                        "retained_private_seven_rule_implementation"
                    ),
                    "aggregation": (
                        "equal_mean_over_declared_movements_and_seven_rules"
                    ),
                    "movement_physical_scores": movement_physical_scores,
                    "strict_joint_pass_role": "diagnostic_only",
                    "r4_verification": (
                        "exact_symbolic_degree_one_time_homogeneity_with_"
                        "dimensionless_coefficients"
                    ),
                    "grid_anchor_scheme": "all_eight_joint_domain_corners",
                    "lhs_points_per_movement": int(config.n_grid_samples),
                    "unique_grid_points_per_movement": int(
                        observed.diagnostics.get("grid_points_per_lane", 0)
                    ),
                    "analytic_limit_fallback": {
                        "mode": (
                            "always_exact_decimal_rational_fitted_coefficients"
                        ),
                        "legacy_two_significant_digit_result_overridden": True,
                        "legacy_generic_positive_parameter_checks_executed": (
                            False
                        ),
                        "timeout_seconds": float(
                            config.symbolic_limit_timeout_seconds
                        ),
                        "exact_fitted_limits_by_movement": (
                            analytic_fallback_limits
                        ),
                    },
                }
            )
            if limit_evaluator is not None:
                diagnostics["symbolic_limit_worker"] = (
                    limit_evaluator.snapshot()
                )
            result = ManuscriptPhysicalScoreResult(
                score=physical_score,
                joint_pass=strict_joint_pass,
                rule_scores=rule_scores,
                lane_rule_scores=lane_rule_scores,
                lane_errors=lane_errors,
                diagnostics=diagnostics,
                config=config,
            )
        finally:
            shared.generate_universal_test_points = previous_test_point_generator
            shared._symbolic_limit_with_timeout = previous_limit_evaluator
            shared.check_finite_nonnegative_low_demand_limit = previous_r5_checker
            shared._proves_positive_infinite_green_exponential = (
                previous_r7_prover
            )
            shared.PHYSICAL_RULE_NAMES = previous_names
            shared.PHYSICAL_RULE_SCHEMA_ID = previous_schema
    return result
