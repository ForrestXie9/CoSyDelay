from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, Optional

import numexpr as ne
import numpy as np
from scipy.optimize import minimize

from .config import (
    FIXED_PLAN_PATH,
    LANE_PHASE,
    LANES,
    LOST_TIME,
    MAX_CYCLE,
    MAX_GREEN,
    MIN_CYCLE,
    MIN_GREEN,
    MODEL_PATH,
    PHASE_LANES,
    PHASES,
    SATURATION_FLOW,
)


def _allocate_green(cycle: float, weights: Dict[str, float]) -> Dict[str, float]:
    effective = cycle - LOST_TIME
    values = np.full(len(PHASES), MIN_GREEN, dtype=float)
    remaining = effective - values.sum()
    active = set(range(len(PHASES)))
    raw_weights = np.asarray([max(weights.get(phase, 0.0), 1e-9) for phase in PHASES])
    while remaining > 1e-9 and active:
        indices = sorted(active)
        proportions = raw_weights[indices] / raw_weights[indices].sum()
        additions = remaining * proportions
        consumed = 0.0
        for index, addition in zip(indices, additions):
            capacity = MAX_GREEN - values[index]
            applied = min(capacity, float(addition))
            values[index] += applied
            consumed += applied
            if capacity - applied <= 1e-9:
                active.discard(index)
        if consumed <= 1e-12:
            break
        remaining -= consumed
    return {phase: float(values[index]) for index, phase in enumerate(PHASES)}


def _critical_ratios(rates: Dict[str, float]) -> Dict[str, float]:
    return {
        phase: max(rates[lane] / SATURATION_FLOW for lane in lanes)
        for phase, lanes in PHASE_LANES.items()
    }


def _timing_constraints(rates: Dict[str, float]):
    constraints = [
        {"type": "ineq", "fun": lambda values: LOST_TIME + np.sum(values) - MIN_CYCLE},
        {"type": "ineq", "fun": lambda values: MAX_CYCLE - LOST_TIME - np.sum(values)},
    ]
    ratios = _critical_ratios(rates)
    maximum_service_fraction = (MAX_CYCLE - LOST_TIME) / MAX_CYCLE
    # Enforce x<=0.98 only when the demand pattern is physically feasible within
    # the common cycle/green bounds. Oversaturated cases remain unconstrained.
    if sum(ratios.values()) / 0.98 <= maximum_service_fraction:
        for phase_index, phase in enumerate(PHASES):
            required_ratio = ratios[phase] / 0.98
            constraints.append({
                "type": "ineq",
                "fun": lambda values, index=phase_index, ratio=required_ratio: (
                    values[index] - (LOST_TIME + np.sum(values)) * ratio
                ),
            })
    return tuple(constraints)


def webster_plan(rates: Dict[str, float]) -> Dict[str, float]:
    ratios = _critical_ratios(rates)
    total = sum(ratios.values())
    if total >= 0.98:
        cycle = MAX_CYCLE
    else:
        cycle = float(np.clip((1.5 * LOST_TIME + 5.0) / max(1.0 - total, 1e-6), MIN_CYCLE, MAX_CYCLE))
    return _allocate_green(cycle, ratios)


@lru_cache(maxsize=1)
def _load_fixed_plan() -> Dict[str, float]:
    if not FIXED_PLAN_PATH.exists():
        raise FileNotFoundError(
            f"Frozen fixed plan not found: {FIXED_PLAN_PATH}; "
            "run prepare_intersection1_fixed_plan.py"
        )
    data = json.loads(FIXED_PLAN_PATH.read_text(encoding="utf-8"))
    core = {
        "plan": data["plan"],
        "training_mean_rates": data["training_mean_rates"],
    }
    actual = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if data.get("status") != "frozen" or actual != data.get("plan_sha256"):
        raise ValueError(f"Frozen fixed-plan checksum mismatch: {FIXED_PLAN_PATH}")
    plan = {phase: float(data["plan"][phase]) for phase in PHASES}
    if any(not np.isfinite(value) or value < MIN_GREEN or value > MAX_GREEN for value in plan.values()):
        raise ValueError(f"Invalid frozen fixed plan: {FIXED_PLAN_PATH}")
    return plan


def fixed_plan() -> Dict[str, float]:
    """Return the preset plan frozen from Intersection-1 training demand."""
    return _load_fixed_plan().copy()


def max_pressure_plan(
    queues: Dict[str, float],
    downstream_queues: Dict[str, float],
) -> Dict[str, float]:
    """Allocate a cyclic plan from upstream-minus-downstream queue pressure."""
    pressures = {
        phase: max(0.0, sum(
            max(
                queues.get(lane, 0.0) - downstream_queues.get(lane, 0.0),
                0.0,
            )
            for lane in lanes
        ))
        for phase, lanes in PHASE_LANES.items()
    }
    # Squaring emphasizes the maximum-pressure phase while _allocate_green
    # preserves a minimum green for every phase and prevents starvation.
    weights = {phase: max(value * value, 1e-6) for phase, value in pressures.items()}
    return _allocate_green(90.0, weights)


def _uniform_delay(rate: float, green: float, cycle: float) -> tuple[float, float, float, float]:
    if rate <= 0:
        return 0.0, 0.0, 0.0, 0.0
    ratio = green / cycle
    capacity = SATURATION_FLOW * ratio
    x = rate / max(capacity, 1e-9)
    y = rate / SATURATION_FLOW
    denominator = max(1.0 - min(1.0, x) * ratio, 1e-6)
    uniform = 0.5 * cycle * (1.0 - ratio) ** 2 / denominator
    return uniform, x, y, capacity


def hcm_delay(rate: float, green: float, cycle: float, analysis_hours: float = 0.25) -> float:
    uniform, x, _, capacity = _uniform_delay(rate, green, cycle)
    if rate <= 0:
        return 0.0
    k, upstream_filter = 0.5, 1.0
    incremental = 900.0 * analysis_hours * (
        (x - 1.0)
        + np.sqrt((x - 1.0) ** 2 + 8.0 * k * upstream_filter * x / max(capacity * analysis_hours, 1e-9))
    )
    return float(max(0.0, uniform + incremental))


def akcelik_delay(rate: float, green: float, cycle: float, analysis_hours: float = 0.25) -> float:
    if rate <= 0:
        return 0.0
    uniform, x, y, capacity = _uniform_delay(rate, green, cycle)
    saturation_per_second = SATURATION_FLOW / 3600.0
    x0 = 0.67 + saturation_per_second * green / 600.0
    overflow_queue = 0.0
    if x > x0:
        z = x - 1.0
        throughput = max(capacity * analysis_hours, 1e-9)
        overflow_queue = 0.25 * throughput * (
            z + np.sqrt(max(0.0, z * z + 12.0 * (x - x0) / throughput))
        )
    uniform_akcelik = 0.5 * cycle * (1.0 - green / cycle) ** 2 / max(1.0 - y, 1e-6)
    overflow_delay = overflow_queue * x / max(rate / 3600.0, 1e-9)
    return float(max(0.0, uniform_akcelik + overflow_delay))


def _optimize_plan(rates: Dict[str, float], delay_function) -> Dict[str, float]:
    initial_plan = webster_plan(rates)
    initial = np.asarray([initial_plan[phase] for phase in PHASES])

    def objective(greens):
        cycle = LOST_TIME + float(np.sum(greens))
        weighted = 0.0
        demand = 0.0
        for lane in LANES:
            rate = rates[lane]
            delay = delay_function(rate, greens[PHASES.index(LANE_PHASE[lane])], cycle)
            if not np.isfinite(delay):
                return 1e12
            weighted += rate * delay
            demand += rate
        return weighted / max(demand, 1e-9)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(MIN_GREEN, MAX_GREEN)] * len(PHASES),
        constraints=_timing_constraints(rates),
        options={"maxiter": 200, "ftol": 1e-8},
    )
    values = result.x if result.success and np.all(np.isfinite(result.x)) else initial
    return {phase: float(values[index]) for index, phase in enumerate(PHASES)}


@dataclass
class SymbolicModel:
    expression: str
    lane_parameters: Dict[str, Dict[str, float]]
    model_sha256: str
    schema_version: int = 1
    status: str = "frozen"

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "SymbolicModel":
        if not path.exists():
            raise FileNotFoundError(f"Frozen symbolic model not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported symbolic model schema_version={data.get('schema_version')!r}: {path}"
            )
        if data.get("status") != "frozen":
            raise ValueError(
                f"Formal control requires status='frozen'; fallback/unfrozen model rejected: {path}"
            )
        expression = data.get("expression")
        lane_parameters = data.get("lane_parameters")
        model_sha256 = data.get("model_sha256")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(f"Frozen symbolic model has no expression: {path}")
        if not isinstance(lane_parameters, dict) or set(lane_parameters) != set(LANES):
            missing = sorted(set(LANES) - set(lane_parameters or {}))
            extra = sorted(set(lane_parameters or {}) - set(LANES))
            raise ValueError(
                f"Frozen symbolic model lane set mismatch: missing={missing}, extra={extra}"
            )
        if not isinstance(model_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", model_sha256):
            raise ValueError(f"Frozen symbolic model has an invalid model_sha256: {path}")

        core = {"expression": expression, "lane_parameters": lane_parameters}
        payload = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != model_sha256:
            raise ValueError(f"Frozen symbolic model checksum mismatch: {path}")

        coefficient_names = set(re.findall(r"\ba\d+\b", expression))
        if not coefficient_names:
            raise ValueError(f"Frozen symbolic model expression has no fitted coefficients: {path}")
        normalized_parameters = {}
        for lane in LANES:
            parameters = lane_parameters[lane]
            if not isinstance(parameters, dict) or set(parameters) != coefficient_names:
                raise ValueError(
                    f"Frozen symbolic model coefficient set mismatch for {lane}: "
                    f"expected={sorted(coefficient_names)}, actual={sorted(parameters or {})}"
                )
            normalized_parameters[lane] = {}
            for name, value in parameters.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Non-numeric coefficient {lane}.{name}: {value!r}") from exc
                if not np.isfinite(numeric):
                    raise ValueError(f"Non-finite coefficient {lane}.{name}: {value!r}")
                normalized_parameters[lane][name] = numeric

        model = cls(
            expression=expression,
            lane_parameters=normalized_parameters,
            model_sha256=model_sha256,
            schema_version=1,
            status="frozen",
        )
        for lane in LANES:
            try:
                probe_context = {
                    "flow_lane": 100.0 / SATURATION_FLOW,
                    "GR_phase": 20.0 / 90.0,
                    "Cycle_Time": 90.0,
                    **model.lane_parameters[lane],
                }
                probe = float(ne.evaluate(model.expression, local_dict=probe_context))
            except Exception as exc:
                raise ValueError(f"Frozen symbolic model cannot evaluate lane {lane}: {exc}") from exc
            if not np.isfinite(probe) or probe < 0.0 or probe > 1e6:
                raise ValueError(f"Frozen symbolic model produces invalid probe delay for {lane}")
        return model

    def delay(self, lane: str, rate: float, green: float, cycle: float) -> float:
        context = {
            "flow_lane": rate / SATURATION_FLOW,
            "GR_phase": green / cycle,
            "Cycle_Time": cycle,
            **self.lane_parameters[lane],
        }
        value = float(ne.evaluate(self.expression, local_dict=context))
        return value if np.isfinite(value) and 0.0 <= value <= 1e6 else 1e6


def require_formal_matrix_model(path: Path = MODEL_PATH) -> SymbolicModel:
    """Reject structurally valid legacy models at formal experiment entrypoints."""
    model = SymbolicModel.load(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    selection = data.get("selection_protocol")
    evolution = data.get("evolution")
    validation = data.get("validation")
    provenance = data.get("provenance")
    if not all(isinstance(value, dict) for value in (
        selection, evolution, validation, provenance
    )):
        raise ValueError(
            "Formal SUMO control requires reviewer-matrix provenance; "
            f"legacy frozen model rejected: {path}"
        )
    physical = validation.get("current_principlewise_physical_recheck")
    source_code = provenance.get("source_code_sha256")

    def at_least(mapping, key, minimum):
        try:
            return int(mapping.get(key, 0)) >= minimum
        except (TypeError, ValueError):
            return False

    required = (
        selection.get("source_experiment") == "physics_score",
        selection.get("eligible_variant") == "principlewise",
        selection.get("eligible_intersection") == 1,
        selection.get("test_used_for_selection") is False,
        selection.get("test_opened_after_winner_fixed") is True,
        evolution.get("intersection_id") == 1,
        evolution.get("variant") == "principlewise",
        evolution.get("score_mode") == "principlewise",
        at_least(evolution, "generations", 10),
        at_least(evolution, "population", 20),
        at_least(evolution, "independent_runs", 10),
        at_least(evolution, "optimizer_restarts", 3),
        validation.get("matrix_complete") is True,
        validation.get("source_artifact_hashes_verified") is True,
        validation.get("history_complete") is True,
        isinstance(physical, dict) and physical.get("joint_pass") is True,
        isinstance(provenance.get("source_plan_sha256"), str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", provenance["source_plan_sha256"])),
        isinstance(provenance.get("source_config_sha256"), str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", provenance["source_config_sha256"])),
        isinstance(source_code, dict)
        and "expression_validation_lane.py" in source_code,
    )
    if not all(required):
        raise ValueError(
            "Formal SUMO control requires a complete audited principle-wise matrix "
            f"artifact; legacy/partial model rejected: {path}"
        )
    return model

def symbolic_plan(rates: Dict[str, float], model: SymbolicModel) -> Dict[str, float]:
    initial_plan = webster_plan(rates)
    initial = np.asarray([initial_plan[phase] for phase in PHASES])

    def objective(greens):
        cycle = LOST_TIME + float(np.sum(greens))
        total = 0.0
        demand = 0.0
        for lane in LANES:
            rate = rates[lane]
            delay = model.delay(lane, rate, greens[PHASES.index(LANE_PHASE[lane])], cycle)
            total += rate * delay
            demand += rate
        return total / max(demand, 1e-9)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(MIN_GREEN, MAX_GREEN)] * len(PHASES),
        constraints=_timing_constraints(rates),
        options={"maxiter": 200, "ftol": 1e-8},
    )
    values = result.x if result.success and np.all(np.isfinite(result.x)) else initial
    return {phase: float(values[index]) for index, phase in enumerate(PHASES)}


class Controller:
    def __init__(self, name: str, model_path: Optional[Path] = None):
        self.name = name
        self.model = SymbolicModel.load(model_path or MODEL_PATH) if name == "symbolic" else None
        if name == "fixed":
            _load_fixed_plan()

    @property
    def artifact_sha256(self) -> str:
        if self.model is not None:
            return self.model.model_sha256
        if self.name == "fixed":
            data = json.loads(FIXED_PLAN_PATH.read_text(encoding="utf-8"))
            return str(data["plan_sha256"])
        return ""

    def plan(
        self,
        rates: Dict[str, float],
        queues: Optional[Dict[str, float]] = None,
        downstream_queues: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if self.name == "webster":
            return webster_plan(rates)
        if self.name == "fixed":
            return fixed_plan()
        if self.name == "actuated":
            # The simulator owns the actuated state machine. This neutral plan
            # keeps the public Controller API valid without coupling actuated
            # control to the frozen fixed-time artifact.
            return _allocate_green(90.0, {phase: 1.0 for phase in PHASES})
        if self.name == "max_pressure":
            queues = queues or {lane: 0.0 for lane in LANES}
            downstream_queues = downstream_queues or {lane: 0.0 for lane in LANES}
            return max_pressure_plan(queues, downstream_queues)
        if self.name == "hcm":
            return _optimize_plan(rates, hcm_delay)
        if self.name == "akcelik":
            return _optimize_plan(rates, akcelik_delay)
        if self.name == "symbolic":
            return symbolic_plan(rates, self.model)
        if self.name == "rule":
            queues = queues or {lane: 0.0 for lane in LANES}
            weights = {
                phase: 1.0 + sum(queues.get(lane, 0.0) for lane in lanes)
                for phase, lanes in PHASE_LANES.items()
            }
            return _allocate_green(90.0, weights)
        raise KeyError(f"Unknown controller: {self.name}")
