from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Dict, Iterable, Optional, Set
import xml.etree.ElementTree as ET

import numpy as np

from .config import (
    ACTUATED_DETECTOR_DISTANCE,
    ACTUATED_GAP,
    GREEN_PHASE_INDEX,
    LANES,
    MAX_CLEARANCE_SECONDS,
    MAX_GREEN,
    MIN_GREEN,
    NETWORK_DIR,
    PHASE_LANES,
    PHASES,
    ROUTES,
    SIMULATION_SCHEMA_VERSION,
    SUMO_STEP_LENGTH,
    TLS_ID,
)
from .controllers import Controller
from .demand import scenario_rates


LOGICAL_TO_SUMO_LANE = {
    f"{direction}_{movement}": f"{direction}_in_{index}"
    for direction in "SNEW"
    for movement, index in (("R", 0), ("T", 1), ("L", 2))
}
LOGICAL_TO_OUTBOUND_LANE = {
    logical: f"{ROUTES[logical][1]}_{ {'R': 0, 'T': 1, 'L': 2}[logical[-1]] }"
    for logical in LANES
}


def _load_traci():
    try:
        import traci
        return traci
    except ImportError:
        candidates = []
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            candidates.append(Path(sumo_home) / "tools")
        candidates.append(Path(r"C:\Program Files (x86)\Eclipse\Sumo\tools"))
        for tools in candidates:
            if tools.exists():
                sys.path.insert(0, str(tools))
                import traci
                return traci
        raise


def _sumo_binary() -> str:
    found = shutil.which("sumo")
    if found:
        return found
    candidate = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe")
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("sumo executable not found")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(payload: Dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return cleaned or "unnamed"


def _tracked_hashes() -> Dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    tracked = [
        Path(__file__).resolve(),
        Path(__file__).with_name("controllers.py").resolve(),
        Path(__file__).with_name("config.py").resolve(),
        Path(__file__).with_name("demand.py").resolve(),
        root / "run_experiment.py",
    ]
    return {path.name: file_sha256(path) for path in tracked}


def _network_hashes() -> Dict[str, str]:
    names = ("intersection.net.xml", "intersection.tll.xml", "intersection.sumocfg")
    paths = [NETWORK_DIR / name for name in names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing SUMO network artifacts: {missing}")
    return {path.name: file_sha256(path) for path in paths}


def build_run_identity(
    controller_name: str,
    scenario: str,
    seed: int,
    duration_s: int,
    update_interval_s: int,
    route_file: Path,
    *,
    rates_override: Optional[Dict[str, float]] = None,
    clearance_limit_s: int = MAX_CLEARANCE_SECONDS,
    controller: Optional[Controller] = None,
) -> Dict:
    """Build a content-addressed identity before a simulation is launched."""
    if not route_file.exists():
        raise FileNotFoundError(route_file)
    controller = controller or Controller(controller_name)
    route_hash = file_sha256(route_file)
    network_hashes = _network_hashes()
    code_hashes = _tracked_hashes()
    model_hash = controller.model.model_sha256 if controller.model is not None else ""
    artifact_hash = controller.artifact_sha256
    rates_hash = (
        canonical_sha256({lane: float(rates_override[lane]) for lane in sorted(rates_override)})
        if rates_override is not None
        else ""
    )
    config_payload = {
        "simulation_schema_version": SIMULATION_SCHEMA_VERSION,
        "duration_s": int(duration_s),
        "update_interval_s": int(update_interval_s),
        "clearance_limit_s": int(clearance_limit_s),
        "route_sha256": route_hash,
        "rates_override_sha256": rates_hash,
        "controller_artifact_sha256": artifact_hash,
        "symbolic_model_sha256": model_hash,
        "network_files_sha256": network_hashes,
        "code_sha256": code_hashes,
    }
    config_hash = canonical_sha256(config_payload)
    run_payload = {
        "config_sha256": config_hash,
        "scenario": str(scenario),
        "controller": str(controller_name),
        "seed": int(seed),
    }
    run_hash = canonical_sha256(run_payload)
    model_tag = model_hash[:12] if model_hash else "none"
    artifact_tag = artifact_hash[:12] if artifact_hash else "none"
    tripinfo_filename = (
        f"trip_{_safe_filename(scenario)}_{_safe_filename(controller_name)}_"
        f"s{int(seed)}_d{int(duration_s)}_u{int(update_interval_s)}_"
        f"m{model_tag}_a{artifact_tag}_r{run_hash[:12]}.xml"
    )
    return {
        "simulation_schema_version": SIMULATION_SCHEMA_VERSION,
        "scenario": str(scenario),
        "controller": str(controller_name),
        "seed": int(seed),
        "duration_s": int(duration_s),
        "update_interval_s": int(update_interval_s),
        "clearance_limit_s": int(clearance_limit_s),
        "route_sha256": route_hash,
        "rates_override_sha256": rates_hash,
        "symbolic_model_sha256": model_hash,
        "controller_artifact_sha256": artifact_hash,
        "network_sha256": canonical_sha256(network_hashes),
        "protocol_sha256": canonical_sha256(code_hashes),
        "config_sha256": config_hash,
        "run_sha256": run_hash,
        "tripinfo_filename": tripinfo_filename,
        "config_payload": config_payload,
    }


@dataclass
class ActuatedPhaseTimer:
    """Minimum-green plus passage-gap timer, independent of SUMO APIs."""

    minimum_green: float = MIN_GREEN
    passage_gap: float = ACTUATED_GAP
    maximum_green: float = MAX_GREEN
    started_at: float = 0.0
    last_passage_at: Optional[float] = None
    seen_vehicle_ids: Set[str] = field(default_factory=set)

    def begin(self, time_s: float) -> None:
        self.started_at = float(time_s)
        self.last_passage_at = None
        self.seen_vehicle_ids.clear()

    def observe(self, time_s: float, detector_vehicle_ids: Iterable[str]) -> Set[str]:
        detected = {str(vehicle_id) for vehicle_id in detector_vehicle_ids}
        new_passages = detected - self.seen_vehicle_ids
        self.seen_vehicle_ids.update(detected)
        if new_passages:
            self.last_passage_at = float(time_s)
        return new_passages

    def desired_remaining(self, time_s: float) -> float:
        now = float(time_s)
        minimum_end = self.started_at + self.minimum_green
        maximum_end = self.started_at + self.maximum_green
        desired_end = minimum_end
        if self.last_passage_at is not None:
            desired_end = max(desired_end, self.last_passage_at + self.passage_gap)
        desired_end = min(desired_end, maximum_end)
        return max(0.0, desired_end - now)


def actuated_sumo_duration(physical_remaining: float) -> float:
    """Translate physical remaining green into SUMO's next-step switch semantics."""
    return max(0.0, float(physical_remaining) - SUMO_STEP_LENGTH)


def _detector_vehicle_ids(traci, phase_name: str) -> Set[str]:
    """Return new-passage candidates in a short stop-line detector zone."""
    detected: Set[str] = set()
    for logical_lane in PHASE_LANES[phase_name]:
        lane_id = LOGICAL_TO_SUMO_LANE[logical_lane]
        lane_length = float(traci.lane.getLength(lane_id))
        detector_start = max(0.0, lane_length - ACTUATED_DETECTOR_DISTANCE)
        for vehicle_id in traci.lane.getLastStepVehicleIDs(lane_id):
            try:
                position = float(traci.vehicle.getLanePosition(vehicle_id))
            except Exception:
                continue
            if position >= detector_start:
                detected.add(str(vehicle_id))
    return detected


def _trip_metrics(path: Path, demand_end_s: Optional[float] = None) -> Dict[str, float]:
    trips = ET.parse(path).getroot().findall("tripinfo")
    if not trips:
        return {
            "generated": 0,
            "tripinfo_vehicles": 0,
            "completed": 0,
            "tripinfo_unfinished": 0,
            "avg_time_loss": float("nan"),
            "avg_depart_delay": float("nan"),
            "avg_total_delay": float("nan"),
            "avg_waiting_time": float("nan"),
            "avg_duration": float("nan"),
            "total_time_loss": 0.0,
            "total_depart_delay": 0.0,
            "total_delay": 0.0,
            "departed_by_demand_end": 0,
            "completed_by_demand_end": 0,
            "clearance_completed_vehicles": 0,
            "demand_window_avg_time_loss": float("nan"),
            "demand_window_avg_depart_delay": float("nan"),
            "demand_window_avg_total_delay": float("nan"),
        }
    arrivals = np.asarray([float(trip.attrib.get("arrival", -1.0)) for trip in trips])
    departures = np.asarray([float(trip.attrib.get("depart", -1.0)) for trip in trips])
    completed_mask = arrivals >= 0.0
    time_loss = np.asarray([float(trip.attrib.get("timeLoss", 0.0)) for trip in trips])
    depart_delay = np.asarray([float(trip.attrib.get("departDelay", 0.0)) for trip in trips])
    waiting = np.asarray([float(trip.attrib.get("waitingTime", 0.0)) for trip in trips])
    duration = np.asarray([float(trip.attrib.get("duration", 0.0)) for trip in trips])
    total_delay = time_loss + depart_delay
    demand_departed = (
        int(np.sum((departures >= 0.0) & (departures <= demand_end_s)))
        if demand_end_s is not None else len(trips)
    )
    demand_completed = (
        int(np.sum(completed_mask & (arrivals <= demand_end_s)))
        if demand_end_s is not None else int(np.sum(completed_mask))
    )
    demand_completion_mask = (
        completed_mask & (arrivals <= demand_end_s)
        if demand_end_s is not None else completed_mask
    )
    def masked_mean(values):
        return (
            float(np.mean(values[demand_completion_mask]))
            if np.any(demand_completion_mask) else float("nan")
        )
    return {
        "generated": len(trips),  # backward-compatible alias
        "tripinfo_vehicles": len(trips),
        "completed": int(np.sum(completed_mask)),
        "tripinfo_unfinished": int(len(trips) - np.sum(completed_mask)),
        "avg_time_loss": float(np.mean(time_loss)),
        "avg_depart_delay": float(np.mean(depart_delay)),
        "avg_total_delay": float(np.mean(total_delay)),
        "avg_waiting_time": float(np.mean(waiting)),
        "avg_duration": float(np.mean(duration)),
        "total_time_loss": float(np.sum(time_loss)),
        "total_depart_delay": float(np.sum(depart_delay)),
        "total_delay": float(np.sum(total_delay)),
        "departed_by_demand_end": demand_departed,
        "completed_by_demand_end": demand_completed,
        "clearance_completed_vehicles": int(np.sum(completed_mask)) - demand_completed,
        "demand_window_avg_time_loss": masked_mean(time_loss),
        "demand_window_avg_depart_delay": masked_mean(depart_delay),
        "demand_window_avg_total_delay": masked_mean(total_delay),
    }


def _queue_statistics(values: Iterable[float], prefix: str) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        f"{prefix}avg_queue": float(np.mean(array)) if array.size else 0.0,
        f"{prefix}p95_queue": float(np.percentile(array, 95)) if array.size else 0.0,
        f"{prefix}max_queue": float(np.max(array)) if array.size else 0.0,
        f"{prefix}queue_samples": int(array.size),
    }


def _count_scheduled_vehicles(path: Path) -> int:
    return sum(1 for element in ET.parse(path).getroot() if element.tag == "vehicle")


def run_simulation(
    controller_name: str,
    scenario: str,
    seed: int,
    duration_s: int,
    route_file: Path,
    output_dir: Path,
    update_interval_s: int = 300,
    rates_override: Optional[Dict[str, float]] = None,
    *,
    scheduled_vehicles: Optional[int] = None,
    run_identity: Optional[Dict] = None,
    clearance_limit_s: int = MAX_CLEARANCE_SECONDS,
) -> Dict[str, float]:
    if duration_s <= 0 or update_interval_s <= 0 or clearance_limit_s <= 0:
        raise ValueError("duration, update interval, and clearance limit must be positive")
    if rates_override is not None:
        if set(rates_override) != set(LANES):
            raise ValueError("rates_override must contain exactly the 12 configured lanes")
        fixed_rates = {lane: float(rates_override[lane]) for lane in LANES}
        if any(not np.isfinite(value) or value < 0.0 for value in fixed_rates.values()):
            raise ValueError("rates_override contains an invalid lane flow")
    else:
        fixed_rates = None

    controller = Controller(controller_name)
    computed_identity = build_run_identity(
        controller_name,
        scenario,
        seed,
        duration_s,
        update_interval_s,
        route_file,
        rates_override=fixed_rates,
        clearance_limit_s=clearance_limit_s,
        controller=controller,
    )
    if run_identity is not None and run_identity.get("run_sha256") != computed_identity["run_sha256"]:
        raise ValueError("Supplied run_identity does not match the current files/configuration")
    identity = computed_identity

    output_dir.mkdir(parents=True, exist_ok=True)
    trip_file = output_dir / identity["tripinfo_filename"]
    if trip_file.exists():
        raise FileExistsError(
            f"Raw tripinfo already exists for run_sha256={identity['run_sha256']}: {trip_file}"
        )
    route_scheduled_count = _count_scheduled_vehicles(route_file)
    if scheduled_vehicles is not None and int(scheduled_vehicles) != route_scheduled_count:
        raise ValueError(
            f"scheduled_vehicles={scheduled_vehicles} does not match route count "
            f"{route_scheduled_count}: {route_file}"
        )
    scheduled_count = route_scheduled_count
    traci = _load_traci()
    command = [
        _sumo_binary(),
        "-c", str(NETWORK_DIR / "intersection.sumocfg"),
        "--route-files", str(route_file),
        "--additional-files", str(NETWORK_DIR / "intersection.tll.xml"),
        "--tripinfo-output", str(trip_file),
        "--tripinfo-output.write-unfinished", "true",
        "--seed", str(seed),
        "--end", str(duration_s + clearance_limit_s + 1),
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--xml-validation", "never",
    ]

    is_actuated = controller_name == "actuated"
    started = time.perf_counter()
    traci.start(command)
    all_queue_samples = []
    demand_queue_samples = []
    clearance_queue_samples = []
    cycle_records = []
    observed_green = {phase: [] for phase in PHASES}
    a_start_times = [0.0]
    phase_timer = ActuatedPhaseTimer()
    final_sim_time = 0.0
    residual_queue = 0.0
    residual_expected = scheduled_count
    clearance_complete = False
    try:
        traci.trafficlight.setProgram(TLS_ID, "experiment")
        current_rates = (
            fixed_rates.copy()
            if fixed_rates is not None
            else scenario_rates(scenario, 0, duration_s)
        )
        queues = {lane: 0.0 for lane in LANES}
        plan = controller.plan(current_rates, queues)
        if not is_actuated:
            cycle_records.append(plan.copy())
        traci.trafficlight.setPhase(TLS_ID, GREEN_PHASE_INDEX["A"])
        green_started_at = 0.0
        previous_phase = GREEN_PHASE_INDEX["A"]
        if is_actuated:
            phase_timer.begin(0.0)
            traci.trafficlight.setPhaseDuration(
                TLS_ID, actuated_sumo_duration(MIN_GREEN)
            )
        else:
            traci.trafficlight.setPhaseDuration(TLS_ID, plan["A"])
        next_update = float(update_interval_s)

        while True:
            traci.simulationStep()
            sim_time = float(traci.simulation.getTime())
            final_sim_time = sim_time
            queues = {
                logical: float(traci.lane.getLastStepHaltingNumber(sumo_lane))
                for logical, sumo_lane in LOGICAL_TO_SUMO_LANE.items()
            }
            residual_queue = float(sum(queues.values()))
            all_queue_samples.append(residual_queue)
            if sim_time <= duration_s:
                demand_queue_samples.append(residual_queue)
            else:
                clearance_queue_samples.append(residual_queue)

            phase_index = int(traci.trafficlight.getPhase(TLS_ID))
            if phase_index != previous_phase:
                if previous_phase in GREEN_PHASE_INDEX.values():
                    previous_name = next(
                        name for name, index in GREEN_PHASE_INDEX.items()
                        if index == previous_phase
                    )
                    observed_green[previous_name].append(sim_time - green_started_at)

                if phase_index in GREEN_PHASE_INDEX.values():
                    phase_name = next(
                        name for name, index in GREEN_PHASE_INDEX.items()
                        if index == phase_index
                    )
                    green_started_at = sim_time
                    if phase_name == "A":
                        a_start_times.append(sim_time)
                    if is_actuated:
                        phase_timer.begin(sim_time)
                        traci.trafficlight.setPhaseDuration(
                            TLS_ID, actuated_sumo_duration(MIN_GREEN)
                        )
                    else:
                        should_update = (
                            controller_name in {"rule", "max_pressure"}
                            or sim_time >= next_update
                        )
                        if phase_name == "A" and should_update:
                            current_rates = (
                                fixed_rates.copy()
                                if fixed_rates is not None
                                else scenario_rates(
                                    scenario,
                                    min(sim_time, duration_s - 1),
                                    duration_s,
                                )
                            )
                            if controller_name == "max_pressure":
                                downstream_queues = {
                                    lane: float(traci.lane.getLastStepHaltingNumber(
                                        LOGICAL_TO_OUTBOUND_LANE[lane]
                                    ))
                                    for lane in LANES
                                }
                                plan = controller.plan(current_rates, queues, downstream_queues)
                            else:
                                plan = controller.plan(current_rates, queues)
                            cycle_records.append(plan.copy())
                            if controller_name not in {"rule", "max_pressure"}:
                                next_update = sim_time + update_interval_s
                        traci.trafficlight.setPhaseDuration(TLS_ID, plan[phase_name])
                previous_phase = phase_index

            if is_actuated and phase_index in GREEN_PHASE_INDEX.values():
                phase_name = next(
                    name for name, index in GREEN_PHASE_INDEX.items()
                    if index == phase_index
                )
                phase_timer.observe(sim_time, _detector_vehicle_ids(traci, phase_name))
                remaining = phase_timer.desired_remaining(sim_time)
                commanded_duration = actuated_sumo_duration(remaining)
                if commanded_duration > 0.0:
                    traci.trafficlight.setPhaseDuration(TLS_ID, commanded_duration)

            residual_expected = int(traci.simulation.getMinExpectedNumber())
            if sim_time >= duration_s and residual_expected <= 0:
                clearance_complete = True
                break
            if sim_time >= duration_s + clearance_limit_s:
                clearance_complete = False
                break
    finally:
        traci.close()

    metrics = _trip_metrics(trip_file, demand_end_s=float(duration_s))
    tripinfo_count = int(metrics["tripinfo_vehicles"])
    completed_count = int(metrics["completed"])
    route_entry_rate = tripinfo_count / scheduled_count if scheduled_count else float("nan")
    completion_rate = completed_count / scheduled_count if scheduled_count else float("nan")
    fully_observed = bool(
        scheduled_count == tripinfo_count == completed_count
        and residual_expected == 0
        and clearance_complete
    )
    clearance_observation = max(0.0, final_sim_time - duration_s)
    clearance_seconds = clearance_observation if clearance_complete else float("nan")

    metrics.update({
        "scheduled_vehicles": scheduled_count,
        "route_entry_rate": route_entry_rate,
        "completion_rate": completion_rate,
        "unfinished": scheduled_count - completed_count,
        "fully_observed": fully_observed,
        "clearance_complete": clearance_complete,
        "clearance_seconds": clearance_seconds,
        "clearance_observation_seconds": clearance_observation,
        "residual_queue": residual_queue,
        "residual_expected_vehicles": residual_expected,
        "controller": controller_name,
        "scenario": scenario,
        "seed": seed,
        "duration_s": duration_s,
        "update_interval_s": update_interval_s,
        "clearance_limit_s": clearance_limit_s,
        "symbolic_model_sha256": identity["symbolic_model_sha256"],
        "controller_artifact_sha256": identity["controller_artifact_sha256"],
        "route_sha256": identity["route_sha256"],
        "rates_override_sha256": identity["rates_override_sha256"],
        "network_sha256": identity["network_sha256"],
        "protocol_sha256": identity["protocol_sha256"],
        "config_sha256": identity["config_sha256"],
        "run_sha256": identity["run_sha256"],
        "route_file": str(route_file.resolve()),
        "tripinfo_file": str(trip_file.resolve()),
        "simulation_schema_version": SIMULATION_SCHEMA_VERSION,
        "wall_seconds": time.perf_counter() - started,
        "simulation_end_s": final_sim_time,
        "throughput_veh_per_hour": completed_count / max(final_sim_time / 3600.0, 1e-9),
        "demand_window_throughput_veh_per_hour": (
            metrics["completed_by_demand_end"] / max(duration_s / 3600.0, 1e-9)
        ),
    })
    metrics.update(_queue_statistics(all_queue_samples, ""))
    metrics.update(_queue_statistics(demand_queue_samples, "demand_window_"))
    metrics.update(_queue_statistics(clearance_queue_samples, "clearance_"))

    if is_actuated:
        completed_cycles = np.diff(a_start_times)
        average_cycle = float(np.mean(completed_cycles)) if completed_cycles.size else float("nan")
    else:
        average_cycle = float(
            np.mean([24.0 + sum(record.values()) for record in cycle_records])
        )
    metrics["avg_cycle"] = average_cycle
    for phase in PHASES:
        values = observed_green[phase] if is_actuated else [record[phase] for record in cycle_records]
        metrics[f"avg_green_{phase}"] = float(np.mean(values)) if values else 0.0
        metrics[f"min_green_{phase}"] = float(np.min(values)) if values else float("nan")
        metrics[f"max_green_{phase}"] = float(np.max(values)) if values else float("nan")
    return metrics


def append_result(
    path: Path,
    result: Dict[str, float],
    *,
    key_fields: Optional[Iterable[str]] = None,
) -> None:
    """Append one schema-consistent row, optionally rejecting duplicate keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(result.keys())
    exists = path.exists()
    if exists:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fields = reader.fieldnames or []
            if existing_fields != fieldnames:
                raise ValueError(
                    f"CSV schema mismatch for {path}; existing={existing_fields}, new={fieldnames}"
                )
            if key_fields:
                keys = tuple(key_fields)
                missing = [field for field in keys if field not in result]
                if missing:
                    raise ValueError(f"Result is missing key fields: {missing}")
                wanted = tuple(str(result[field]) for field in keys)
                for row in reader:
                    if tuple(row[field] for field in keys) == wanted:
                        raise FileExistsError(
                            f"Duplicate result key {dict(zip(keys, wanted))} in {path}"
                        )
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(result)
