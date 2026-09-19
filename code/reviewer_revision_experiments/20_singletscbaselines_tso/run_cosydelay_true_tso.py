"""Run a small, leakage-free SUMO TSO comparison on SingleTSCBaselines.

The benchmark ships real-world intersection networks and route files.  This
adapter deliberately uses SUMO directly rather than the optional ``tshub``
wrapper so it can run in the existing CoSyDelay environment.  Each controller
sees only the route-derived movement demand before SUMO starts; performance is
then measured from SUMO tripinfo after the simulation.

This first adapter targets scenarios with exactly four three-lane approaches
(12 movement lanes), which match the CoSyDelay lane schema.  It is intentionally
separate from the existing Intersection-1 SUMO experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
# Portable release layout: this file lives under code/, while the selected
# SUMO scenarios live under data/.  The original repository used sibling
# paths inside the research workspace; keep the adapter self-contained here.
PACKAGE_ROOT = HERE.parents[2]
GMini = PACKAGE_ROOT / "code"
SOURCE_ROOT = PACKAGE_ROOT / "data" / "signal_optimization" / "SingleTSCBaselines"
SCENARIO_ROOT = SOURCE_ROOT / "junction_scenarios"
SIGNAL_EXPERIMENT = GMini / "signal_optimization_experiment"
MODEL_PATH = SIGNAL_EXPERIMENT / "models" / "symbolic_lane_model.json"
LANES = (
    "S_R", "S_T", "S_L", "N_R", "N_T", "N_L",
    "E_R", "E_T", "E_L", "W_R", "W_T", "W_L",
)
PHASES = ("A", "B", "C", "D")
TURN_TO_SUFFIX = {"r": "R", "s": "T", "l": "L"}
MODEL_LOST_TIME = 24.0
CUSTOM_YELLOW = 3.0
CUSTOM_CLEAR = 3.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sumo_binary() -> str:
    found = shutil.which("sumo")
    if found:
        return found
    candidates = (
        Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"),
        Path(r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"),
    )
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError("SUMO executable not found")


def _git_revision(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _compass(upstream_dx: float, upstream_dy: float) -> str:
    """Return the approach side from the junction toward the upstream node."""
    if abs(upstream_dx) >= abs(upstream_dy):
        return "E" if upstream_dx >= 0 else "W"
    return "N" if upstream_dy >= 0 else "S"


@dataclass
class Scenario:
    junction: str
    env_name: str
    scenario_dir: Path
    network: Path
    route: Path
    additional: tuple[Path, ...]
    tls_id: str
    green_phase_indices: tuple[int, ...]
    source_green_phase_indices: tuple[int, ...]
    movement_map: Dict[tuple[str, str], str]
    lane_map: Dict[str, str]
    logical_link_indices: Dict[str, tuple[int, ...]]
    rates: Dict[str, float]
    scheduled_vehicles: int
    max_depart: float


def _scenario_paths(junction: str, env_name: str):
    if "_" not in env_name:
        raise ValueError("env_name must be e.g. normal_low_density")
    difficulty, pattern = env_name.split("_", 1)
    scenario_dir = SCENARIO_ROOT / junction
    network = scenario_dir / "networks" / f"{difficulty}.net.xml"
    route = scenario_dir / "routes" / f"{pattern}.rou.xml"
    config = scenario_dir / f"{env_name}.sumocfg"
    if not network.exists() or not route.exists() or not config.exists():
        raise FileNotFoundError(
            f"Missing scenario files for {junction}/{env_name}: "
            f"network={network.exists()} route={route.exists()} config={config.exists()}"
        )
    additional = tuple(
        path for path in (
            scenario_dir / "add" / "map.poly.xml",
            scenario_dir / "add" / "e2.add.xml",
            scenario_dir / "add" / "tls.add.xml",
        ) if path.exists()
    )
    return scenario_dir, network, route, additional


def _load_net_metadata(network: Path):
    root = ET.parse(network).getroot()
    tls = root.find("tlLogic")
    if tls is None:
        raise ValueError(f"No traffic-light program found in {network}")
    tls_id = str(tls.attrib["id"])
    phases = tls.findall("phase")
    green_indices = tuple(
        index for index, phase in enumerate(phases)
        if "G" in phase.attrib.get("state", "") and "y" not in phase.attrib.get("state", "")
    )
    if len(green_indices) != 4:
        raise ValueError(
            f"CoSyDelay adapter requires exactly four protected green phases; "
            f"{network} has indices {green_indices}"
        )
    yellow_durations = tuple(float(phases[index].attrib["duration"]) for index in range(len(phases)))
    junction = next((node for node in root.findall("junction") if node.get("id") == tls_id), None)
    if junction is None:
        raise ValueError(f"TLS junction {tls_id!r} not found in {network}")
    inc_lanes = junction.get("incLanes", "").split()
    if len(inc_lanes) != 12:
        raise ValueError(
            f"CoSyDelay adapter requires exactly 12 incoming lanes; {network} has {len(inc_lanes)}"
        )
    return root, tls_id, green_indices, yellow_durations, junction


def _load_movement_mapping(
    root: ET.Element, tls_id: str,
) -> tuple[Dict[tuple[str, str], str], Dict[str, str], Dict[str, tuple[int, ...]]]:
    """Map each route movement to a CoSyDelay logical lane using SUMO connections."""
    edge_nodes = {node.get("id"): node for node in root.findall("edge") if node.get("id")}
    junction_node = next(node for node in root.findall("junction") if node.get("id") == tls_id)
    junction_x = float(junction_node.get("x", "0"))
    junction_y = float(junction_node.get("y", "0"))
    movement_map: Dict[tuple[str, str], str] = {}
    lane_map: Dict[str, str] = {}
    logical_link_indices: Dict[str, list[int]] = {lane: [] for lane in LANES}
    for connection in root.findall("connection"):
        if connection.get("tl") != tls_id:
            continue
        from_edge = connection.get("from")
        to_edge = connection.get("to")
        lane_index = connection.get("fromLane")
        turn = TURN_TO_SUFFIX.get(connection.get("dir", "").lower())
        edge = edge_nodes.get(from_edge)
        if not from_edge or not to_edge or lane_index is None or turn is None or edge is None:
            continue
        shape = edge.find("lane")
        if shape is None:
            continue
        points = shape.get("shape", "").split()
        if not points:
            continue
        try:
            upstream_x, upstream_y = map(float, points[0].split(",")[:2])
        except (TypeError, ValueError):
            continue
        side = _compass(upstream_x - junction_x, upstream_y - junction_y)
        logical = f"{side}_{turn}"
        lane_key = f"{from_edge}_{lane_index}"
        # A lane should have one principal movement in a single-intersection
        # benchmark. Keep the first deterministic connection if a lane has
        # multiple outgoing alternatives.
        lane_map.setdefault(lane_key, logical)
        movement_map.setdefault((from_edge, to_edge), logical)
        link_index = connection.get("linkIndex")
        if link_index is not None:
            try:
                logical_link_indices[logical].append(int(link_index))
            except ValueError:
                pass
    if set(lane_map.values()) != set(LANES):
        raise ValueError(
            "Could not derive a one-to-one 12-movement mapping; "
            f"found={sorted(lane_map.values())}"
        )
    normalized_links = {
        lane: tuple(sorted(set(indices))) for lane, indices in logical_link_indices.items()
    }
    return movement_map, lane_map, normalized_links


def _route_demand(route: Path, movement_map: Dict[tuple[str, str], str]) -> tuple[Dict[str, float], int, float]:
    root = ET.parse(route).getroot()
    counts = {lane: 0 for lane in LANES}
    scheduled = 0
    max_depart = 0.0
    for vehicle in root.findall("vehicle"):
        route_node = vehicle.find("route")
        if route_node is None:
            continue
        edges = route_node.get("edges", "").split()
        if len(edges) < 2:
            continue
        scheduled += 1
        try:
            max_depart = max(max_depart, float(vehicle.get("depart", "0")))
        except (TypeError, ValueError):
            pass
        logical = None
        for from_edge, to_edge in zip(edges, edges[1:]):
            logical = movement_map.get((from_edge, to_edge))
            if logical is not None:
                break
        if logical is not None:
            counts[logical] += 1
    if scheduled <= 0 or max_depart <= 0.0:
        raise ValueError(f"No usable vehicles found in {route}")
    horizon = max_depart + 1.0
    rates = {lane: count * 3600.0 / horizon for lane, count in counts.items()}
    return rates, scheduled, max_depart


def load_scenario(junction: str, env_name: str) -> Scenario:
    scenario_dir, network, route, additional = _scenario_paths(junction, env_name)
    root, tls_id, source_green_indices, _, _ = _load_net_metadata(network)
    movement_map, lane_map, logical_link_indices = _load_movement_mapping(root, tls_id)
    rates, scheduled, max_depart = _route_demand(route, movement_map)
    return Scenario(
        junction=junction,
        env_name=env_name,
        scenario_dir=scenario_dir,
        network=network,
        route=route,
        additional=additional,
        tls_id=tls_id,
        # The benchmark's original program is four green + four yellow
        # phases.  We replace it with a CoSyDelay-compatible four-group program
        # (green/yellow/clear for A--D), hence indices 0,3,6,9 below.
        green_phase_indices=(0, 3, 6, 9),
        source_green_phase_indices=source_green_indices,
        movement_map=movement_map,
        lane_map=lane_map,
        logical_link_indices=logical_link_indices,
        rates=rates,
        scheduled_vehicles=scheduled,
        max_depart=max_depart,
    )


def _load_cosydelay():
    if str(SIGNAL_EXPERIMENT) not in sys.path:
        sys.path.insert(0, str(SIGNAL_EXPERIMENT))
    from experiment.controllers import SymbolicModel, require_formal_matrix_model, symbolic_plan, webster_plan
    model = require_formal_matrix_model(MODEL_PATH)
    return model, symbolic_plan, webster_plan


def _read_tripinfo(path: Path, scheduled: int) -> dict:
    root = ET.parse(path).getroot()
    records = root.findall("tripinfo")
    completed = []
    for record in records:
        if record.get("vaporized"):
            continue
        try:
            float(record.get("arrival", "nan"))
            total_delay = float(record.get("timeLoss", "0")) + float(record.get("departDelay", "0"))
            completed.append({
                "time_loss": float(record.get("timeLoss", "0")),
                "waiting_time": float(record.get("waitingTime", "0")),
                "total_delay": total_delay,
            })
        except (TypeError, ValueError):
            continue
    def mean(key: str) -> float:
        values = [row[key] for row in completed]
        return float(np.mean(values)) if values else float("nan")
    return {
        "scheduled_vehicles": int(scheduled),
        "tripinfo_vehicles": int(len(records)),
        "completed_vehicles": int(len(completed)),
        "completion_rate": float(len(completed) / scheduled) if scheduled else float("nan"),
        "avg_time_loss_s": mean("time_loss"),
        "avg_waiting_time_s": mean("waiting_time"),
        "avg_total_delay_s": mean("total_delay"),
    }


def _run_controller(
    scenario: Scenario,
    controller_name: str,
    plan: Dict[str, float],
    seed: int,
    out_dir: Path,
    port: int | None = None,
) -> dict:
    import traci

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{scenario.junction}_{scenario.env_name}_{controller_name}_seed{seed}"
    tripinfo = out_dir / f"{tag}.tripinfo.xml"
    command = [
        _sumo_binary(), "-n", str(scenario.network), "-r", str(scenario.route),
        "--seed", str(seed), "--end", str(math.ceil(scenario.max_depart) + 900),
        "--tripinfo-output", str(tripinfo), "--tripinfo-output.write-unfinished", "true",
        "--no-step-log", "true", "--no-warnings", "true", "--xml-validation", "never",
    ]
    if scenario.additional:
        command += ["-a", ",".join(str(path) for path in scenario.additional)]
    green_indices = scenario.green_phase_indices
    phase_groups = {
        "A": ("S_R", "N_R"),
        "B": ("S_T", "S_L", "N_T", "N_L"),
        "C": ("E_R", "W_R"),
        "D": ("E_T", "E_L", "W_T", "W_L"),
    }
    n_links = max(index for values in scenario.logical_link_indices.values() for index in values) + 1
    phases = []
    for phase_name in PHASES:
        active = {
            index
            for lane in phase_groups[phase_name]
            for index in scenario.logical_link_indices[lane]
        }
        green_state = "".join("G" if index in active else "r" for index in range(n_links))
        yellow_state = "".join("y" if index in active else "r" for index in range(n_links))
        clear_state = "r" * n_links
        phases.extend([
            traci.trafficlight.Phase(float(plan[phase_name]), green_state),
            traci.trafficlight.Phase(CUSTOM_YELLOW, yellow_state),
            traci.trafficlight.Phase(CUSTOM_CLEAR, clear_state),
        ])
    logic = traci.trafficlight.Logic("cosydelay_tso", 0, 0, phases)
    start = time.perf_counter()
    # The default TraCI port is suitable for the original sequential runner.
    # Transfer-sensitivity jobs may run concurrently, so callers can provide
    # a dedicated port to avoid cross-process SUMO collisions.
    if port is None:
        traci.start(command)
    else:
        traci.start(command, port=int(port))
    previous_phase = None
    phase_name_by_index = {index: PHASES[pos] for pos, index in enumerate(green_indices)}
    try:
        traci.trafficlight.setProgramLogic(scenario.tls_id, logic)
        traci.trafficlight.setPhase(scenario.tls_id, green_indices[0])
        while True:
            traci.simulationStep()
            now = float(traci.simulation.getTime())
            current = int(traci.trafficlight.getPhase(scenario.tls_id))
            # The full CoSyDelay-compatible program already contains all durations;
            # this branch only records phase transitions for debugging.
            if current != previous_phase and current in phase_name_by_index:
                _ = phase_name_by_index[current]
            previous_phase = current
            if traci.simulation.getMinExpectedNumber() <= 0:
                break
            if now >= math.ceil(scenario.max_depart) + 900:
                break
    finally:
        traci.close()
    metrics = _read_tripinfo(tripinfo, scenario.scheduled_vehicles)
    return {
        "junction": scenario.junction,
        "env_name": scenario.env_name,
        "controller": controller_name,
        "seed": int(seed),
        "applied_plan": {key: float(value) for key, value in plan.items()},
        "tripinfo": str(tripinfo.resolve()),
        "runtime_s": float(time.perf_counter() - start),
        **metrics,
    }


def run(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.junction, args.env_name)
    model, symbolic_plan, webster_plan = _load_cosydelay()
    plans = {}
    if "cosydelay" in args.controllers:
        plans["cosydelay"] = symbolic_plan(scenario.rates, model)
    if "webster" in args.controllers:
        plans["webster"] = webster_plan(scenario.rates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        for controller in args.controllers:
            print(f"running {scenario.junction}/{scenario.env_name} {controller} seed={seed}", flush=True)
            rows.append(_run_controller(scenario, controller, plans[controller], seed, args.output_dir / "tripinfo"))
            print(
                f"  avg_total_delay={rows[-1]['avg_total_delay_s']:.3f}s "
                f"completion={rows[-1]['completed_vehicles']}/{rows[-1]['scheduled_vehicles']}",
                flush=True,
            )
    result_path = args.output_dir / "results.csv"
    fieldnames = sorted({key for row in rows for key in row if key != "applied_plan"}) + ["applied_plan_json"]
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {key: value for key, value in row.items() if key != "applied_plan"}
            record["applied_plan_json"] = json.dumps(row["applied_plan"], sort_keys=True)
            writer.writerow(record)
    metadata = {
        "status": "complete",
        "protocol": "true_sumo_tso_fixed_plan",
        "junction": scenario.junction,
        "env_name": scenario.env_name,
        "controllers": args.controllers,
        "seeds": args.seeds,
        "model_path": str(MODEL_PATH.resolve()),
        "model_sha256": _sha256(MODEL_PATH),
        "source_repo": str(SOURCE_ROOT.resolve()),
        "source_repo_revision": _git_revision(SOURCE_ROOT),
        "network": str(scenario.network.resolve()),
        "network_sha256": _sha256(scenario.network),
        "route": str(scenario.route.resolve()),
        "route_sha256": _sha256(scenario.route),
        "tls_id": scenario.tls_id,
        "green_phase_indices": scenario.green_phase_indices,
        "source_green_phase_indices": scenario.source_green_phase_indices,
        "custom_yellow_s": CUSTOM_YELLOW,
        "custom_clear_s": CUSTOM_CLEAR,
        "model_lost_time_s": MODEL_LOST_TIME,
        "rates_vph": scenario.rates,
        "scheduled_vehicles": scenario.scheduled_vehicles,
        "mapping": scenario.lane_map,
        "logical_link_indices": scenario.logical_link_indices,
        "results": str(result_path.resolve()),
        "test_labels_used_for_selection": False,
        "note": "SUMO replay outcome; this pilot targets one benchmark scenario and is not yet the full 12-junction matrix.",
    }
    manifest_path = args.output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(result_path.resolve())
    print(manifest_path.resolve())
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junction", default="Beijing_Gaojiaoyuan")
    parser.add_argument("--env-name", default="normal_low_density")
    parser.add_argument("--controllers", nargs="+", choices=("cosydelay", "webster"), default=["cosydelay", "webster"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
