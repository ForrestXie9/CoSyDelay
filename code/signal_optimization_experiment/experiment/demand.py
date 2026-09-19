from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict
import xml.etree.ElementTree as ET

import numpy as np

from .config import LANES, ROUTES


TURN_SHARES = {"R": 0.15, "T": 0.70, "L": 0.15}


def _approach_rates(totals: Dict[str, float], shares=None) -> Dict[str, float]:
    shares = shares or TURN_SHARES
    return {
        f"{approach}_{movement}": total * shares[movement]
        for approach, total in totals.items()
        for movement in ("R", "T", "L")
    }


def scenario_rates(name: str, time_s: float, duration_s: float) -> Dict[str, float]:
    """Return demand in veh/h for every lane movement."""
    if name == "balanced":
        return _approach_rates({direction: 750.0 for direction in "SNEW"})
    if name == "asymmetric":
        return _approach_rates({"S": 1050.0, "N": 1050.0, "E": 450.0, "W": 450.0})
    if name == "near_saturation":
        return _approach_rates({direction: 1250.0 for direction in "SNEW"})
    if name == "turning_surge":
        return _approach_rates(
            {direction: 850.0 for direction in "SNEW"},
            shares={"R": 0.15, "T": 0.45, "L": 0.40},
        )
    if name == "switch":
        first_half = time_s < duration_s / 2
        ns, ew = (1100.0, 450.0) if first_half else (450.0, 1100.0)
        return _approach_rates({"S": ns, "N": ns, "E": ew, "W": ew})
    raise KeyError(f"Unknown demand scenario: {name}")


def _generate_route_file(
    path: Path,
    rate_provider: Callable[[int], Dict[str, float]],
    seed: int,
    duration_s: int,
) -> int:
    """Generate a shared stochastic arrival realization from lane rates."""
    rng = np.random.default_rng(seed)
    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        id="passenger",
        accel="2.6",
        decel="4.5",
        sigma="0.5",
        length="5.0",
        minGap="2.5",
        maxSpeed="13.89",
    )
    for lane, edges in ROUTES.items():
        ET.SubElement(root, "route", id=lane, edges=" ".join(edges))

    vehicle_index = 0
    for second in range(duration_s):
        rates = rate_provider(second)
        if set(rates) != set(LANES):
            missing = sorted(set(LANES) - set(rates))
            extra = sorted(set(rates) - set(LANES))
            raise ValueError(f"Lane-rate mismatch; missing={missing}, extra={extra}")
        for lane in LANES:
            rate = float(rates[lane])
            if not np.isfinite(rate) or rate < 0.0:
                raise ValueError(f"Invalid rate for {lane}: {rate}")
            arrivals = int(rng.poisson(rate / 3600.0))
            for _ in range(arrivals):
                ET.SubElement(
                    root,
                    "vehicle",
                    id=f"veh_{vehicle_index}",
                    type="passenger",
                    route=lane,
                    depart=str(second),
                    departLane="best",
                    departSpeed="max",
                )
                vehicle_index += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return vehicle_index


def generate_route_file(path: Path, scenario: str, seed: int, duration_s: int) -> int:
    """Generate identical stochastic arrivals for a synthetic scenario."""
    return _generate_route_file(
        path,
        lambda second: scenario_rates(scenario, second, duration_s),
        seed,
        duration_s,
    )


def generate_route_file_from_rates(
    path: Path,
    rates: Dict[str, float],
    seed: int,
    duration_s: int,
) -> int:
    """Generate arrivals from one observed, constant 12-movement demand row."""
    frozen_rates = {lane: float(value) for lane, value in rates.items()}
    return _generate_route_file(
        path,
        lambda _second: frozen_rates,
        seed,
        duration_s,
    )


def count_route_vehicles(path: Path) -> int:
    """Count scheduled vehicles in a cached route file for audit purposes."""
    return sum(1 for element in ET.parse(path).getroot() if element.tag == "vehicle")
