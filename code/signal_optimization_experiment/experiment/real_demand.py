"""Load Intersection-1 movement flows without reading green times or labels."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List

from scipy.optimize import minimize_scalar


GMINI = Path(__file__).resolve().parents[2]
from .config import (
    LANES,
    LOST_TIME,
    MAX_CYCLE,
    MAX_GREEN,
    MIN_CYCLE,
    MIN_GREEN,
    PHASE_LANES,
    SATURATION_FLOW,
)


DATA_ROOT = GMINI.parent.parent / "Final_cosy_delay" / "jsonl_files"
DATA_FILES = {
    "train": DATA_ROOT / "Intersection_1_Train.jsonl",
    "validation": DATA_ROOT / "Intersection_1_Train.jsonl",
    "test": DATA_ROOT / "Intersection_1_Test.jsonl",
}
SOURCE_ROW_RANGES = {
    "train": (1, 1001),
    "validation": (1002, 1251),
    "test": (1, 503),
}
FLOW_COLUMN_ORDER = tuple(
    f"{approach}_{movement}"
    for approach in ("S", "E", "N", "W")
    for movement in ("L", "T", "R")
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def demand_hash(rates: Dict[str, float]) -> str:
    payload = json.dumps(rates, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def minimum_achievable_max_saturation(rates: Dict[str, float]) -> float:
    """Best theoretical max lane x under the common cycle/green bounds."""
    critical = {
        phase: max(rates[lane] / SATURATION_FLOW for lane in lanes)
        for phase, lanes in PHASE_LANES.items()
    }

    def best_for_cycle(cycle: float) -> float:
        effective_green = cycle - LOST_TIME
        lower = max(
            cycle * value / MAX_GREEN for value in critical.values()
        )
        if lower <= 0.0:
            return 0.0
        upper = max(
            cycle * value / MIN_GREEN for value in critical.values()
        )
        for _ in range(60):
            trial = (lower + upper) / 2.0
            required = [
                max(MIN_GREEN, cycle * value / trial)
                for value in critical.values()
            ]
            feasible = (
                all(value <= MAX_GREEN + 1e-12 for value in required)
                and sum(required) <= effective_green + 1e-12
            )
            if feasible:
                upper = trial
            else:
                lower = trial
        return upper

    optimized = minimize_scalar(
        best_for_cycle,
        bounds=(MIN_CYCLE, MAX_CYCLE),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(min(
        best_for_cycle(MIN_CYCLE),
        best_for_cycle(MAX_CYCLE),
        optimized.fun,
    ))


def capacity_regime(minimum_max_x: float) -> str:
    if minimum_max_x <= 0.98:
        return "feasible_x_le_0.98"
    if minimum_max_x <= 1.20:
        return "light_oversaturation_0.98_to_1.20"
    return "heavy_oversaturation_gt_1.20"


def _parse_flows_only(dialogue: str) -> Dict[str, float]:
    pattern = re.compile(
        r"(South|East|North|West)\s+Approach:.*?"
        r"Left Turn\s*=\s*(\d+),\s*Through\s*=\s*(\d+),\s*"
        r"Right Turn\s*=\s*(\d+)",
        re.DOTALL,
    )
    matches = pattern.findall(dialogue)
    if len(matches) != 4:
        raise ValueError(f"Expected four approach-flow records, found {len(matches)}")
    prefixes = {"South": "S", "East": "E", "North": "N", "West": "W"}
    rates = {}
    for name, left, through, right in matches:
        prefix = prefixes[name]
        if f"{prefix}_L" in rates:
            raise ValueError(f"Duplicate {name} approach")
        rates.update({
            f"{prefix}_L": float(left),
            f"{prefix}_T": float(through),
            f"{prefix}_R": float(right),
        })
    return rates


@dataclass(frozen=True)
class RealDemandRow:
    split: str
    row_id: int
    source_row_id: int
    rates: Dict[str, float]
    demand_sha256: str
    source_path: Path
    source_sha256: str


def load_intersection1_demands(split: str) -> List[RealDemandRow]:
    """Return only the 12 movement-flow fields from an Intersection-1 split.

    The JSONL ``summary`` field and the dialogue's green-time section are
    deliberately never accessed by the parser.
    """
    split = split.lower()
    if split not in DATA_FILES:
        raise ValueError(f"Unsupported split {split!r}; choose from {sorted(DATA_FILES)}")
    source = DATA_FILES[split]
    if not source.exists():
        raise FileNotFoundError(source)
    source_hash = _sha256_file(source)
    rows = []
    first_source_row, last_source_row = SOURCE_ROW_RANGES[split]
    with source.open("r", encoding="utf-8") as handle:
        for source_row_id, line in enumerate(handle, start=1):
            if source_row_id < first_source_row or source_row_id > last_source_row:
                continue
            row_id = source_row_id - first_source_row + 1
            item = json.loads(line)
            rates = _parse_flows_only(item["dialogue"])
            if set(rates) != set(LANES):
                raise ValueError(f"Unexpected movement set in row {row_id}")
            if any(value < 0.0 for value in rates.values()):
                raise ValueError(f"Negative movement flow in row {row_id}")
            rows.append(RealDemandRow(
                split=split,
                row_id=row_id,
                source_row_id=source_row_id,
                rates=rates,
                demand_sha256=demand_hash(rates),
                source_path=source,
                source_sha256=source_hash,
            ))
    if not rows:
        raise ValueError(f"No demand rows found in {source}")
    expected = last_source_row - first_source_row + 1
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} {split} rows, found {len(rows)}")
    return rows
