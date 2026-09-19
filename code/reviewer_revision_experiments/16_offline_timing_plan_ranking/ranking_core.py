"""Core utilities for offline timing-plan ranking experiments."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from constants import INTERSECTION_CONFIGS, SAT_FLOW
from data_processing import load_dataset_flexible, preprocess_data_flexible
from optimization_lane import (
    DEFAULT_PARAM_BOUNDS,
    calculate_approach_delays_from_universal,
    fit_lane_parameters_to_approaches,
    prepare_optimization_context,
)
import optimization_lane as optimization_lane_module


CLASSIC_MODELS: Dict[Tuple[str, str], str] = {
    ("webster", "variant1"): (
        "a1 * Cycle_Time * (1 - GR_phase)**a3 / "
        "(a2 * (1 - flow_lane * GR_phase))"
    ),
    ("hcm", "variant1"): (
        "a1 * Cycle_Time * (1 - GR_phase)**a3 / "
        "(a2 - where(flow_lane > 1, 1, flow_lane) * GR_phase)"
    ),
    ("akcelik", "variant1"): (
        "a1 * Cycle_Time * (1 - GR_phase)**a3 / "
        "(a2 * (1 - flow_lane * GR_phase))"
    ),
}

BOUND_OVERRIDES = {
    ("hcm", "variant1"): {"a2": (1.0, 1000.0)},
}

DISPLAY_NAMES = {
    "cosydelay": "CoSyDelay",
    ("webster", "variant1"): "Webster-optimized",
    ("hcm", "variant1"): "HCM-optimized",
    ("akcelik", "variant1"): "Akcelik-optimized",
}

DEFAULT_CoSyDelay_ROOT = (
    Path(__file__).resolve().parents[2]
    / "methods"
    / "cosydelay"
    / "experiments"
    / "cosydelay_i1_i6_single_20260826_195254"
)
# In the release bundle the reproducible three-way split is the default.  A
# caller can still pass another directory explicitly to ``load_train_test``.
DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "locked_splits"
)


@dataclass(frozen=True)
class DelayModel:
    name: str
    expression: str
    lane_parameters: Dict[str, Dict[str, float]]
    lanes: Tuple[str, ...]
    lane_to_approach: Dict[str, str]
    intersection_id: int


def lanes_for(intersection_id: int) -> Tuple[List[str], Dict[str, str]]:
    config = INTERSECTION_CONFIGS[intersection_id]
    lanes = [
        f"{approach}_{movement}"
        for approach in config["approaches"]
        for movement in config["movements"][approach]
    ]
    mapping = {lane: lane.split("_", 1)[0] for lane in lanes}
    return lanes, mapping


def load_train_test(
    intersection_id: int, data_dir: Path = DEFAULT_DATA_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = preprocess_data_flexible(
        load_dataset_flexible(
            str(data_dir / f"Intersection_{intersection_id}_Train.jsonl"),
            intersection_id,
        ),
        intersection_id,
    ).reset_index(drop=True)
    test = preprocess_data_flexible(
        load_dataset_flexible(
            str(data_dir / f"Intersection_{intersection_id}_Test.jsonl"),
            intersection_id,
        ),
        intersection_id,
    ).reset_index(drop=True)
    return train, test


def approach_flows(row: pd.Series, intersection_id: int) -> Dict[str, float]:
    config = INTERSECTION_CONFIGS[intersection_id]
    return {
        approach: float(
            sum(row[f"{approach}_{movement}"] for movement in config["movements"][approach])
        )
        for approach in config["approaches"]
    }


def weighted_observed_delay(row: pd.Series, intersection_id: int) -> float:
    flows = approach_flows(row, intersection_id)
    total = sum(flows.values())
    if total <= 0.0:
        return float("nan")
    return float(
        sum(float(row[f"Delay_{approach}"]) * flows[approach] for approach in flows) / total
    )


def extract_greens(row: pd.Series, intersection_id: int) -> Dict[str, float]:
    config = INTERSECTION_CONFIGS[intersection_id]
    return {phase: float(row[f"Green_{phase}"]) for phase in config["phases"]}


def apply_greens(
    base_row: pd.Series, greens: Dict[str, float], intersection_id: int
) -> pd.Series:
    config = INTERSECTION_CONFIGS[intersection_id]
    row = base_row.copy()
    for phase in ("A", "B", "C", "D"):
        row[f"Green_{phase}"] = float(greens.get(phase, 0.0))
    cycle = float(sum(row[f"Green_{phase}"] for phase in config["phases"]) + config["cycle_offset"])
    row["Cycle_Time"] = cycle
    for phase in config["phases"]:
        row[f"GR_{phase}"] = row[f"Green_{phase}"] / cycle if cycle > 0 else 0.0
    for phase in ("A", "B", "C", "D"):
        if phase not in config["phases"]:
            row[f"GR_{phase}"] = 0.0
    return row


def flow_vector(row: pd.Series, lanes: Sequence[str]) -> np.ndarray:
    return np.asarray([float(row[lane]) for lane in lanes], dtype=float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@contextmanager
def _manual_coefficient_bounds():
    original = optimization_lane_module.build_parameter_bounds

    def patched(_expr: str, coef_names: list[str]):
        return [DEFAULT_PARAM_BOUNDS] * len(coef_names)

    optimization_lane_module.build_parameter_bounds = patched
    try:
        yield
    finally:
        optimization_lane_module.build_parameter_bounds = original


def fit_classic_model(
    intersection_id: int,
    train: pd.DataFrame,
    method: str,
    variant: str = "variant1",
    seed: int = 20260712,
) -> DelayModel:
    expression = CLASSIC_MODELS[(method, variant)]
    lanes, mapping = lanes_for(intersection_id)
    approaches = INTERSECTION_CONFIGS[intersection_id]["approaches"]
    targets = {approach: train[f"Delay_{approach}"] for approach in approaches}
    context = prepare_optimization_context(train, lanes, mapping, intersection_id)
    bounds_override = BOUND_OVERRIDES.get((method, variant))
    with _manual_coefficient_bounds():
        parameters = fit_lane_parameters_to_approaches(
            expression,
            train,
            lanes,
            mapping,
            targets,
            intersection_id,
            prepared_context=context,
            rng=np.random.default_rng(seed + intersection_id),
            n_restarts=3,
            coefficient_bounds_override=bounds_override,
        )
    if bounds_override:
        for lane in lanes:
            for name, (lower, _) in bounds_override.items():
                if name in parameters[lane]:
                    parameters[lane][name] = max(float(parameters[lane][name]), float(lower))
    return DelayModel(
        name=DISPLAY_NAMES[(method, variant)],
        expression=expression,
        lane_parameters=parameters,
        lanes=tuple(lanes),
        lane_to_approach=mapping,
        intersection_id=intersection_id,
    )


def load_cosydelay_model(intersection_id: int, search_root: Path = DEFAULT_CoSyDelay_ROOT) -> DelayModel:
    result_path = search_root / f"intersection_{intersection_id:02d}" / "run_01" / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    lanes, mapping = lanes_for(intersection_id)
    return DelayModel(
        name=DISPLAY_NAMES["cosydelay"],
        expression=str(payload["selected_expression"]),
        lane_parameters=payload["selected_parameters"],
        lanes=tuple(lanes),
        lane_to_approach=mapping,
        intersection_id=intersection_id,
    )


def predicted_weighted_delay(
    row: pd.Series,
    model: DelayModel,
    prepared_context: Optional[dict] = None,
) -> float:
    frame = pd.DataFrame([row])
    try:
        predictions = calculate_approach_delays_from_universal(
            df=frame,
            universal_expr=model.expression,
            lane_parameters=model.lane_parameters,
            lanes=list(model.lanes),
            lane_to_approach=model.lane_to_approach,
            intersection_id=model.intersection_id,
            prepared_context=prepared_context,
            strict=False,
        )
    except FloatingPointError:
        return float("nan")
    flows = approach_flows(row, model.intersection_id)
    total = sum(flows.values())
    if total <= 0.0:
        return float("nan")
    value = 0.0
    for approach, flow in flows.items():
        value += float(predictions[approach][0]) * flow
    return value / total


def build_models(
    intersection_id: int,
    train: pd.DataFrame,
    cosydelay_root: Path,
    classics: Sequence[Tuple[str, str]] = (
        ("hcm", "variant1"),
        ("akcelik", "variant1"),
        ("webster", "variant1"),
    ),
) -> List[DelayModel]:
    models = [load_cosydelay_model(intersection_id, cosydelay_root)]
    for method, variant in classics:
        models.append(fit_classic_model(intersection_id, train, method, variant))
    return models


def connected_clusters(similarity: np.ndarray, threshold: float) -> List[List[int]]:
    n = similarity.shape[0]
    visited = set()
    clusters: List[List[int]] = []
    for start in range(n):
        if start in visited:
            continue
        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            neighbors = np.where(similarity[node] >= threshold)[0]
            stack.extend(int(index) for index in neighbors if int(index) not in visited)
        if len(component) >= 3:
            clusters.append(sorted(component))
    return clusters


def perturb_greens(
    greens: Dict[str, float],
    intersection_id: int,
    rng: np.random.Generator,
    scale: float = 0.25,
) -> Dict[str, float]:
    config = INTERSECTION_CONFIGS[intersection_id]
    perturbed = {}
    for phase in config["phases"]:
        base = max(float(greens[phase]), 5.0)
        factor = float(rng.uniform(1.0 - scale, 1.0 + scale))
        perturbed[phase] = float(np.clip(base * factor, 5.0, 60.0))
    return perturbed


def pairwise_accuracy(true_values: np.ndarray, pred_values: np.ndarray) -> float:
    correct = 0
    total = 0
    for i in range(len(true_values)):
        for j in range(i + 1, len(true_values)):
            if true_values[i] == true_values[j]:
                continue
            total += 1
            if (true_values[i] - true_values[j]) * (pred_values[i] - pred_values[j]) > 0:
                correct += 1
    return float(correct / total) if total else float("nan")


def rank_of_minimum(values: Sequence[float]) -> int:
    order = np.argsort(values)
    return int(np.where(order == 0)[0][0]) + 1
