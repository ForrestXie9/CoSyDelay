"""Core utilities for V22 vs classic-optimized generalization comparisons."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[2]
RANKING_DIR = HERE.parents[0] / "16_offline_timing_plan_ranking"
for path in (str(GMINI), str(RANKING_DIR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from constants import INTERSECTION_CONFIGS, SAT_FLOW  # noqa: E402
from optimization_lane import (  # noqa: E402
    calculate_approach_delays_from_universal,
    prepare_optimization_context,
)
from ranking_core import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_V22_ROOT,
    DelayModel,
    apply_greens,
    extract_greens,
    fit_classic_model,
    lanes_for,
    load_train_test,
    load_v22_model,
    predicted_weighted_delay,
    rank_of_minimum,
    weighted_observed_delay,
)

TABLE_VII_MODELS: Dict[Tuple[str, str], str] = {
    ("webster", "uniform"): (
        "1 * Cycle_Time * (1 - GR_phase)**2 / "
        "(2 * (1 - flow_lane * GR_phase))"
    ),
    ("webster", "variant1"): (
        "a1 * Cycle_Time * (1 - GR_phase)**a3 / "
        "(a2 * (1 - flow_lane * GR_phase))"
    ),
    ("hcm", "uniform"): (
        "0.5 * Cycle_Time * (1 - GR_phase)**2 / "
        "(1 - where(flow_lane > 1, 1, flow_lane) * GR_phase)"
    ),
    ("hcm", "variant1"): (
        "a1 * Cycle_Time * (1 - GR_phase)**a3 / "
        "(a2 - where(flow_lane > 1, 1, flow_lane) * GR_phase)"
    ),
    ("akcelik", "uniform"): (
        "1 * Cycle_Time * (1 - GR_phase)**2 / "
        "(2 * (1 - where(flow_lane > 1, 1, flow_lane) * GR_phase))"
    ),
    ("akcelik", "variant1"): (
        "a1 * Cycle_Time * (1 - GR_phase)**a3 / "
        "(a2 * (1 - flow_lane * GR_phase))"
    ),
}

SATURATION_STRATA = ("x_lt_0.85", "0.85_le_x_lt_1.00", "x_ge_1.00")
MERGED_DEMAND_STRATA = ("undersaturated", "saturated")
DEMAND_IMBALANCE_STRATA = ("balanced", "imbalanced")
TIMING_VISIBILITY_STRATA = ("seen_in_train", "unseen_in_train")

TABLE_VII_DISPLAY = {
    ("webster", "uniform"): "Webster",
    ("webster", "variant1"): "Webster-optimized",
    ("hcm", "uniform"): "HCM",
    ("hcm", "variant1"): "HCM-optimized",
    ("akcelik", "uniform"): "Akcelik",
    ("akcelik", "variant1"): "Akcelik-optimized",
}
TIMING_VISIBILITY_STRATA = ("seen_in_train", "unseen_in_train")
TIMING_RANGE_STRATA = ("timing_in_range", "timing_ood")


def saturation_stratum(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)) or value < 0:
        return "undefined"
    if value < 0.85:
        return "x_lt_0.85"
    if value < 1.0:
        return "0.85_le_x_lt_1.00"
    return "x_ge_1.00"


def merged_demand_stratum(value: Optional[float]) -> str:
    """Collapse near-capacity and oversaturated into one saturated bucket."""
    if value is None or not math.isfinite(float(value)) or value < 0:
        return "undefined"
    if value < 0.85:
        return "undersaturated"
    return "saturated"


def lane_controlling_gr(row: pd.Series, lane: str, intersection_id: int) -> float:
    config = INTERSECTION_CONFIGS[intersection_id]
    approach, movement = lane.split("_", 1)
    total = 0.0
    for phase, mapping in config["phase_movements"].items():
        if approach in mapping and movement in mapping[approach]:
            total += float(row[f"GR_{phase}"])
    return total


def sample_max_degree_of_saturation(row: pd.Series, intersection_id: int) -> float:
    lanes, _ = lanes_for(intersection_id)
    values: List[float] = []
    for lane in lanes:
        flow_col = f"flow_{lane}"
        if flow_col not in row.index:
            continue
        effective_green = lane_controlling_gr(row, lane, intersection_id)
        if effective_green <= 0.0:
            continue
        values.append(float(row[flow_col]) / effective_green)
    if not values:
        return float("nan")
    return float(max(values))


def timing_plan_key(row: pd.Series, intersection_id: int) -> str:
    config = INTERSECTION_CONFIGS[intersection_id]
    parts = [f"{float(row['Cycle_Time']):.1f}"]
    for phase in config["phases"]:
        parts.append(f"{phase}:{float(row[f'Green_{phase}']):.1f}")
    return "|".join(parts)


def fit_timing_ranges(train: pd.DataFrame, intersection_id: int) -> Dict[str, Any]:
    config = INTERSECTION_CONFIGS[intersection_id]
    cycles = train["Cycle_Time"].astype(float)
    green_ratio_ranges = {}
    for phase in config["phases"]:
        ratios = train[f"GR_{phase}"].astype(float)
        green_ratio_ranges[phase] = {
            "min": float(ratios.min()),
            "max": float(ratios.max()),
        }
    return {
        "cycle_min": float(cycles.min()),
        "cycle_max": float(cycles.max()),
        "green_ratio_ranges": green_ratio_ranges,
    }


def timing_outside_fit_range(row: pd.Series, ranges: Mapping[str, Any], intersection_id: int) -> bool:
    config = INTERSECTION_CONFIGS[intersection_id]
    cycle = float(row["Cycle_Time"])
    if cycle < ranges["cycle_min"] or cycle > ranges["cycle_max"]:
        return True
    for phase in config["phases"]:
        value = float(row[f"GR_{phase}"])
        limits = ranges["green_ratio_ranges"][phase]
        if value < limits["min"] or value > limits["max"]:
            return True
    return False


def dominant_approach_share(row: pd.Series, intersection_id: int) -> float:
    config = INTERSECTION_CONFIGS[intersection_id]
    totals = []
    for approach in config["approaches"]:
        totals.append(
            float(sum(row[f"{approach}_{movement}"] for movement in config["movements"][approach]))
        )
    total = sum(totals)
    if total <= 0.0:
        return float("nan")
    return float(max(totals) / total)


def demand_imbalance_stratum(share: float, threshold: float = 0.40) -> str:
    if not math.isfinite(share):
        return "undefined"
    return "imbalanced" if share >= threshold else "balanced"


def fit_table_vii_model(
    intersection_id: int,
    train: pd.DataFrame,
    family: str,
    variant: str,
    seed: int = 20260712,
) -> DelayModel:
    expression = TABLE_VII_MODELS[(family, variant)]
    if variant == "variant1":
        model = fit_classic_model(intersection_id, train, family, variant, seed=seed)
        return DelayModel(
            name=TABLE_VII_DISPLAY[(family, variant)],
            expression=model.expression,
            lane_parameters=model.lane_parameters,
            lanes=model.lanes,
            lane_to_approach=model.lane_to_approach,
            intersection_id=intersection_id,
        )
    lanes, mapping = lanes_for(intersection_id)
    parameters = {lane: {} for lane in lanes}
    return DelayModel(
        name=TABLE_VII_DISPLAY[(family, variant)],
        expression=expression,
        lane_parameters=parameters,
        lanes=tuple(lanes),
        lane_to_approach=mapping,
        intersection_id=intersection_id,
    )


def build_all_models(
    intersection_id: int,
    train: pd.DataFrame,
    v22_root: Path = DEFAULT_V22_ROOT,
    seed: int = 20260712,
) -> List[DelayModel]:
    models = [load_v22_model(intersection_id, v22_root)]
    for family in ("webster", "hcm", "akcelik"):
        for variant in ("uniform", "variant1"):
            models.append(
                fit_table_vii_model(
                    intersection_id, train, family, variant, seed=seed
                )
            )
    return models


def enrich_test_metadata(
    train: pd.DataFrame,
    test: pd.DataFrame,
    intersection_id: int,
    *,
    merge_near_into_saturated: bool = False,
) -> pd.DataFrame:
    ranges = fit_timing_ranges(train, intersection_id)
    train_keys = {timing_plan_key(train.iloc[index], intersection_id) for index in range(len(train))}
    rows = []
    for index in range(len(test)):
        row = test.iloc[index]
        max_x = sample_max_degree_of_saturation(row, intersection_id)
        share = dominant_approach_share(row, intersection_id)
        plan_key = timing_plan_key(row, intersection_id)
        demand_stratum = (
            merged_demand_stratum(max_x)
            if merge_near_into_saturated
            else saturation_stratum(max_x)
        )
        rows.append(
            {
                "row_index": index,
                "max_degree_of_saturation": max_x,
                "saturation_stratum": saturation_stratum(max_x),
                "demand_stratum": demand_stratum,
                "dominant_approach_share": share,
                "demand_imbalance_stratum": demand_imbalance_stratum(share),
                "timing_plan_key": plan_key,
                "timing_visibility": (
                    "unseen_in_train" if plan_key not in train_keys else "seen_in_train"
                ),
                "timing_range_stratum": (
                    "timing_ood"
                    if timing_outside_fit_range(row, ranges, intersection_id)
                    else "timing_in_range"
                ),
            }
        )
    return pd.DataFrame(rows)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {
            "n": 0.0,
            "r2": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "nonfinite_rate": 1.0,
        }
    truth = y_true[mask]
    pred = y_pred[mask]
    errors = pred - truth
    return {
        "n": float(mask.sum()),
        "r2": float(r2_score(truth, pred)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "nonfinite_rate": float(1.0 - mask.mean()),
    }


def predict_weighted_delays(
    frame: pd.DataFrame,
    model: DelayModel,
    row_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    lanes, mapping = lanes_for(model.intersection_id)
    if row_indices is None:
        eval_frame = frame
        positions = list(range(len(frame)))
    else:
        eval_frame = frame.iloc[list(row_indices)].reset_index(drop=True)
        positions = list(range(len(eval_frame)))
    prepared = prepare_optimization_context(
        eval_frame, list(lanes), mapping, model.intersection_id
    )
    try:
        predictions = calculate_approach_delays_from_universal(
            df=eval_frame,
            universal_expr=model.expression,
            lane_parameters=model.lane_parameters,
            lanes=list(model.lanes),
            lane_to_approach=model.lane_to_approach,
            intersection_id=model.intersection_id,
            prepared_context=prepared,
            strict=False,
        )
    except FloatingPointError:
        return np.full(len(eval_frame), np.nan, dtype=float)

    values = np.full(len(eval_frame), np.nan, dtype=float)
    config = INTERSECTION_CONFIGS[model.intersection_id]
    for position in positions:
        row = eval_frame.iloc[position]
        flows = {
            approach: float(
                sum(row[f"{approach}_{movement}"] for movement in config["movements"][approach])
            )
            for approach in config["approaches"]
        }
        total = sum(flows.values())
        if total <= 0.0:
            continue
        weighted = 0.0
        for approach, flow in flows.items():
            pred = float(predictions[approach][position])
            if not np.isfinite(pred):
                weighted = float("nan")
                break
            weighted += pred * flow
        else:
            values[position] = weighted / total
    return values


def evaluate_prediction_subset(
    test: pd.DataFrame,
    metadata: pd.DataFrame,
    model: DelayModel,
    mask: pd.Series,
) -> Dict[str, float]:
    if not mask.any():
        return regression_metrics(np.array([]), np.array([]))
    subset = metadata.loc[mask, "row_index"].astype(int).tolist()
    y_true = np.asarray(
        [weighted_observed_delay(test.iloc[index], model.intersection_id) for index in subset],
        dtype=float,
    )
    y_pred = predict_weighted_delays(test, model, subset)
    return regression_metrics(y_true, y_pred)


def evaluate_timing_pool_ranking(
    test: pd.DataFrame,
    train: pd.DataFrame,
    model: DelayModel,
    n_candidates: int,
    seed: int,
) -> Dict[str, float]:
    intersection_id = model.intersection_id
    rng = np.random.default_rng(seed + intersection_id)
    train_greens = [extract_greens(train.iloc[index], intersection_id) for index in range(len(train))]
    ranks: List[int] = []
    for sample_index in range(len(test)):
        sample = test.iloc[sample_index]
        true_plan = extract_greens(sample, intersection_id)
        candidates = [true_plan]
        choices = rng.choice(len(train_greens), size=n_candidates, replace=True)
        candidates.extend(train_greens[int(choice)] for choice in choices)
        scored = []
        for greens in candidates:
            row = apply_greens(sample, greens, intersection_id)
            scored.append(predicted_weighted_delay(row, model))
        if not np.all(np.isfinite(scored)):
            continue
        ranks.append(rank_of_minimum(scored))
    if not ranks:
        return {
            "n": 0.0,
            "hit_rate_at_1": float("nan"),
            "mean_rank": float("nan"),
            "median_rank": float("nan"),
        }
    ranks_array = np.asarray(ranks, dtype=float)
    return {
        "n": float(len(ranks)),
        "hit_rate_at_1": float(np.mean(ranks_array == 1.0)),
        "mean_rank": float(np.mean(ranks_array)),
        "median_rank": float(np.median(ranks_array)),
    }
