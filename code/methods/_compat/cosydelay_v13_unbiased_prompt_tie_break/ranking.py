"""Training-only tie-break beneath the unchanged paper Fitness."""

import math
from typing import Mapping, Any


def _finite(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def paper_fitness_training_tie_key(individual: Mapping[str, Any]) -> tuple[float, ...]:
    """Rank by Fitness, then raw Training R2, then lower Training RMSE."""
    details = individual.get("evaluation_details", {}) or {}
    metrics = details.get("selection_metrics", {}) or {}
    return (
        _finite(individual.get("fitness"), -math.inf),
        _finite(metrics.get("macro_raw_r2"), -math.inf),
        -_finite(metrics.get("pooled_rmse", individual.get("train_rmse")), math.inf),
    )
