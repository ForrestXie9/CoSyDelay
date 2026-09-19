"""Training/reporting metrics used by the clean formal method."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def score_predictions(
    targets: Mapping[str, pd.Series | np.ndarray],
    predictions: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    """Return paper-selection and unbiased reporting metrics.

    The paper implementation clips each approach R2 at zero before taking its
    macro mean for evolutionary fitness.  Raw approach R2 and pooled errors
    remain separate reporting metrics for comparison with data-driven models.
    """
    by_approach: Dict[str, Dict[str, float]] = {}
    pooled_truth = []
    pooled_prediction = []
    for approach, target in targets.items():
        if approach not in predictions:
            continue
        truth = np.asarray(
            target.values if hasattr(target, "values") else target,
            dtype=float,
        )
        prediction = np.asarray(predictions[approach], dtype=float)
        valid = np.isfinite(truth) & np.isfinite(prediction)
        if np.count_nonzero(valid) < 2:
            continue
        truth = truth[valid]
        prediction = prediction[valid]
        pooled_truth.append(truth)
        pooled_prediction.append(prediction)
        raw_r2 = float(r2_score(truth, prediction))
        rmse = float(np.sqrt(mean_squared_error(truth, prediction)))
        mae = float(mean_absolute_error(truth, prediction))
        scale = max(float(np.std(truth)), 1e-12)
        by_approach[str(approach)] = {
            "r2": raw_r2,
            "r2_nonnegative": max(0.0, raw_r2),
            "rmse": rmse,
            "mae": mae,
            "normalized_rmse": rmse / scale,
            "normalized_mae": mae / scale,
            "rows": int(len(truth)),
        }
    if not by_approach:
        return {
            "macro_raw_r2": 0.0,
            "macro_nonnegative_r2": 0.0,
            "macro_rmse": float("inf"),
            "macro_mae": float("inf"),
            "pooled_rmse": float("inf"),
            "pooled_mae": float("inf"),
            "normalized_rmse": float("inf"),
            "normalized_mae": float("inf"),
            "by_approach": {},
        }
    records = list(by_approach.values())
    pooled_truth_array = np.concatenate(pooled_truth)
    pooled_prediction_array = np.concatenate(pooled_prediction)
    return {
        "macro_raw_r2": float(np.mean([item["r2"] for item in records])),
        "macro_nonnegative_r2": float(
            np.mean([item["r2_nonnegative"] for item in records])
        ),
        "macro_rmse": float(np.mean([item["rmse"] for item in records])),
        "macro_mae": float(np.mean([item["mae"] for item in records])),
        "pooled_rmse": float(
            np.sqrt(mean_squared_error(pooled_truth_array, pooled_prediction_array))
        ),
        "pooled_mae": float(
            mean_absolute_error(pooled_truth_array, pooled_prediction_array)
        ),
        "normalized_rmse": float(
            np.mean([item["normalized_rmse"] for item in records])
        ),
        "normalized_mae": float(
            np.mean([item["normalized_mae"] for item in records])
        ),
        "by_approach": by_approach,
    }
