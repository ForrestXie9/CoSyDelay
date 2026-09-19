"""Lightweight, predeclared V22 coefficient-range policy."""

V22_RANGE_PROFILE = {
    "scale": (0.001, 1000.0),
    "power_exponent": (0.01, 8.0),
    "exp_coefficient": (0.00001, 2.0),
}

V22_RANGE_PROFILE_ID = "expanded_nonlinear_training_screen_v1"

# Frozen from the paired Training-only confirmation screen.  This provenance
# is metadata only; the runtime never opens Validation/Test to choose bounds.
V22_RANGE_SELECTION_MANIFEST = {
    "screen": "range_confirm_i135_20260822_215946/range_screen.json",
    "promotion_rule": (
        "all paired Training R2 wins, positive mean delta R2, and negative "
        "mean delta RMSE and MAE"
    ),
    "pairs": 6,
    "r2_wins": 6,
    "mean_delta_r2": 0.0010955116284639226,
    "mean_delta_rmse": -0.024074506652753753,
    "mean_delta_mae": -0.03780756958155207,
    "validation_used": False,
    "test_used": False,
}
