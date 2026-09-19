"""Freeze a coefficient-range profile from a Training-only paired screen.

The promotion gate is deliberately predeclared and split-safe: every paired
fit must improve R2 and reduce both RMSE and MAE relative to the default
profile.  Validation and Test files are never read by this utility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def promote(payload: dict, profile: str) -> dict:
    if payload.get("status") != "complete_training_only_paired_range_screen":
        raise ValueError("range screen is not a completed Training-only artifact")
    policy = str(payload.get("data_policy", "")).lower()
    if "training only" not in policy:
        raise ValueError("range screen does not declare Training-only fitting")
    if (
        "validation/test forbidden" not in policy
        and "validation/test/api forbidden" not in policy
        and "validation/test unused" not in policy
    ):
        raise ValueError("range screen does not explicitly forbid Validation/Test use")
    profiles = payload.get("profiles", {})
    summary = payload.get("summary", {})
    if profile not in profiles or profile == "default":
        raise ValueError("profile must be a non-default screened profile")
    stats = summary.get(profile, {})
    pairs = int(stats.get("pairs", 0))
    wins = int(stats.get("r2_wins", 0))
    gate = {
        "all_paired_r2_wins": pairs > 0 and wins == pairs,
        "mean_delta_r2_positive": float(stats.get("mean_delta_r2", 0.0)) > 0.0,
        "mean_delta_rmse_negative": float(stats.get("mean_delta_rmse", 0.0)) < 0.0,
        "mean_delta_mae_negative": float(stats.get("mean_delta_mae", 0.0)) < 0.0,
    }
    if not all(gate.values()):
        raise ValueError(f"profile failed the predeclared promotion gate: {gate}")
    return {
        "status": "frozen_training_only_range_profile",
        "profile": profile,
        "bounds": profiles[profile],
        "promotion_gate": gate,
        "screen_summary": stats,
        "screen_intersections": payload.get("intersections", []),
        "screen_seeds": payload.get("seeds", []),
        "validation_used": False,
        "test_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--profile", default="expanded_nonlinear")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.screen.resolve().read_text(encoding="utf-8"))
    frozen = promote(payload, args.profile)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(frozen, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
