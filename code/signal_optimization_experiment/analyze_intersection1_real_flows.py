"""Analyze paired real-flow experiments with explicit completion auditing."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.results)
    required = {
        "controller", "demand_row_id", "seed", "scheduled_vehicles",
        "generated", "completed", "completion_rate", "total_time_loss",
        "avg_time_loss", "p95_queue",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing real-flow audit columns: {missing}")

    data["fully_observed"] = (
        (data["scheduled_vehicles"] == data["generated"])
        & (data["generated"] == data["completed"])
        & np.isclose(data["completion_rate"], 1.0)
    )
    summaries = []
    for controller, group in data.groupby("controller"):
        scheduled = float(group["scheduled_vehicles"].sum())
        summaries.append({
            "controller": controller,
            "runs": len(group),
            "fully_observed_runs": int(group["fully_observed"].sum()),
            "fully_observed_run_rate": float(group["fully_observed"].mean()),
            "vehicle_completion_rate": float(
                group["completed"].sum() / max(scheduled, 1.0)
            ),
            "mean_avg_time_loss_censored": float(group["avg_time_loss"].mean()),
            "mean_p95_queue": float(group["p95_queue"].mean()),
            "vehicle_weighted_time_loss_censored": float(
                group["total_time_loss"].sum() / max(scheduled, 1.0)
            ),
        })
    summary = pd.DataFrame(summaries).sort_values("controller")
    summary_path = args.results.with_name(f"{args.results.stem}_audit_summary.csv")
    summary.to_csv(summary_path, index=False)

    stratified_rows = []
    if "capacity_regime" in data.columns:
        for (regime, controller), group in data.groupby(["capacity_regime", "controller"]):
            scheduled = float(group["scheduled_vehicles"].sum())
            stratified_rows.append({
                "capacity_regime": regime,
                "controller": controller,
                "runs": len(group),
                "fully_observed_runs": int(group["fully_observed"].sum()),
                "fully_observed_run_rate": float(group["fully_observed"].mean()),
                "vehicle_completion_rate": float(
                    group["completed"].sum() / max(scheduled, 1.0)
                ),
                "mean_avg_time_loss_censored": float(group["avg_time_loss"].mean()),
                "mean_p95_queue": float(group["p95_queue"].mean()),
                "vehicle_weighted_time_loss_censored": float(
                    group["total_time_loss"].sum() / max(scheduled, 1.0)
                ),
            })
    stratified = pd.DataFrame(stratified_rows)
    stratified_path = args.results.with_name(
        f"{args.results.stem}_capacity_strata.csv"
    )
    stratified.to_csv(stratified_path, index=False)

    paired_rows = []
    subsets = [("all", data)]
    if "capacity_regime" in data.columns:
        subsets.extend(data.groupby("capacity_regime"))
    index = ["demand_row_id", "seed"]
    for regime, subset in subsets:
        delay = subset.pivot(index=index, columns="controller", values="avg_time_loss")
        complete = subset.pivot(index=index, columns="controller", values="fully_observed")
        if "symbolic" not in delay:
            continue
        for baseline in [name for name in delay.columns if name != "symbolic"]:
            available = delay[[baseline, "symbolic"]].notna().all(axis=1)
            valid = (
                available
                & complete[baseline].fillna(False).astype(bool)
                & complete["symbolic"].fillna(False).astype(bool)
            )
            differences = (delay.loc[valid, baseline] - delay.loc[valid, "symbolic"])
            if differences.empty:
                low = high = mean = win_rate = float("nan")
            else:
                rng = np.random.default_rng(20260712)
                values = differences.to_numpy()
                bootstrap = np.asarray([
                    rng.choice(values, size=len(values), replace=True).mean()
                    for _ in range(5000)
                ])
                mean = float(values.mean())
                low = float(np.percentile(bootstrap, 2.5))
                high = float(np.percentile(bootstrap, 97.5))
                win_rate = float(np.mean(values > 0.0))
            paired_rows.append({
                "capacity_regime": regime,
                "baseline": baseline,
                "available_pairs": int(available.sum()),
                "fully_observed_pairs": int(valid.sum()),
                "baseline_minus_symbolic_delay": mean,
                "ci95_low": low,
                "ci95_high": high,
                "symbolic_win_rate": win_rate,
            })
    paired = pd.DataFrame(paired_rows)
    paired_path = args.results.with_name(f"{args.results.stem}_complete_pairs.csv")
    paired.to_csv(paired_path, index=False)
    print(summary.to_string(index=False))
    print(summary_path)
    print(stratified_path)
    print(paired_path)


if __name__ == "__main__":
    main()
