from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd


AUDIT_COLUMNS = {
    "scheduled_vehicles",
    "tripinfo_vehicles",
    "completed",
    "route_entry_rate",
    "fully_observed",
    "clearance_complete",
    "residual_queue",
    "residual_expected_vehicles",
    "run_sha256",
    "config_sha256",
    "route_sha256",
    "duration_s",
    "update_interval_s",
    "clearance_limit_s",
}
LOGICAL_KEY = [
    "scenario", "controller", "seed", "duration_s", "update_interval_s",
    "clearance_limit_s",
]
CELL_KEY = [
    "scenario", "seed", "duration_s", "update_interval_s",
    "clearance_limit_s", "route_sha256",
]


def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _bootstrap_interval(values: np.ndarray, seed: int = 20260712):
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.asarray([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(5000)
    ])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="?", default=Path("outputs/results.csv"))
    parser.add_argument("--reference", default="symbolic")
    parser.add_argument("--metric", default="avg_total_delay")
    args = parser.parse_args()

    data = pd.read_csv(args.results)
    required_identity = {"scenario", "controller", "seed"}
    missing_identity = sorted(required_identity - set(data.columns))
    if missing_identity:
        raise ValueError(f"Missing result identity columns: {missing_identity}")

    audit_missing = sorted(AUDIT_COLUMNS - set(data.columns))
    data["audit_schema_complete"] = not audit_missing
    if audit_missing:
        print(
            "WARNING: legacy/incomplete audit schema; rows are excluded from the "
            f"fully-observed main table. Missing={audit_missing}",
            file=sys.stderr,
        )
        for column in AUDIT_COLUMNS:
            if column not in data.columns:
                data[column] = np.nan
    if "generated" not in data.columns:
        data["generated"] = data["tripinfo_vehicles"]

    data["fully_observed_declared"] = data["fully_observed"].map(_as_bool)
    numeric = [
        "scheduled_vehicles", "tripinfo_vehicles", "completed", "route_entry_rate",
        "residual_queue", "residual_expected_vehicles",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["clearance_complete_verified"] = data["clearance_complete"].map(_as_bool)
    recomputed_fully_observed = (
        data["audit_schema_complete"]
        & data["scheduled_vehicles"].notna()
        & (data["scheduled_vehicles"] == data["tripinfo_vehicles"])
        & (data["tripinfo_vehicles"] == data["completed"])
        & np.isclose(data["route_entry_rate"], 1.0, equal_nan=False)
        & data["clearance_complete_verified"]
        & np.isclose(data["residual_queue"], 0.0, equal_nan=False)
        & (data["residual_expected_vehicles"] == 0)
    )
    data["fully_observed_verified"] = (
        data["fully_observed_declared"] & recomputed_fully_observed
    )

    duplicate_mask = data.duplicated(LOGICAL_KEY, keep=False)
    run_hash_duplicate = data["run_sha256"].notna() & data.duplicated(
        ["run_sha256"], keep=False
    )
    duplicate_mask |= run_hash_duplicate
    duplicates = data.loc[duplicate_mask].copy()
    duplicate_path = args.results.with_name(f"{args.results.stem}_duplicate_runs.csv")
    duplicates.to_csv(duplicate_path, index=False)
    if not duplicates.empty:
        print(
            f"WARNING: excluded {len(duplicates)} duplicate rows; see {duplicate_path}",
            file=sys.stderr,
        )

    expected_controllers = sorted(data["controller"].dropna().astype(str).unique())
    data["complete_controller_cell"] = False
    data["exclusion_reason"] = ""
    data.loc[duplicate_mask, "exclusion_reason"] = "duplicate_logical_or_run_hash"
    nonduplicates = data.loc[~duplicate_mask]
    for _, indices in nonduplicates.groupby(CELL_KEY, dropna=False).groups.items():
        group = data.loc[indices]
        controllers = sorted(group["controller"].astype(str).tolist())
        controller_complete = (
            controllers == expected_controllers
            and not group["controller"].duplicated().any()
        )
        observed_complete = bool(group["fully_observed_verified"].all())
        if controller_complete and observed_complete:
            data.loc[indices, "complete_controller_cell"] = True
        else:
            reasons = []
            if not controller_complete:
                reasons.append("incomplete_controller_pairing")
            if not observed_complete:
                reasons.append("unfinished_or_unverified")
            data.loc[indices, "exclusion_reason"] = ";".join(reasons)

    eligible = data.loc[data["complete_controller_cell"]].copy()
    excluded = data.loc[~data["complete_controller_cell"]].copy()
    excluded_path = args.results.with_name(
        f"{args.results.stem}_unfinished_or_invalid.csv"
    )
    excluded.to_csv(excluded_path, index=False)
    if not excluded.empty:
        print(
            f"WARNING: excluded {len(excluded)} unfinished/unpaired/unverified rows; "
            f"see {excluded_path}",
            file=sys.stderr,
        )

    audit_metrics = [
        column for column in (
            "scheduled_vehicles", "tripinfo_vehicles", "completed", "route_entry_rate",
            "completion_rate", "unfinished", "clearance_seconds", "residual_queue",
            "residual_expected_vehicles", "avg_depart_delay", "avg_time_loss",
            "avg_total_delay",
        )
        if column in data.columns
    ]
    audit = data.groupby(["scenario", "controller"], dropna=False)[audit_metrics].agg(
        ["mean", "min", "max", "count"]
    )
    audit_path = args.results.with_name(f"{args.results.stem}_completion_audit.csv")
    audit.to_csv(audit_path)

    summary_metrics = [
        column for column in (
            "avg_total_delay", "avg_time_loss", "avg_depart_delay", "avg_waiting_time",
            "demand_window_p95_queue", "demand_window_max_queue", "p95_queue", "max_queue",
            "clearance_seconds", "throughput_veh_per_hour", "wall_seconds",
        )
        if column in eligible.columns
    ]
    if eligible.empty:
        summary = pd.DataFrame()
        print(
            "WARNING: no fully observed, duplicate-free, controller-complete cells; "
            "main summary and paired comparisons are empty.",
            file=sys.stderr,
        )
    else:
        summary = eligible.groupby(["scenario", "controller"])[summary_metrics].agg(
            ["mean", "std", "count"]
        )
    summary_path = args.results.with_name(f"{args.results.stem}_summary.csv")
    summary.to_csv(summary_path)
    complete_path = args.results.with_name(f"{args.results.stem}_fully_observed.csv")
    eligible.to_csv(complete_path, index=False)

    metric = args.metric
    if metric not in data.columns:
        if metric == "avg_total_delay" and "avg_time_loss" in data.columns:
            metric = "avg_time_loss"
            print(
                "WARNING: avg_total_delay unavailable; using legacy avg_time_loss.",
                file=sys.stderr,
            )
        else:
            raise ValueError(f"Requested comparison metric is absent: {args.metric}")

    paired = []
    for scenario, group in eligible.groupby("scenario"):
        if args.reference not in set(group["controller"]):
            continue
        pivot = group.pivot_table(
            index=[field for field in CELL_KEY if field != "scenario"],
            columns="controller",
            values=metric,
            aggfunc="first",
        )
        for baseline in [column for column in pivot.columns if column != args.reference]:
            differences = (pivot[baseline] - pivot[args.reference]).dropna()
            low, high = _bootstrap_interval(differences.to_numpy(float))
            paired.append({
                "scenario": scenario,
                "reference": args.reference,
                "baseline": baseline,
                "metric": metric,
                "n_complete_pairs": len(differences),
                "baseline_minus_reference": (
                    float(differences.mean()) if len(differences) else np.nan
                ),
                "ci95_low": low,
                "ci95_high": high,
                "reference_win_rate": (
                    float(np.mean(differences > 0)) if len(differences) else np.nan
                ),
                "pairing_rule": "fully_observed and complete controller cell only",
            })
    paired_path = args.results.with_name(f"{args.results.stem}_paired_comparisons.csv")
    paired_columns = [
        "scenario", "reference", "baseline", "metric", "n_complete_pairs",
        "baseline_minus_reference", "ci95_low", "ci95_high", "reference_win_rate",
        "pairing_rule",
    ]
    pd.DataFrame(paired, columns=paired_columns).to_csv(paired_path, index=False)

    if not summary.empty:
        print(summary.to_string())
    for path in (
        summary_path, paired_path, complete_path, excluded_path, duplicate_path, audit_path
    ):
        print(path)


if __name__ == "__main__":
    main()
