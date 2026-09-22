"""Direction-invariant residual diagnostics for universal-expression search.

The helper in this module is deliberately independent of approach labels in
the text returned to the LLM.  Labels remain in the audit payload so that a
researcher can verify how each feedback statement was produced.

Only coefficient-fit rows are intended as input.  Final expression selection
must still use a separate validation split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ResidualGuidanceConfig:
    """Thresholds for stable, scale-free regime diagnostics."""

    low_quantile: float = 1.0 / 3.0
    high_quantile: float = 2.0 / 3.0
    normalized_effect_threshold: float = 0.12
    half_effect_threshold: float = 0.06
    bottleneck_r2_gap_threshold: float = 0.10
    min_rows_per_half: int = 12
    min_shared_approaches: int = 2
    max_prompt_patterns: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < self.low_quantile < self.high_quantile < 1.0:
            raise ValueError("Residual-guidance quantiles must satisfy 0 < low < high < 1")
        if self.normalized_effect_threshold <= 0:
            raise ValueError("normalized_effect_threshold must be positive")
        if self.half_effect_threshold <= 0:
            raise ValueError("half_effect_threshold must be positive")
        if self.bottleneck_r2_gap_threshold < 0:
            raise ValueError("bottleneck_r2_gap_threshold must be non-negative")
        if self.min_rows_per_half < 2:
            raise ValueError("min_rows_per_half must be at least 2")
        if self.min_shared_approaches < 2:
            raise ValueError("min_shared_approaches must be at least 2")
        if self.max_prompt_patterns < 1:
            raise ValueError("max_prompt_patterns must be at least 1")


FEATURE_LABELS = {
    "relative_demand": "relative demand",
    "effective_green": "effective green ratio",
    "cycle_length": "cycle length",
}
# Time-dimensional legality requires the complete expression to remain
# homogeneous of degree one in Cycle_Time.  Cycle-length residual patterns are
# useful diagnostics, but they are not actionable structural mutations under
# the current paper's model class.
ACTIONABLE_PROMPT_FEATURES = {
    "relative_demand",
    "effective_green",
}


def _raw_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    centered = truth - float(np.mean(truth))
    denominator = float(np.sum(centered ** 2))
    if denominator <= 0.0:
        return float("-inf")
    return float(1.0 - np.sum((truth - prediction) ** 2) / denominator)


def _approach_regime_features(
    prepared_context: Mapping[str, Any],
    approach: str,
) -> Dict[str, np.ndarray]:
    weight_context = prepared_context["approach_weights"][approach]
    lanes = list(weight_context["lanes"])
    lane_contexts = prepared_context["lane_contexts"]
    n_rows = len(weight_context["total_flow"])

    effective_green = np.zeros(n_rows, dtype=float)
    for lane in lanes:
        effective_green += (
            np.asarray(weight_context["weights"][lane], dtype=float)
            * np.asarray(lane_contexts[lane]["GR_phase"], dtype=float)
        )
    cycle = np.asarray(lane_contexts[lanes[0]]["Cycle_Time"], dtype=float)
    return {
        "relative_demand": np.asarray(weight_context["total_flow"], dtype=float),
        "effective_green": effective_green,
        "cycle_length": cycle,
    }


def _stable_regime_records(
    approach: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    features: Mapping[str, np.ndarray],
    config: ResidualGuidanceConfig,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    truth_valid = truth[valid]
    prediction_valid = prediction[valid]
    target_scale = float(np.std(truth_valid))
    if not np.isfinite(target_scale) or target_scale <= 1e-12:
        return {
            "approach": approach,
            "rows": int(np.count_nonzero(valid)),
            "status": "constant_or_invalid_target",
        }, []

    normalized_residual = (truth - prediction) / target_scale
    valid_positions = np.flatnonzero(valid)
    split_a = np.zeros(len(truth), dtype=bool)
    split_b = np.zeros(len(truth), dtype=bool)
    split_a[valid_positions[::2]] = True
    split_b[valid_positions[1::2]] = True
    records: list[Dict[str, Any]] = []

    for feature_name, raw_values in features.items():
        values = np.asarray(raw_values, dtype=float)
        feature_valid = valid & np.isfinite(values)
        finite_values = values[feature_valid]
        if finite_values.size < 4 * config.min_rows_per_half:
            continue
        low_cut = float(np.quantile(finite_values, config.low_quantile))
        high_cut = float(np.quantile(finite_values, config.high_quantile))
        feature_scale = max(1.0, abs(low_cut), abs(high_cut))
        if high_cut - low_cut <= 1e-10 * feature_scale:
            continue
        regimes = {
            "low": feature_valid & (values <= low_cut),
            "high": feature_valid & (values >= high_cut),
        }
        for regime, regime_mask in regimes.items():
            half_a = regime_mask & split_a
            half_b = regime_mask & split_b
            count_a = int(np.count_nonzero(half_a))
            count_b = int(np.count_nonzero(half_b))
            if min(count_a, count_b) < config.min_rows_per_half:
                continue
            effect_a = float(np.mean(normalized_residual[half_a]))
            effect_b = float(np.mean(normalized_residual[half_b]))
            effect = float(np.mean(normalized_residual[regime_mask]))
            same_sign = np.sign(effect_a) == np.sign(effect_b) != 0
            stable = (
                same_sign
                and abs(effect) >= config.normalized_effect_threshold
                and min(abs(effect_a), abs(effect_b))
                >= config.half_effect_threshold
            )
            if not stable:
                continue
            records.append(
                {
                    "approach": approach,
                    "feature": feature_name,
                    "regime": regime,
                    "sign": "underprediction" if effect > 0 else "overprediction",
                    "normalized_effect": effect,
                    "half_effects": [effect_a, effect_b],
                    "rows": int(np.count_nonzero(regime_mask)),
                    "cut_points": {"low": low_cut, "high": high_cut},
                }
            )

    diagnostics = {
        "approach": approach,
        "rows": int(np.count_nonzero(valid)),
        "target_std": target_scale,
        "r2": _raw_r2(truth_valid, prediction_valid),
        "stable_pattern_count": len(records),
        "status": "ok",
    }
    return diagnostics, records


def _prompt_line(
    record: Mapping[str, Any],
    *,
    scope: str,
    support: int,
    total_approaches: int,
) -> str:
    effect = float(record["normalized_effect"])
    direction = (
        "underpredicts delay (positive normalized residual)"
        if effect > 0
        else "overpredicts delay (negative normalized residual)"
    )
    if scope == "shared":
        prefix = f"Across {support}/{total_approaches} anonymous approach groups"
    else:
        prefix = "Within one split-stable anonymous bottleneck group"
    return (
        f"- {prefix}, the {record['regime']} {FEATURE_LABELS[record['feature']]} "
        f"regime {direction}; mean standardized effect={effect:+.3f}. "
        "Seek a parsimonious shared structure with enough curvature in this "
        "operating regime without adding any group indicator."
    )


def build_direction_invariant_residual_guidance(
    approach_targets: Mapping[str, Any],
    predictions: Mapping[str, Sequence[float]],
    prepared_context: Mapping[str, Any],
    *,
    config: ResidualGuidanceConfig | None = None,
    include_stable_bottleneck: bool = True,
) -> Dict[str, Any]:
    """Build anonymous regime feedback from coefficient-fit residuals.

    Shared patterns are preferred.  A single bottleneck group may contribute a
    pattern only when its raw R2 is materially below the cross-group median and
    the residual effect has the same sign in deterministic split halves.
    """

    config = config or ResidualGuidanceConfig()
    diagnostics: Dict[str, Dict[str, Any]] = {}
    stable_records: list[Dict[str, Any]] = []
    for approach, target in approach_targets.items():
        if approach not in predictions:
            continue
        truth = np.asarray(target.values, dtype=float)
        prediction = np.asarray(predictions[approach], dtype=float)
        weight_context = prepared_context["approach_weights"].get(approach)
        if weight_context is None:
            continue
        valid = (
            np.asarray(weight_context["valid_mask"], dtype=bool)
            & np.isfinite(truth)
            & np.isfinite(prediction)
        )
        if np.count_nonzero(valid) < 2:
            continue
        approach_diagnostics, records = _stable_regime_records(
            approach,
            truth,
            prediction,
            valid,
            _approach_regime_features(prepared_context, approach),
            config,
        )
        diagnostics[approach] = approach_diagnostics
        stable_records.extend(records)

    usable = [
        value
        for value in diagnostics.values()
        if value.get("status") == "ok" and np.isfinite(value.get("r2", np.nan))
    ]
    total_approaches = len(usable)
    if total_approaches < 2:
        return {
            "schema_version": 1,
            "status": "insufficient_approaches",
            "prompt_text": "",
            "config": asdict(config),
            "approach_diagnostics": diagnostics,
            "patterns": [],
            "approach_labels_exposed_to_prompt": False,
            "source_split": "coefficient_fit",
        }

    grouped: Dict[tuple[str, str, str], list[Dict[str, Any]]] = {}
    for record in stable_records:
        key = (record["feature"], record["regime"], record["sign"])
        grouped.setdefault(key, []).append(record)

    candidates: list[Dict[str, Any]] = []
    required_shared = min(config.min_shared_approaches, total_approaches)
    for (feature, regime, sign), records in grouped.items():
        if feature not in ACTIONABLE_PROMPT_FEATURES:
            continue
        approaches = sorted({record["approach"] for record in records})
        if len(approaches) < required_shared:
            continue
        effects = [float(record["normalized_effect"]) for record in records]
        candidates.append(
            {
                "scope": "shared",
                "feature": feature,
                "regime": regime,
                "sign": sign,
                "normalized_effect": float(np.mean(effects)),
                "support": len(approaches),
                "supporting_approaches": approaches,
                "source_records": records,
            }
        )

    r2_by_approach = {
        value["approach"]: float(value["r2"]) for value in usable
    }
    worst_approach = min(r2_by_approach, key=r2_by_approach.get)
    median_r2 = float(np.median(list(r2_by_approach.values())))
    bottleneck_gap = median_r2 - r2_by_approach[worst_approach]
    if include_stable_bottleneck and bottleneck_gap >= config.bottleneck_r2_gap_threshold:
        for record in stable_records:
            if record["approach"] != worst_approach:
                continue
            if record["feature"] not in ACTIONABLE_PROMPT_FEATURES:
                continue
            candidates.append(
                {
                    **record,
                    "scope": "bottleneck",
                    "support": 1,
                    "supporting_approaches": [worst_approach],
                    "bottleneck_r2_gap": bottleneck_gap,
                    "source_records": [record],
                }
            )

    candidates.sort(
        key=lambda item: (
            item["scope"] == "shared",
            int(item["support"]),
            abs(float(item["normalized_effect"])),
        ),
        reverse=True,
    )
    selected: list[Dict[str, Any]] = []
    seen_regimes: set[tuple[str, str]] = set()
    for candidate in candidates:
        regime_key = (candidate["feature"], candidate["regime"])
        if regime_key in seen_regimes:
            continue
        selected.append(candidate)
        seen_regimes.add(regime_key)
        if len(selected) >= config.max_prompt_patterns:
            break

    prompt_lines = [
        _prompt_line(
            record,
            scope=str(record["scope"]),
            support=int(record["support"]),
            total_approaches=total_approaches,
        )
        for record in selected
    ]
    prompt_text = ""
    if prompt_lines:
        prompt_text = "\n".join(
            [
                "GENERALIZATION-ORIENTED RESIDUAL FEEDBACK",
                (
                    "Computed only on coefficient-fit rows. Approach identities "
                    "are intentionally withheld; react to operating regimes, "
                    "not to a named direction."
                ),
                *prompt_lines,
                (
                    "The child must remain one universal expression shared by "
                    "all lanes, with lane-specific coefficients fitted later."
                ),
            ]
        )

    return {
        "schema_version": 1,
        "status": "patterns_available" if selected else "no_stable_pattern",
        "prompt_text": prompt_text,
        "config": asdict(config),
        "approach_diagnostics": diagnostics,
        "patterns": selected,
        "stable_records": stable_records,
        "r2_by_approach": r2_by_approach,
        "anonymous_bottleneck_detected": bool(
            bottleneck_gap >= config.bottleneck_r2_gap_threshold
        ),
        "bottleneck_r2_gap": bottleneck_gap,
        "approach_labels_exposed_to_prompt": False,
        "source_split": "coefficient_fit",
    }
