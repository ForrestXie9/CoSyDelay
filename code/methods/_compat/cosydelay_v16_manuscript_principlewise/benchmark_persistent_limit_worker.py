"""Offline equivalence/timing benchmark for the persistent limit worker."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import time

from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import utc_now

from .physics import (
    MANUSCRIPT_RULE_NAMES,
    ManuscriptVerifierConfig,
    PersistentSymbolicLimitEvaluator,
    score_fitted_lanes_manuscript_principlewise,
)


LANES = (
    "S_L", "S_T", "S_R", "E_L", "E_T", "E_R",
    "N_L", "N_T", "N_R", "W_L", "W_T", "W_R",
)
FEATURES = ("flow_lane", "GR_phase", "Cycle_Time")
EXPRESSIONS = (
    "Cycle_Time*(a1+a2*flow_lane/GR_phase)",
    "Cycle_Time*(a1+a2*flow_lane/(GR_phase+a3))",
    "Cycle_Time*(a1+a2*(flow_lane/GR_phase)**a3+a4*log(1+a5/GR_phase))",
    "Cycle_Time*(a1+a2*(flow_lane/GR_phase)**a3+a4*exp(a5/GR_phase))",
    "Cycle_Time*flow_lane*(1-a1*flow_lane)/GR_phase",
)


def _parameters(expression: str) -> dict[str, dict[str, float]]:
    names = sorted(
        set(re.findall(r"\ba\d+\b", expression)),
        key=lambda item: int(item[1:]),
    )
    values = {name: 1.0 for name in names}
    if "**a3" in expression:
        values["a3"] = 0.75
    if expression == "Cycle_Time*flow_lane*(1-a1*flow_lane)/GR_phase":
        values["a1"] = 0.4
    return {lane: dict(values) for lane in LANES}


def _run(expression: str, evaluator=None) -> tuple[dict, float]:
    started = time.perf_counter()
    observed = score_fitted_lanes_manuscript_principlewise(
        expression,
        _parameters(expression),
        LANES,
        FEATURES,
        config=ManuscriptVerifierConfig(),
        limit_evaluator=evaluator,
    )
    return observed.to_dict(), time.perf_counter() - started


def _assert_equivalent(baseline: dict, persistent: dict) -> None:
    if baseline["rule_order"] != persistent["rule_order"]:
        raise RuntimeError("rule order changed")
    if baseline["joint_pass"] != persistent["joint_pass"]:
        raise RuntimeError("joint-pass decision changed")
    if abs(float(baseline["score"]) - float(persistent["score"])) > 1e-12:
        raise RuntimeError("physical score changed")
    for rule in MANUSCRIPT_RULE_NAMES:
        if abs(
            float(baseline["rule_scores"][rule])
            - float(persistent["rule_scores"][rule])
        ) > 1e-12:
            raise RuntimeError(f"aggregate {rule} changed")
    for lane in LANES:
        for rule in MANUSCRIPT_RULE_NAMES:
            if abs(
                float(baseline["lane_rule_scores"][lane][rule])
                - float(persistent["lane_rule_scores"][lane][rule])
            ) > 1e-12:
                raise RuntimeError(f"{lane} {rule} changed")
    if baseline["lane_errors"] != persistent["lane_errors"]:
        raise RuntimeError("physical feedback/errors changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")

    started_utc = utc_now()
    baseline_rows = []
    for expression in EXPRESSIONS:
        observed, wall = _run(expression)
        baseline_rows.append(
            {"expression": expression, "wall_seconds": wall, "result": observed}
        )

    evaluator = PersistentSymbolicLimitEvaluator()
    persistent_rows = []
    try:
        for expression, baseline in zip(EXPRESSIONS, baseline_rows):
            observed, wall = _run(expression, evaluator=evaluator)
            _assert_equivalent(baseline["result"], observed)
            persistent_rows.append(
                {"expression": expression, "wall_seconds": wall, "result": observed}
            )
        worker = evaluator.snapshot()
    finally:
        evaluator.close()

    baseline_wall = float(sum(row["wall_seconds"] for row in baseline_rows))
    persistent_wall = float(sum(row["wall_seconds"] for row in persistent_rows))
    payload = {
        "schema_version": 1,
        "status": "complete_offline_exact_equivalence_benchmark",
        "not_an_accuracy_or_promotion_experiment": True,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "expressions": len(EXPRESSIONS),
        "declared_movements": len(LANES),
        "verifier_config": asdict(ManuscriptVerifierConfig()),
        "equivalent": True,
        "comparison_scope": (
            "joint pass, aggregate score, seven aggregate rule scores, "
            "movement-by-rule scores, and physical error feedback"
        ),
        "baseline_wall_seconds": baseline_wall,
        "persistent_wall_seconds": persistent_wall,
        "speedup": baseline_wall / persistent_wall,
        "baseline_per_expression_wall_seconds": [
            row["wall_seconds"] for row in baseline_rows
        ],
        "persistent_per_expression_wall_seconds": [
            row["wall_seconds"] for row in persistent_rows
        ],
        "worker": worker,
        "cases": [
            {
                "expression": row["expression"],
                "score": row["result"]["score"],
                "joint_pass": row["result"]["joint_pass"],
                "rule_scores": row["result"]["rule_scores"],
            }
            for row in baseline_rows
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
