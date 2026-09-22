"""Offline equivalence replay for the prompt-v6 efficiency execution layer.

This diagnostic reads only archived run artifacts.  It is not a search, does
not access Training/Validation/Test data, and must not be used for selection.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping

from methods.cosydelay.engine.optimizer_parallel4.prefit_gate import (
    evaluate_structural_prefit_gate,
)
from methods.cosydelay.engine.optimizer_parallel4_v3.physics_audit import (
    audit_fitted_physics,
    audit_search_physics,
)


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE / "experiments"
RUN_NAMES = (
    "final_prompt_v6_p3g2_i1_20260811_run01",
    "final_prompt_v6_p3g2_i2_20260811_run01",
)
OUTPUT = EXPERIMENTS / "v6_efficiency_equivalence_replay.json"
PHYSICS_BOOLEAN_KEYS = (
    "standard_joint_pass",
    "symbolic_r8_exact",
    "symbolic_r9_positive_infinity",
    "dense_finite",
    "dense_nonnegative",
    "dense_flow_nondecreasing",
    "dense_green_nonincreasing",
    "joint_pass",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _standard_summary(details: Mapping[str, Any]) -> dict[str, Any]:
    verifier = dict(details["verifier"])
    return {
        "joint_pass": bool(
            verifier.get("standard_joint_pass", verifier["joint_pass"])
        ),
        "rule_scores": dict(verifier["rule_scores"]),
    }


def _physics_differences(
    archived: Mapping[str, Any],
    replayed: Mapping[str, Any],
) -> list[str]:
    differences = [
        key
        for key in PHYSICS_BOOLEAN_KEYS
        if bool(archived.get(key)) != bool(replayed.get(key))
    ]
    if dict(archived.get("standard_rule_scores", {})) != dict(
        replayed.get("standard_rule_scores", {})
    ):
        differences.append("standard_rule_scores")
    return differences


def main() -> int:
    started = time.perf_counter()
    prefit_rows: list[dict[str, Any]] = []
    fitted_rows: list[dict[str, Any]] = []
    for run_name in RUN_NAMES:
        run = EXPERIMENTS / run_name
        prefit = read_json(run / "prefit_gate_audit.json")
        endpoints = {str(item["expression"]): item for item in prefit}
        for item in prefit:
            expression = str(item["expression"])
            item_started = time.perf_counter()
            structural = evaluate_structural_prefit_gate(expression)
            symbolic = (
                audit_search_physics(expression, standard_joint_pass=True)
                if structural.passed
                else None
            )
            replay_pass = bool(
                structural.passed
                and symbolic is not None
                and symbolic["symbolic_r8_exact"]
                and symbolic["symbolic_r9_positive_infinity"]
            )
            prefit_rows.append(
                {
                    "run": run_name,
                    "expression": expression,
                    "archived_pass": bool(item["passed"]),
                    "replayed_pass": replay_pass,
                    "equivalent": bool(item["passed"]) == replay_pass,
                    "r9_method": (
                        None if symbolic is None else symbolic["symbolic_r9_method"]
                    ),
                    "wall_seconds": time.perf_counter() - item_started,
                }
            )

        result = read_json(run / "result.json")
        history = read_json(run / "history.json")
        selected_expression = str(result["selected_expression"])
        selected_history = next(
            item
            for item in history
            if item.get("event") == "evaluated"
            and str(item.get("expression")) == selected_expression
        )
        fitted_cases = [
            {
                "stage": "selected",
                "expression": selected_expression,
                "parameters": result["selected_parameters"],
                "archived": result["selected_enhanced_physics"],
                "details": selected_history["evaluation_details"],
            }
        ]
        for rejected in read_json(run / "fitted_rejections.json"):
            fitted_cases.append(
                {
                    "stage": "fitted_rejection",
                    "expression": str(rejected["expression"]),
                    "parameters": rejected["parameters"],
                    "archived": rejected["enhanced_physics"],
                    "details": rejected["details"],
                }
            )

        for case in fitted_cases:
            expression = str(case["expression"])
            item_started = time.perf_counter()
            replayed = audit_fitted_physics(
                expression,
                case["parameters"],
                list(case["parameters"]),
                precomputed_standard=_standard_summary(case["details"]),
                precomputed_symbolic=endpoints[expression][
                    "coefficient_robust_endpoint_audit"
                ],
            )
            differences = _physics_differences(case["archived"], replayed)
            fitted_rows.append(
                {
                    "run": run_name,
                    "stage": case["stage"],
                    "expression": expression,
                    "equivalent": not differences,
                    "differences": differences,
                    "standard_audit_reused": replayed["standard_audit_reused"],
                    "symbolic_endpoint_audit_reused": replayed[
                        "symbolic_endpoint_audit_reused"
                    ],
                    "wall_seconds": time.perf_counter() - item_started,
                }
            )

    prefit_equivalent = sum(bool(item["equivalent"]) for item in prefit_rows)
    fitted_equivalent = sum(bool(item["equivalent"]) for item in fitted_rows)
    report = {
        "schema_version": 1,
        "audit_type": "offline_archived_candidate_equivalence_replay",
        "not_a_search_result": True,
        "dataset_files_accessed": False,
        "runs": list(RUN_NAMES),
        "prefit": {
            "records": len(prefit_rows),
            "equivalent": prefit_equivalent,
            "all_equivalent": prefit_equivalent == len(prefit_rows),
            "fast_r9_certificates": sum(
                item["r9_method"]
                == "fixed_inverse_green_finite_positive_h_certificate"
                for item in prefit_rows
            ),
            "wall_seconds": sum(item["wall_seconds"] for item in prefit_rows),
        },
        "fitted": {
            "records_with_archived_parameters": len(fitted_rows),
            "equivalent": fitted_equivalent,
            "all_equivalent": fitted_equivalent == len(fitted_rows),
            "wall_seconds": sum(item["wall_seconds"] for item in fitted_rows),
        },
        "all_equivalent": bool(
            prefit_equivalent == len(prefit_rows)
            and fitted_equivalent == len(fitted_rows)
        ),
        "total_wall_seconds": time.perf_counter() - started,
        "prefit_rows": prefit_rows,
        "fitted_rows": fitted_rows,
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "all_equivalent": report["all_equivalent"],
                "prefit": report["prefit"],
                "fitted": report["fitted"],
                "output": str(OUTPUT),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["all_equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
