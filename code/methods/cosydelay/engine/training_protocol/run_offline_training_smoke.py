"""No-LLM, Training-only integration smoke for the V16 execution stack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
import population_evolution_lane as population
from methods.cosydelay.engine.data_protocol.run_formal_training_search import (
    DEFAULT_DATA_DIR,
    lanes_for,
    utc_now,
)

from .integration import install_v16_candidate
from .contract import V16_CONTRACT, validate_v16_contract
from .fitter import _LEGACY_DIAGNOSTIC_KEYS
from .physics import MANUSCRIPT_RULE_NAMES, ManuscriptVerifierConfig
from .policy import V16_POLICY
from .source_manifest import V16_SOURCE_FILES, validate_v16_source_manifest


EXPRESSION_PROFILES = {
    "positive_low_demand": {
        "expression": "Cycle_Time*(a1+a2*flow_lane/GR_phase)",
        "expected_failed_rules": (),
        "purpose": "verify that a positive finite low-demand limit is accepted",
    },
    "finite_zero_green": {
        "expression": "Cycle_Time*(a1+a2*flow_lane/(GR_phase+a3))",
        "expected_failed_rules": ("R7_zero_green_limit",),
        "purpose": "verify that a six-of-seven candidate is scored, not hard-rejected",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(EXPRESSION_PROFILES),
        default="positive_low_demand",
    )
    args = parser.parse_args()
    profile = EXPRESSION_PROFILES[args.profile]
    expression = str(profile["expression"])
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    validate_v16_source_manifest()
    validate_v16_contract()

    train_path = (
        args.data_dir / f"Intersection_{args.intersection}_Train.jsonl"
    ).resolve()
    train = preprocess_data_flexible(
        load_dataset_flexible(str(train_path), args.intersection),
        args.intersection,
    ).reset_index(drop=True)
    config = INTERSECTION_CONFIGS[args.intersection]
    targets = {
        approach: train[f"Delay_{approach}"].reset_index(drop=True)
        for approach in config["approaches"]
    }
    lanes, lane_to_approach = lanes_for(config)
    context = population.prepare_optimization_context(
        train, lanes, lane_to_approach, args.intersection
    )

    started_utc = utc_now()
    started = time.perf_counter()
    with install_v16_candidate(
        df_train=train,
        targets=targets,
        lanes=lanes,
        lane_to_approach=lane_to_approach,
        intersection_id=args.intersection,
        verifier_config=ManuscriptVerifierConfig(),
    ) as runtime:
        observed = population.evaluate_expression_with_fitting(
            expression,
            train,
            lanes,
            lane_to_approach,
            targets,
            args.intersection,
            context,
            score_mode="principlewise",
            physics_weight=1.0,
            optimizer_restarts=V16_POLICY.optimizer_restarts,
        )
        physical_score, train_r2, fitness, parameters, feedback, details = observed
        evaluation = runtime.evaluations.get(expression)

    verifier = details.get("verifier", {})
    if details.get("status") != "evaluated" or not parameters:
        raise RuntimeError(f"V16 smoke fitting failed: {details}")
    if tuple(verifier.get("rule_order", ())) != MANUSCRIPT_RULE_NAMES:
        raise RuntimeError("V16 smoke returned the wrong physical rule order")
    rule_scores = verifier.get("rule_scores", {})
    if rule_scores.get("R5_finite_nonnegative_low_demand_limit") != 1.0:
        raise RuntimeError("positive finite low-demand limit was not accepted")
    if "R8_zero_flow_boundary" in verifier.get("rule_scores", {}):
        raise RuntimeError("superseded exact-zero rule leaked into V16")
    failed_rules = tuple(
        name for name in MANUSCRIPT_RULE_NAMES if float(rule_scores[name]) < 1.0
    )
    if failed_rules != tuple(profile["expected_failed_rules"]):
        raise RuntimeError(
            f"unexpected failed rules for {args.profile}: {failed_rules}"
        )
    expected_score = 1.0 - len(failed_rules) / len(MANUSCRIPT_RULE_NAMES)
    if abs(float(physical_score) - expected_score) > 1e-12:
        raise RuntimeError(
            f"unexpected physical score {physical_score}; expected {expected_score}"
        )
    if failed_rules and details.get("status") != "evaluated":
        raise RuntimeError("partial physical candidate was hard-rejected")
    if details.get("outer_validation_accessed") or details.get("test_file_opened"):
        raise RuntimeError("V16 smoke crossed the Training-only boundary")
    if details.get("physical_timing_source") != "v16_seven_rule_rescore":
        raise RuntimeError("V16 smoke did not record the real seven-rule timing")
    if "v16_physics_wall_seconds" not in details:
        raise RuntimeError("V16 smoke is missing the physical timing value")
    if details.get("enhanced_physics_wall_seconds") != 0.0:
        raise RuntimeError("V16 smoke unexpectedly repeated the enhanced audit")
    serialized_details = json.dumps(details, ensure_ascii=False, default=str)
    leaked_probe_fields = [
        field
        for field in sorted(_LEGACY_DIAGNOSTIC_KEYS)
        if f'"{field}"' in serialized_details
    ]
    if leaked_probe_fields:
        raise RuntimeError(
            "superseded numerical-R9 diagnostics leaked into V16 smoke: "
            f"{leaked_probe_fields}"
        )
    if details.get("legacy_numeric_r9_probe_executed") is not False:
        raise RuntimeError("V16 smoke did not prove the legacy probe was omitted")

    payload = {
        "schema_version": 1,
        "status": "complete_training_only_no_llm_smoke",
        "not_an_accuracy_or_promotion_experiment": True,
        "method_id": V16_POLICY.method_id,
        "intersection_id": args.intersection,
        "profile": args.profile,
        "purpose": profile["purpose"],
        "expression": expression,
        "expression_has_positive_low_demand_intercept": True,
        "expected_failed_rules": list(profile["expected_failed_rules"]),
        "observed_failed_rules": list(failed_rules),
        "optimizer": V16_POLICY.optimizer,
        "optimizer_restarts": V16_POLICY.optimizer_restarts,
        "approach_workers_cap": V16_POLICY.approach_workers_cap,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "accessed_splits": ["train"],
        "llm_called": False,
        "training_accuracy": float(train_r2),
        "physical_score": float(physical_score),
        "fitness": float(fitness),
        "strict_joint_pass": bool(verifier.get("joint_pass")),
        "legacy_numeric_r9_probe_executed": False,
        "optimizer_restart_selection": "minimum_training_mse",
        "rule_scores": rule_scores,
        "verifier": verifier,
        "feedback": feedback,
        "parameters": parameters,
        "runtime_evaluation_recorded": evaluation is not None,
        "timing": {
            "fit_wall_seconds": float(details.get("fit_wall_seconds", 0.0)),
            "v16_physics_wall_seconds": float(
                details["v16_physics_wall_seconds"]
            ),
            "compatibility_bypass_physical_wall_seconds": float(
                details.get(
                    "compatibility_bypass_physical_wall_seconds", 0.0
                )
            ),
            "compatibility_bypass_enhanced_physics_wall_seconds": float(
                details.get(
                    "compatibility_bypass_enhanced_physics_wall_seconds", 0.0
                )
            ),
            "physical_timing_source": details["physical_timing_source"],
            "score_reused_as_enhanced_audit": bool(
                details.get("v16_score_reused_as_enhanced_audit")
            ),
        },
        "method_contract": V16_CONTRACT.to_dict(),
        "source_sha256": {
            str(path.resolve()): _sha256(path) for path in V16_SOURCE_FILES
        },
        "train_sha256": _sha256(train_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
