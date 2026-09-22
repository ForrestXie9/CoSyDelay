"""Freeze, run, and integrity-check the six V16 Training-only searches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from constants import INTERSECTION_CONFIGS
from expression_rules import coefficient_names
from llm_config import load_llm_config
from methods.cosydelay.engine.data_protocol.operations.run04_pythonw_supervisor import (
    recover_api_key,
)
from methods.cosydelay.engine.data_protocol.run_formal_training_search import (
    DEFAULT_DATA_DIR,
    sha256_file,
)

from .contract import V16_CONTRACT, validate_v16_contract
from .physics import MANUSCRIPT_RULE_NAMES, MANUSCRIPT_RULE_SCHEMA_ID
from .policy import V16_POLICY
from .source_manifest import GMINI, V16_SOURCE_FILES, validate_v16_source_manifest


INTERSECTIONS = tuple(range(1, 7))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_hashes() -> dict[str, str]:
    return {
        (
            str(path.relative_to(GMINI)).replace("\\", "/")
            if path.is_relative_to(GMINI)
            else str(path)
        ): sha256_file(path)
        for path in V16_SOURCE_FILES
    }


def _package_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "sympy", "scikit-learn", "numexpr")
    return {name: metadata.version(name) for name in names}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def validate_formal_result(
    result: dict[str, Any], protocol: dict[str, Any], *, intersection: int
) -> None:
    """Fail closed when a child differs from the pre-API Training freeze."""
    required = {
        "status": "completed_v16_p10g10_training_only_search",
        "not_a_formal_result": False,
        "formal_protocol_complete": True,
        "method_id": V16_POLICY.method_id,
        "intersection_id": int(intersection),
        "population": V16_CONTRACT.population,
        "generations": V16_CONTRACT.generations,
        "policy": protocol["policy"],
        "method_contract": protocol["method_contract"],
        "train_file_sha256": protocol["training_sha256"][str(intersection)][
            "sha256"
        ],
        "selection_source": "full_training_evolution_only",
        "selection_order": [
            "paper_training_fitness",
            "raw_training_r2_on_exact_fitness_ties",
            "negative_training_rmse_on_remaining_ties",
        ],
        "population_update": V16_CONTRACT.population_update,
        "strict_joint_pass_role": "diagnostic_only_not_a_hard_gate",
        "paper_fitness_definition_changed": False,
        "accessed_splits": ["train"],
        "outer_validation_accessed": False,
        "test_file_opened": False,
        "validation_or_test_used_for_selection": False,
        "post_evolution_cv_reranking": False,
        "post_evolution_refit": False,
        "external_incumbent_allowed": False,
        "initial_population_all_generated_in_current_run": True,
        "completed_generations": V16_CONTRACT.generations,
        "legacy_numeric_r9_probe_executed": False,
        "candidate_evaluations": (
            V16_CONTRACT.population * (V16_CONTRACT.generations + 1)
        ),
        "v16_source_sha256": protocol["source_sha256"],
    }
    mismatches = {
        key: {"observed": result.get(key), "required": expected}
        for key, expected in required.items()
        if result.get(key) != expected
    }
    physical = result.get("selected_enhanced_physics", {})
    if physical.get("rule_schema_id") != MANUSCRIPT_RULE_SCHEMA_ID:
        mismatches["selected_enhanced_physics.rule_schema_id"] = {
            "observed": physical.get("rule_schema_id"),
            "required": MANUSCRIPT_RULE_SCHEMA_ID,
        }
    if tuple(physical.get("rule_order", ())) != MANUSCRIPT_RULE_NAMES:
        mismatches["selected_enhanced_physics.rule_order"] = {
            "observed": physical.get("rule_order"),
            "required": list(MANUSCRIPT_RULE_NAMES),
        }
    try:
        physical_score = float(physical["score"])
    except (KeyError, TypeError, ValueError):
        physical_score = float("nan")
    if not np.isfinite(physical_score) or not 0.0 <= physical_score <= 1.0:
        mismatches["selected_enhanced_physics.score"] = {
            "observed": physical.get("score"),
            "required": "finite value in [0,1]",
        }

    declared_rule_scores_raw = physical.get("rule_scores", {})
    movement_rule_scores_raw = physical.get("lane_rule_scores", {})
    lane_parameters_raw = result.get("lane_parameters", {})
    declared_rule_scores = (
        declared_rule_scores_raw
        if isinstance(declared_rule_scores_raw, dict)
        else {}
    )
    movement_rule_scores = (
        movement_rule_scores_raw
        if isinstance(movement_rule_scores_raw, dict)
        else {}
    )
    lane_parameters = (
        lane_parameters_raw if isinstance(lane_parameters_raw, dict) else {}
    )
    intersection_config = INTERSECTION_CONFIGS[int(intersection)]
    expected_movements = {
        f"{approach}_{movement}"
        for approach in intersection_config["approaches"]
        for movement in intersection_config["movements"][approach]
    }
    expression = str(result.get("expression", ""))
    try:
        required_coefficients = set(coefficient_names(expression))
    except Exception:
        required_coefficients = set()
    physical_payload_errors: list[dict[str, Any]] = []
    if not isinstance(movement_rule_scores_raw, dict) or not movement_rule_scores:
        physical_payload_errors.append({"reason": "missing movement rule scores"})
    if not isinstance(lane_parameters_raw, dict) or not lane_parameters:
        physical_payload_errors.append({"reason": "missing movement parameters"})
    if set(movement_rule_scores) != set(lane_parameters):
        physical_payload_errors.append(
            {
                "reason": "movement key mismatch",
                "physical_movements": sorted(movement_rule_scores),
                "parameter_movements": sorted(lane_parameters),
            }
        )
    if set(movement_rule_scores) != expected_movements:
        physical_payload_errors.append(
            {
                "reason": "intersection movement layout mismatch",
                "observed_movements": sorted(movement_rule_scores),
                "required_movements": sorted(expected_movements),
            }
        )
    if not required_coefficients:
        physical_payload_errors.append(
            {"reason": "expression has no valid fitted coefficient identifiers"}
        )
    cells: list[float] = []
    recomputed_by_rule: dict[str, list[float]] = {
        name: [] for name in MANUSCRIPT_RULE_NAMES
    }
    for lane, scores in movement_rule_scores.items():
        if not isinstance(scores, dict) or set(scores) != set(MANUSCRIPT_RULE_NAMES):
            physical_payload_errors.append(
                {
                    "reason": "movement rule schema mismatch",
                    "movement": lane,
                    "observed_rules": sorted(scores) if isinstance(scores, dict) else None,
                }
            )
            continue
        for name in MANUSCRIPT_RULE_NAMES:
            try:
                value = float(scores[name])
            except (TypeError, ValueError):
                value = float("nan")
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                physical_payload_errors.append(
                    {
                        "reason": "invalid movement-rule score",
                        "movement": lane,
                        "rule": name,
                        "value": scores.get(name),
                    }
                )
            else:
                cells.append(value)
                recomputed_by_rule[name].append(value)
        parameters = lane_parameters.get(lane, {})
        if not isinstance(parameters, dict) or set(parameters) != required_coefficients:
            physical_payload_errors.append(
                {
                    "reason": "movement coefficient schema mismatch",
                    "movement": lane,
                    "observed_coefficients": (
                        sorted(parameters) if isinstance(parameters, dict) else None
                    ),
                    "required_coefficients": sorted(required_coefficients),
                }
            )
            continue
        for name, raw_value in parameters.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = float("nan")
            if not np.isfinite(value) or value <= 0.0:
                physical_payload_errors.append(
                    {
                        "reason": "nonpositive or nonfinite fitted coefficient",
                        "movement": lane,
                        "coefficient": name,
                        "value": raw_value,
                    }
                )
    if set(declared_rule_scores) != set(MANUSCRIPT_RULE_NAMES):
        physical_payload_errors.append(
            {
                "reason": "top-level rule schema mismatch",
                "observed_rules": (
                    sorted(declared_rule_scores_raw)
                    if isinstance(declared_rule_scores_raw, dict)
                    else None
                ),
            }
        )
    elif all(recomputed_by_rule.values()):
        for name in MANUSCRIPT_RULE_NAMES:
            expected = float(np.mean(recomputed_by_rule[name]))
            try:
                observed = float(declared_rule_scores[name])
            except (TypeError, ValueError):
                observed = float("nan")
            if not np.isfinite(observed) or not np.isclose(
                observed, expected, rtol=0.0, atol=1e-12
            ):
                physical_payload_errors.append(
                    {
                        "reason": "top-level rule mean mismatch",
                        "rule": name,
                        "observed": declared_rule_scores.get(name),
                        "recomputed": expected,
                    }
                )
    if cells:
        recomputed_physical = float(np.mean(cells))
        if not np.isclose(
            physical_score, recomputed_physical, rtol=0.0, atol=1e-12
        ):
            physical_payload_errors.append(
                {
                    "reason": "physical mean mismatch",
                    "observed": physical.get("score"),
                    "recomputed": recomputed_physical,
                }
            )
        recomputed_joint_pass = all(value >= 1.0 - 1e-12 for value in cells)
        if physical.get("joint_pass") is not recomputed_joint_pass:
            physical_payload_errors.append(
                {
                    "reason": "strict joint-pass mismatch",
                    "observed": physical.get("joint_pass"),
                    "recomputed": recomputed_joint_pass,
                }
            )
    required_verifier_config = protocol["method_contract"].get("verifier")
    if physical.get("config") != required_verifier_config:
        physical_payload_errors.append(
            {
                "reason": "verifier configuration mismatch",
                "observed": physical.get("config"),
                "required": required_verifier_config,
            }
        )
    if physical_payload_errors:
        mismatches["selected_enhanced_physics.recomputation"] = {
            "errors": physical_payload_errors,
            "required": (
                "complete movement-by-seven-rule payload, equal means, exact "
                "joint-pass flag, frozen verifier config, and positive fitted "
                "coefficients for every declared movement"
            ),
        }

    training_metrics = result.get("training_metrics", {})
    try:
        accuracy_score = float(training_metrics["macro_nonnegative_r2"])
        training_fitness = float(result["training_fitness"])
    except (KeyError, TypeError, ValueError):
        accuracy_score = float("nan")
        training_fitness = float("nan")
    expected_fitness = float(
        min(max(accuracy_score + physical_score, 0.0), 2.0)
    )
    if (
        not np.isfinite(accuracy_score)
        or not np.isfinite(training_fitness)
        or not np.isclose(training_fitness, expected_fitness, rtol=0.0, atol=1e-12)
    ):
        mismatches["training_fitness_contract"] = {
            "observed": {
                "macro_nonnegative_r2": training_metrics.get(
                    "macro_nonnegative_r2"
                ),
                "physical_score": physical.get("score"),
                "training_fitness": result.get("training_fitness"),
            },
            "required": (
                "min(max(macro_nonnegative_r2 + physical_score, 0), 2)"
            ),
        }

    prompt_audit = result.get("prompt_audit", {})
    required_prompt_checks = (
        "single_physical_contract_per_prompt",
        "single_output_schema_per_prompt",
        "output_schema_is_last_in_every_prompt",
        "mutation_has_parent",
        "training_metric_feedback_absent",
        "finite_nonnegative_low_demand_present",
        "exact_zero_flow_absent",
        "numeric_r9_threshold_absent",
        "approach_supervision_disclosed",
        "llm_does_not_fit_coefficients",
    )
    failed_prompt_checks = [
        name for name in required_prompt_checks if prompt_audit.get(name) is not True
    ]
    if (
        int(prompt_audit.get("successful_prompt_calls", 0)) <= 0
        or int(prompt_audit.get("mutation_prompt_calls", 0)) <= 0
        or failed_prompt_checks
    ):
        mismatches["prompt_audit"] = {
            "observed": prompt_audit,
            "required": {
                "successful_prompt_calls": ">0",
                "mutation_prompt_calls": ">0",
                "true_checks": list(required_prompt_checks),
            },
        }

    content_archive = (
        result.get("llm_runtime", {}).get("content_archive_audit", {})
    )
    required_archive = {
        "status": "pass",
        "instantiated_prompt_content_archived": True,
        "normalized_assistant_response_content_archived": True,
        "prompt_and_response_sha256_verified": True,
        "api_key_or_authorization_fields_present": False,
        "model_identifier_present_in_every_successful_record": True,
        "exact_snapshot_inference_from_alias_permitted": False,
    }
    if any(content_archive.get(key) != expected for key, expected in required_archive.items()):
        mismatches["llm_runtime.content_archive_audit"] = {
            "observed": content_archive,
            "required": required_archive,
        }
    if content_archive.get("requested_model_aliases") != [
        V16_CONTRACT.requested_model_alias
    ]:
        mismatches["llm_runtime.requested_model_aliases"] = {
            "observed": content_archive.get("requested_model_aliases"),
            "required": [V16_CONTRACT.requested_model_alias],
        }
    returned_models = content_archive.get("returned_model_identifiers")
    if not isinstance(returned_models, list) or not returned_models or any(
        not isinstance(value, str) or not value.strip() for value in returned_models
    ):
        mismatches["llm_runtime.returned_model_identifiers"] = {
            "observed": returned_models,
            "required": "one or more nonempty identifiers copied from API responses",
        }

    restart_audit = result.get("optimizer_restart_composition_audit", {})
    required_restart = {
        "status": "pass",
        "restarts_per_approach_block": V16_CONTRACT.optimizer_restarts,
        "parent_warm_start_replaces_one_local_slot": True,
        "external_warm_start_allowed": False,
    }
    if any(restart_audit.get(key) != expected for key, expected in required_restart.items()):
        mismatches["optimizer_restart_composition_audit"] = {
            "observed": restart_audit,
            "required": required_restart,
        }
    epsilon_diagnostic = result.get("epsilon_parsimony_diagnostic", {})
    if epsilon_diagnostic.get("diagnostic_only_not_used_for_evolution") is not True:
        mismatches["epsilon_parsimony_diagnostic"] = {
            "observed": epsilon_diagnostic,
            "required": "diagnostic_only_not_used_for_evolution=true",
        }
    if not result.get("expression") or not result.get("lane_parameters"):
        mismatches["frozen_prediction_payload"] = {
            "observed": "missing expression or lane_parameters",
            "required": "both present",
        }
    if mismatches:
        raise RuntimeError(
            f"I{intersection} failed V16 frozen-result integrity checks: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _check_sources_unchanged(protocol: dict[str, Any]) -> None:
    observed = _source_hashes()
    if observed != protocol["source_sha256"]:
        changed = sorted(
            key
            for key in set(observed) | set(protocol["source_sha256"])
            if observed.get(key) != protocol["source_sha256"].get(key)
        )
        raise RuntimeError("V16 source changed after freeze: " + ", ".join(changed))


def _run_one(
    *, intersection: int, data_dir: Path, output_root: Path, env: dict[str, str]
) -> dict[str, Any]:
    parent = output_root / f"intersection_{intersection:02d}"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / "run_01"
    stdout_path = parent / "run_01.stdout.log"
    stderr_path = parent / "run_01.stderr.log"
    started_utc = _now()
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "methods.cosydelay.engine.training_protocol.run_p10g10_training",
                "--intersection",
                str(intersection),
                "--data-dir",
                str(data_dir),
                "--output",
                str(output),
            ],
            cwd=GMINI,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=False,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    return {
        "intersection_id": intersection,
        "started_utc": started_utc,
        "completed_utc": _now(),
        "return_code": int(completed.returncode),
        "wall_seconds": time.perf_counter() - started,
        "result_exists": (output / "result.json").is_file(),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def main() -> int:
    args = arguments()
    root = args.output_root.resolve()
    data_dir = args.data_dir.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite or resume: {root}")
    validate_v16_contract()
    validate_v16_source_manifest()
    training_paths = {
        str(intersection): (
            data_dir / f"Intersection_{intersection}_Train.jsonl"
        ).resolve()
        for intersection in INTERSECTIONS
    }
    missing = [str(path) for path in training_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Training files: " + ", ".join(missing))

    previous_key = os.environ.get("LLM_API_KEY")
    try:
        os.environ["LLM_API_KEY"] = recover_api_key()
        llm = load_llm_config()
        observed_sampling = {
            "model": llm.model,
            "temperature": llm.temperature,
            "top_p": llm.top_p,
            "max_tokens": llm.max_tokens,
            "seed": llm.seed,
        }
        if observed_sampling != V16_POLICY.llm_sampling:
            raise RuntimeError(
                "LLM configuration changed: "
                f"{observed_sampling} != {V16_POLICY.llm_sampling}"
            )
        root.mkdir(parents=True, exist_ok=False)
        protocol = {
            "schema_version": 2,
            "status": "frozen_before_any_v16_formal_api_call",
            "frozen_utc": _now(),
            "method_id": V16_POLICY.method_id,
            "policy": V16_POLICY.to_dict(),
            "method_contract": V16_CONTRACT.to_dict(),
            "intersections": list(INTERSECTIONS),
            "runs_per_intersection": 1,
            "population": V16_CONTRACT.population,
            "generations": V16_CONTRACT.generations,
            "intersection_execution": "sequential",
            "approach_workers_cap": V16_CONTRACT.approach_workers_cap,
            "selection_split": "Training only",
            "validation_allowed": False,
            "existing_test_allowed_during_search": False,
            "test_sha256": None,
            "project_wide_existing_test_previously_seen": True,
            "fresh_unseen_test_required_for_strictly_blind_claim": True,
            "llm": {
                "provider_endpoint": llm.endpoint,
                "request_path": llm.request_path,
                "sampling": V16_POLICY.llm_sampling,
                "sampling_parameter_sources": llm.sampling_parameter_sources,
                "api_key_recorded": False,
            },
            "python": sys.version,
            "packages": _package_versions(),
            "source_sha256": _source_hashes(),
            "training_sha256": {
                key: {"file_name": path.name, "sha256": sha256_file(path)}
                for key, path in training_paths.items()
            },
        }
        _write(root / "FROZEN_PROTOCOL.json", protocol)
        protocol = _read(root / "FROZEN_PROTOCOL.json")
        status = {
            "status": "running",
            "started_utc": _now(),
            "method_id": V16_POLICY.method_id,
            "intersections": list(INTERSECTIONS),
            "population": V16_CONTRACT.population,
            "generations": V16_CONTRACT.generations,
            "runs": [],
            "frozen_protocol_sha256": sha256_file(root / "FROZEN_PROTOCOL.json"),
        }
        _write(root / "RUN_STATUS.json", status)
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        for intersection in INTERSECTIONS:
            _check_sources_unchanged(protocol)
            train_path = training_paths[str(intersection)]
            if sha256_file(train_path) != protocol["training_sha256"][
                str(intersection)
            ]["sha256"]:
                raise RuntimeError(f"I{intersection} Training file changed after freeze")
            status.update(active_intersection=intersection, active_started_utc=_now())
            _write(root / "RUN_STATUS.json", status)
            run_status = _run_one(
                intersection=intersection,
                data_dir=data_dir,
                output_root=root,
                env=child_env,
            )
            status["runs"].append(run_status)
            status.pop("active_intersection", None)
            status.pop("active_started_utc", None)
            _write(root / "RUN_STATUS.json", status)
            if run_status["return_code"] or not run_status["result_exists"]:
                status.update(status="failed", failed_intersection=intersection)
                _write(root / "RUN_STATUS.json", status)
                _write(
                    root / "BATCH_COMPLETE.json",
                    {
                        "status": "failed",
                        "completed_utc": _now(),
                        "failed_intersection": intersection,
                    },
                )
                return 1

        _check_sources_unchanged(protocol)
        results = []
        result_hashes = {}
        integrity_errors = []
        for intersection in INTERSECTIONS:
            path = root / f"intersection_{intersection:02d}" / "run_01" / "result.json"
            result = _read(path)
            try:
                validate_formal_result(result, protocol, intersection=intersection)
            except Exception as exc:
                integrity_errors.append(
                    {
                        "intersection_id": intersection,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            results.append(result)
            result_hashes[str(intersection)] = sha256_file(path)
        if integrity_errors:
            status.update(status="failed_integrity")
            _write(root / "RUN_STATUS.json", status)
            _write(
                root / "BATCH_COMPLETE.json",
                {
                    "status": "failed_integrity",
                    "completed_utc": _now(),
                    "integrity_errors": integrity_errors,
                },
            )
            return 1

        aggregate = {
            "schema_version": 2,
            "scope": "full Training search metrics; not Test generalization",
            "intersections": list(INTERSECTIONS),
            "mean_raw_approach_macro_r2": float(
                np.mean(
                    [
                        item["formal_comparison_training_metrics"][
                            "raw_approach_macro_r2"
                        ]
                        for item in results
                    ]
                )
            ),
            "mean_pooled_rmse": float(
                np.mean(
                    [
                        item["formal_comparison_training_metrics"]["pooled_rmse"]
                        for item in results
                    ]
                )
            ),
            "mean_pooled_mae": float(
                np.mean(
                    [
                        item["formal_comparison_training_metrics"]["pooled_mae"]
                        for item in results
                    ]
                )
            ),
            "mean_winner_physical_score": float(
                np.mean(
                    [item["selected_enhanced_physics"]["score"] for item in results]
                )
            ),
            "strict_joint_passing_winners": int(
                sum(
                    bool(item["selected_enhanced_physics"].get("joint_pass"))
                    for item in results
                )
            ),
            "mean_invalid_expression_ratio": float(
                np.mean(
                    [
                        item["expression_attempt_summary"]["invalid_output_ratio"]
                        for item in results
                    ]
                )
            ),
            "total_api_attempts": int(
                sum(item["llm_attempt_summary"]["api_attempts"] for item in results)
            ),
            "total_wall_seconds_sum": float(
                sum(float(item["wall_seconds"]) for item in results)
            ),
        }
        _write(root / "TRAINING_AGGREGATE.json", aggregate)
        status.update(status="completed", completed_utc=_now())
        _write(root / "RUN_STATUS.json", status)
        _write(
            root / "BATCH_COMPLETE.json",
            {
                "schema_version": 2,
                "status": "complete",
                "completed_utc": _now(),
                "formal_training_runs": 6,
                "frozen_protocol_sha256": sha256_file(
                    root / "FROZEN_PROTOCOL.json"
                ),
                "training_aggregate_sha256": sha256_file(
                    root / "TRAINING_AGGREGATE.json"
                ),
                "result_sha256": result_hashes,
                "test_opened_during_search": False,
                "fresh_unseen_test_still_required_for_strictly_blind_claim": True,
            },
        )
        return 0
    finally:
        if previous_key is None:
            os.environ.pop("LLM_API_KEY", None)
        else:
            os.environ["LLM_API_KEY"] = previous_key


if __name__ == "__main__":
    raise SystemExit(main())
