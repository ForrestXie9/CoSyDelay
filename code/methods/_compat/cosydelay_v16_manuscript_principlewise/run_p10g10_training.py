"""Run one frozen V16 P10/G10 Training-only search from scratch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from llm_config import load_llm_config
from methods.cosydelay_v9_clean_from_scratch.diagnostics import summarize_compute_efficiency
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import (
    DEFAULT_DATA_DIR,
    sha256_file,
    utc_now,
)
from methods.cosydelay_v10_accuracy_first_parsimony_alllog import (
    run_final_prompt_pilot as engine,
)

from .contract import V16_CONTRACT, validate_v16_contract
from .diagnostics import audit_parameter_quality_log_coordinates
from .integration import install_v16_candidate
from .physics import MANUSCRIPT_RULE_NAMES
from .policy import V16_POLICY
from .run_training_pilot import (
    _diagnostic_epsilon_parsimony,
    _force_principlewise_evolution,
    _history_principle_summary,
    _prompt_audit,
)
from .source_manifest import GMINI, V16_SOURCE_FILES, validate_v16_source_manifest


POPULATION = V16_CONTRACT.population
GENERATIONS = V16_CONTRACT.generations


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _audit_content_integrity(records: list[dict], *, secret: str = "") -> dict:
    """Fail closed unless every successful call preserves its exact I/O text."""
    successful = [item for item in records if item.get("status") == "success"]
    errors: list[dict] = []
    for index, item in enumerate(successful, start=1):
        prompt = item.get("prompt")
        response = item.get("response")
        for field, content in (("prompt", prompt), ("response", response)):
            expected = item.get(f"{field}_sha256")
            observed = _sha256_text(content) if isinstance(content, str) else None
            if not content or observed != expected:
                errors.append(
                    {
                        "successful_record": index,
                        "field": field,
                        "expected_sha256": expected,
                        "observed_sha256": observed,
                    }
                )
        lowered_keys = {str(key).lower() for key in item}
        if {"api_key", "authorization"} & lowered_keys:
            errors.append(
                {
                    "successful_record": index,
                    "field": "secret_metadata",
                    "observed_keys": sorted(lowered_keys),
                }
            )
        if secret and secret in json.dumps(item, ensure_ascii=False):
            errors.append(
                {
                    "successful_record": index,
                    "field": "unredacted_api_key_content",
                }
            )
        for field in ("model", "response_model"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                errors.append(
                    {
                        "successful_record": index,
                        "field": f"missing_{field}",
                    }
                )
    if not successful or errors:
        raise RuntimeError(
            "V16 LLM content archive is incomplete or inconsistent: "
            f"successful={len(successful)}, errors={errors}"
        )
    return {
        "status": "pass",
        "successful_records": len(successful),
        "instantiated_prompt_content_archived": True,
        "normalized_assistant_response_content_archived": True,
        "raw_provider_response_envelope_archived": False,
        "prompt_and_response_sha256_verified": True,
        "api_key_or_authorization_fields_present": False,
        "requested_model_aliases": sorted(
            {str(item["model"]).strip() for item in successful}
        ),
        "returned_model_identifiers": sorted(
            {str(item["response_model"]).strip() for item in successful}
        ),
        "model_identifier_present_in_every_successful_record": True,
        "exact_snapshot_inference_from_alias_permitted": False,
    }


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _source_hashes() -> dict[str, str]:
    return {
        (
            str(path.relative_to(GMINI)).replace("\\", "/")
            if path.is_relative_to(GMINI)
            else str(path)
        ): sha256_file(path)
        for path in V16_SOURCE_FILES
    }


def _restart_composition_audit(
    history: list[dict], *, population: int = POPULATION, generations: int = GENERATIONS
) -> dict:
    """Verify the declared ten-start policy for every fitted approach block."""
    fitted_candidates = 0
    approach_blocks = 0
    initialization_blocks = 0
    mutation_blocks = 0
    warm_mutation_blocks = 0
    unexpected = []
    initial_expected = ["official_local_replay"] * 9 + ["sobol_role_wide"]
    warm_expected = (
        ["parent_warm"]
        + ["official_local_replay"] * 8
        + ["sobol_role_wide"]
    )
    for item in history:
        if item.get("event") != "evaluated":
            continue
        fit = (item.get("evaluation_details") or {}).get("fit") or {}
        if not fit:
            unexpected.append(
                {
                    "candidate_id": item.get("candidate_id"),
                    "reason": "missing fit diagnostics",
                }
            )
            continue
        fitted_candidates += 1
        generation = int(item.get("generation", -1))
        for approach in fit.get("approaches", []):
            approach_blocks += 1
            kinds = [
                str(restart.get("start_kind"))
                for restart in approach.get("restarts", [])
            ]
            if generation == 0:
                initialization_blocks += 1
                expected = initial_expected
            elif bool(fit.get("parent_warm_start_available")):
                mutation_blocks += 1
                warm_mutation_blocks += 1
                expected = warm_expected
            else:
                mutation_blocks += 1
                expected = initial_expected
            if int(fit.get("n_restarts", -1)) != 10 or kinds != expected:
                unexpected.append(
                    {
                        "candidate_id": item.get("candidate_id"),
                        "generation": generation,
                        "approach": approach.get("approach"),
                        "n_restarts": fit.get("n_restarts"),
                        "observed": kinds,
                        "expected": expected,
                    }
                )
    expected_candidates = int(population) * (int(generations) + 1)
    if fitted_candidates != expected_candidates:
        unexpected.append(
            {
                "reason": "fitted candidate count mismatch",
                "observed": fitted_candidates,
                "expected": expected_candidates,
            }
        )
    if unexpected:
        raise RuntimeError(f"V16 optimizer restart composition drift: {unexpected}")
    return {
        "status": "pass",
        "population": int(population),
        "generations": int(generations),
        "fitted_candidates": fitted_candidates,
        "approach_blocks": approach_blocks,
        "initialization_blocks": initialization_blocks,
        "mutation_blocks": mutation_blocks,
        "warm_mutation_blocks": warm_mutation_blocks,
        "restarts_per_approach_block": 10,
        "initialization_composition": initial_expected,
        "warm_mutation_composition": warm_expected,
        "parent_warm_start_replaces_one_local_slot": True,
        "external_warm_start_allowed": False,
    }


def main() -> int:
    _configure_utf8_console()
    args = arguments()
    validate_v16_contract()
    validate_v16_source_manifest()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    train_path = (
        args.data_dir / f"Intersection_{args.intersection}_Train.jsonl"
    ).resolve()
    if not train_path.is_file():
        raise FileNotFoundError(train_path)

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

    started_utc = utc_now()
    previous = (
        engine.PILOT_POPULATION,
        engine.PILOT_GENERATIONS,
        engine.install_v10_candidate,
        engine.V10_POLICY,
        engine._prompt_audit,
        engine.evolve_universal_lane_expression,
        engine.select_epsilon_parsimonious,
        sys.argv[:],
    )
    try:
        engine.PILOT_POPULATION = POPULATION
        engine.PILOT_GENERATIONS = GENERATIONS
        engine.install_v10_candidate = install_v16_candidate
        engine.V10_POLICY = V16_POLICY
        engine._prompt_audit = _prompt_audit
        engine.evolve_universal_lane_expression = _force_principlewise_evolution(
            engine.evolve_universal_lane_expression
        )
        original_epsilon_diagnostic = engine.select_epsilon_parsimonious
        engine.select_epsilon_parsimonious = lambda evaluations: (
            _diagnostic_epsilon_parsimony(
                evaluations, original_epsilon_diagnostic
            )
        )
        sys.argv = [
            str(engine.__file__),
            "--intersection",
            str(args.intersection),
            "--data-dir",
            str(args.data_dir),
            "--output",
            str(args.output),
        ]
        code = engine.main()
    finally:
        (
            engine.PILOT_POPULATION,
            engine.PILOT_GENERATIONS,
            engine.install_v10_candidate,
            engine.V10_POLICY,
            engine._prompt_audit,
            engine.evolve_universal_lane_expression,
            engine.select_epsilon_parsimonious,
            old_argv,
        ) = previous
        sys.argv = old_argv
    if code:
        return int(code)

    result_path = args.output / "result.json"
    history_path = args.output / "history.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    llm_audit_path = args.output / "llm_audit.jsonl"
    llm_records = [
        json.loads(line)
        for line in llm_audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    llm_content_archive = _audit_content_integrity(
        llm_records, secret=llm.api_key
    )
    restart_composition = _restart_composition_audit(history)
    serialized_history = json.dumps(history, ensure_ascii=False)
    forbidden_probe_fields = (
        '"r9_probe_pass"',
        '"r9_probe_delays_seconds"',
        '"r9_constrained_restart_selection"',
        '"r9_feasible_restart_count"',
        '"r9_feasible_solution_available"',
    )
    leaked_probe_fields = [
        field for field in forbidden_probe_fields if field in serialized_history
    ]
    if leaked_probe_fields:
        raise RuntimeError(
            "superseded numerical-R9 diagnostics leaked into V16 history: "
            f"{leaked_probe_fields}"
        )
    budget = next(
        item for item in reversed(history) if item.get("event") == "search_budget_summary"
    )
    required_budget = {
        "population": POPULATION,
        "generations": GENERATIONS,
        "completed_generations": GENERATIONS,
        "candidate_evaluations": POPULATION * (GENERATIONS + 1),
    }
    observed_budget = {
        "population": result.get("population"),
        "generations": result.get("generations"),
        "completed_generations": int(budget["completed_generations"]),
        "candidate_evaluations": int(budget["candidate_evaluations"]),
    }
    if observed_budget != required_budget:
        raise RuntimeError(
            f"P10/G10 budget mismatch: {observed_budget} != {required_budget}"
        )
    if result.get("accessed_splits") != ["train"]:
        raise RuntimeError(f"split leakage detected: {result.get('accessed_splits')}")

    physical = result.get("selected_enhanced_physics", {})
    if tuple(physical.get("rule_order", ())) != MANUSCRIPT_RULE_NAMES:
        raise RuntimeError("winner is missing the declared seven-rule V16 audit")
    physical_score = float(physical.get("score", -1.0))
    if not 0.0 <= physical_score <= 1.0:
        raise RuntimeError(f"invalid V16 physical score: {physical_score}")

    metrics = dict(result["selected_training_metrics"])
    principle_summary = _history_principle_summary(history_path)
    v16_timing = principle_summary["v16_physics_timing"]
    result.update(principle_summary)
    result.setdefault("efficiency_timing", {}).update(
        {
            "enhanced_physics_wall_seconds_sum": v16_timing[
                "wall_seconds_sum"
            ],
            "v16_principlewise_physics_wall_seconds_sum": v16_timing[
                "wall_seconds_sum"
            ],
            "v16_principlewise_physics_wall_seconds_mean": v16_timing[
                "wall_seconds_mean"
            ],
            "compatibility_bypass_wall_seconds_sum": v16_timing[
                "compatibility_bypass_wall_seconds_sum"
            ],
            "v16_score_reused_as_enhanced_audit": True,
        }
    )
    result.update(
        {
            "status": "completed_v16_p10g10_training_only_search",
            "not_a_formal_result": False,
            "formal_protocol_complete": True,
            "method_id": V16_POLICY.method_id,
            "method_status": "v16_prospective_p10g10_i1_i6_confirmation",
            "policy": V16_POLICY.to_dict(),
            "method_contract": V16_CONTRACT.to_dict(),
            "formal_started_utc": started_utc,
            "formal_completed_utc": utc_now(),
            "train_file_name": train_path.name,
            "train_file_sha256": sha256_file(train_path),
            "selection_source": "full_training_evolution_only",
            "selection_order": [
                "paper_training_fitness",
                "raw_training_r2_on_exact_fitness_ties",
                "negative_training_rmse_on_remaining_ties",
            ],
            "evolution_selection": (
                "Training mean nonnegative approach R2 plus the equal mean "
                "over movement-by-seven-rule physical scores"
            ),
            "accuracy_supervision_level": "approach",
            "movement_delay_labels_available": False,
            "coefficient_parameterization_level": "movement",
            "paper_fitness_definition_changed": False,
            "fitness_changed_from_v15_binary_implementation": True,
            "fitness_contract_alignment": (
                "recovered_manuscript_R2_plus_principlewise_physical_score"
            ),
            "strict_joint_pass_role": "diagnostic_only_not_a_hard_gate",
            "population_update": V16_CONTRACT.population_update,
            "legacy_numeric_r9_probe_executed": False,
            "llm_training_metric_feedback_exposed": False,
            "validation_or_test_used_for_selection": False,
            "post_evolution_cv_reranking": False,
            "post_evolution_refit": False,
            "external_incumbent_allowed": False,
            "optimizer_restart_composition_audit": restart_composition,
            "initial_population_all_generated_in_current_run": True,
            "completed_generations": GENERATIONS,
            "expression": result["selected_expression"],
            "lane_parameters": result["selected_parameters"],
            "training_metrics": metrics,
            "formal_comparison_training_metrics": {
                "raw_approach_macro_r2": metrics["macro_raw_r2"],
                "pooled_rmse": metrics["pooled_rmse"],
                "pooled_mae": metrics["pooled_mae"],
            },
            "training_fitness": result["selected_training_fitness"],
            "parameter_quality_audit": audit_parameter_quality_log_coordinates(
                result["selected_expression"],
                result["selected_parameters"],
                V16_POLICY.coefficient_bounds,
            ),
            "compute_efficiency": summarize_compute_efficiency(history),
            "llm_runtime": {
                "provider_endpoint": llm.endpoint,
                "request_path": llm.request_path,
                "requested_model_aliases": llm_content_archive[
                    "requested_model_aliases"
                ],
                "returned_model_identifiers": llm_content_archive[
                    "returned_model_identifiers"
                ],
                "exact_model_snapshot": "unknown_not_exposed_by_provider",
                "sampling": V16_POLICY.llm_sampling,
                "sampling_parameter_sources": llm.sampling_parameter_sources,
                "api_key_recorded": False,
                "content_archive_audit": llm_content_archive,
            },
            "v16_source_sha256": _source_hashes(),
        }
    )
    result.pop("physically_passing_evaluations", None)
    result.pop("postfit_pass_rate", None)
    engine.write_json(result_path, result)
    print(
        f"I{args.intersection} V16 P10/G10: "
        f"R2={metrics['macro_raw_r2']:.6f}, "
        f"RMSE={metrics['pooled_rmse']:.6f}, "
        f"MAE={metrics['pooled_mae']:.6f}, "
        f"physics={physical_score:.6f}, "
        f"strict_joint={bool(physical.get('joint_pass'))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
