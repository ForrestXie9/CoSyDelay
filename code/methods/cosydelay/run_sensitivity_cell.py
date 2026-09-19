"""One CoSyDelay sensitivity cell: temperature × traffic-principle prompt ablation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

GMINI = Path(__file__).resolve().parents[2]
if str(GMINI) not in sys.path:
    sys.path.insert(0, str(GMINI))

import expression_rules
import expression_adaptation_lane
from methods.cosydelay_v16_manuscript_principlewise import prompt as v16_prompt
from methods.cosydelay_v16_manuscript_principlewise.run_training_pilot import (
    _diagnostic_epsilon_parsimony,
)
from methods.cosydelay_v16_manuscript_principlewise import integration as v16_integration
from methods.cosydelay_v16_manuscript_principlewise import run_p10g10_training as v16_run
from methods.cosydelay_v18_fit_timeout_guard import integration as v18_integration
from methods.cosydelay_v18_fit_timeout_guard import fit_timeout as v18_fit_timeout
from methods.cosydelay_v18_fit_timeout_guard import run_p10g10_training as v18_run
from methods.cosydelay_v17_cross_batch_retry import run_p10g10_training as v17_run
from methods.cosydelay_v18_fit_timeout_guard.contract import V18_CONTRACT
from methods.cosydelay_v18_fit_timeout_guard.policy import V18_POLICY
from methods import structural_diversity_population_v1 as pairwise_population
from methods.cosydelay import prompt as cosydelay_prompt
from methods.cosydelay.fitter import accelerated_fit_supervisor_main
from methods.cosydelay.regeneration import install_after_compatibility_layers
from methods.cosydelay_v19_accelerated_equivalent.llm_seed import install_reproducible_seed_schedule
from methods.cosydelay_v19_accelerated_equivalent.normalization import install_expression_normalization
from methods.cosydelay_v19_accelerated_equivalent.binary_fitness import install_binary_joint_fitness
from methods.cosydelay_v19_accelerated_equivalent.prefit_r4 import install_prefit_r4_gate

SENSITIVITY_DIR = (
    GMINI
    / "reviewer_revision_experiments"
    / "15_cosydelay_llm_prompt_temperature_sensitivity"
)

SENSITIVITY_POPULATION = 10
SENSITIVITY_GENERATIONS = 9
SENSITIVITY_BUDGET = 100
_PENDING_SENSITIVITY_WINNER: dict[str, object] = {}


def _record_from_history(expression: str, parameters, history: list) -> dict:
    for item in reversed(history):
        if item.get("event") != "evaluated":
            continue
        if str(item.get("expression")) != str(expression):
            continue
        details = item.get("evaluation_details", {}) or {}
        verifier = details.get("verifier", {}) or {}
        train_r2 = float(item.get("train_r2", details.get("train_r2", 0.0)))
        fitness = float(item.get("fitness", details.get("fitness", train_r2)))
        metrics = {
            "macro_nonnegative_r2": train_r2,
            "train_r2": train_r2,
            "train_rmse": float(item.get("train_rmse", details.get("train_rmse", 0.0))),
        }
        enhanced = verifier or details.get("clean_enhanced_physics", {}) or {
            "joint_pass": bool(item.get("physical_joint_pass", False)),
            "score": float(item.get("physical_score", 0.0)),
            "rule_scores": item.get("rule_scores", {}),
        }
        return {
            "expression": str(expression),
            "parameters": parameters,
            "metrics": metrics,
            "fitness": fitness,
            "enhanced_physics": enhanced,
            "details": details,
            "sensitivity_nonpassing_winner_fallback": True,
        }
    raise RuntimeError(
        f"no evaluated history record for sensitivity winner: {expression}"
    )


class _SensitivityEvaluations:
    """Allow sensitivity cells to finish when no joint-pass winner exists."""

    def __init__(self, storage: dict):
        self._storage = storage

    def get(self, key, default=None):
        if key in self._storage:
            return self._storage[key]
        pending = _PENDING_SENSITIVITY_WINNER
        if pending and str(key) == str(pending.get("expression")):
            record = _record_from_history(
                pending["expression"],
                pending["parameters"],
                pending["history"],
            )
            self._storage[str(key)] = record
            return record
        return default

    def __len__(self) -> int:
        return len(self._storage)

    def values(self):
        return self._storage.values()

    def __setitem__(self, key, value) -> None:
        self._storage[key] = value


def _install_prompt_treatment(instruction_on: bool):
    if instruction_on:
        init_builder = cosydelay_prompt.build_init_prompt
        regen_builder = cosydelay_prompt.build_regeneration_prompt
    else:
        if str(SENSITIVITY_DIR) not in sys.path:
            sys.path.insert(0, str(SENSITIVITY_DIR))
        from experiment import (  # noqa: WPS433
            build_no_instruction_init_prompt,
            build_no_instruction_regeneration_prompt,
        )

        init_builder = build_no_instruction_init_prompt
        regen_builder = build_no_instruction_regeneration_prompt
    v16_prompt.build_init_prompt = init_builder
    v16_prompt.build_mutation_prompt = regen_builder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intersection", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--paired-seed", type=int, required=True)
    parser.add_argument("--instruction", choices=("off", "on"), required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    instruction_on = args.instruction == "on"
    run_contract = replace(
        V18_CONTRACT,
        schema_version=15,
        method_id="cosydelay_llm_prompt_temperature_sensitivity",
        generations=SENSITIVITY_GENERATIONS,
        population=SENSITIVITY_POPULATION,
        successful_candidate_budget=SENSITIVITY_BUDGET,
        temperature=float(args.temperature),
        requested_seed=int(args.paired_seed),
    )
    run_policy = replace(
        V18_POLICY,
        method_id="cosydelay_llm_prompt_temperature_sensitivity",
        method_status="reviewer_cosydelay_prompt_temperature_sensitivity_not_main_performance",
        population=SENSITIVITY_POPULATION,
        generations=SENSITIVITY_GENERATIONS,
        llm_temperature=float(args.temperature),
        prompt_knowledge=instruction_on,
    )

    v16_run.GENERATIONS = SENSITIVITY_GENERATIONS
    v16_run.POPULATION = SENSITIVITY_POPULATION
    original_restart_audit = v16_run._restart_composition_audit
    original_guarded_fitter = v18_integration.GuardedCandidateFitter
    original_engine_evolve = v16_run.engine.evolve_universal_lane_expression
    original_install_v18 = v18_run.install_v18_candidate
    original_audit = v18_run.validate_fit_guard_audit

    class CoSyDelayGuardedCandidateFitter(v18_fit_timeout.GuardedCandidateFitter):
        def __init__(self, **kwargs):
            kwargs["supervisor_target"] = accelerated_fit_supervisor_main
            super().__init__(**kwargs)

        def fit(self, *args, **kwargs):
            diagnostics = kwargs.get("diagnostics")
            result = super().fit(*args, **kwargs)
            if isinstance(diagnostics, dict):
                diagnostics.setdefault(
                    "formal_method_adapter",
                    "cosydelay.fitter",
                )
            return result

    original_init_prompt = v16_prompt.build_init_prompt
    original_mutation_prompt = v16_prompt.build_mutation_prompt

    @contextmanager
    def install_cosydelay_sensitivity_candidate(**kwargs):
        with install_after_compatibility_layers(
            original_install_v18,
            pairwise_population,
            **kwargs,
        ) as runtime:
            with install_reproducible_seed_schedule(
                expression_adaptation_lane,
                base_seed=int(args.paired_seed),
                intersection_id=args.intersection,
            ):
                with install_expression_normalization(expression_adaptation_lane):
                    with install_prefit_r4_gate(expression_adaptation_lane):
                        with install_binary_joint_fitness(
                            pairwise_population,
                            runtime,
                        ):
                            runtime.clean.evaluations = _SensitivityEvaluations(
                                runtime.clean.evaluations
                            )
                            yield runtime

    def force_treatment_evolution(*evolve_args, **evolve_kwargs):
        evolve_kwargs["score_mode"] = "binary"
        evolve_kwargs["targeted_physical_feedback"] = False
        evolve_kwargs["prompt_knowledge"] = instruction_on
        evolve_kwargs["seed"] = int(args.paired_seed)
        evolve_kwargs["pop_size"] = SENSITIVITY_POPULATION
        evolve_kwargs["generations"] = SENSITIVITY_GENERATIONS
        expression, thought, explanation, parameters, history = original_engine_evolve(
            *evolve_args, **evolve_kwargs
        )
        _PENDING_SENSITIVITY_WINNER.clear()
        _PENDING_SENSITIVITY_WINNER.update(
            {
                "expression": expression,
                "parameters": parameters,
                "history": history,
            }
        )
        return expression, thought, explanation, parameters, history

    previous = (
        v18_run.V18_CONTRACT,
        v18_run.V18_POLICY,
        v18_run.validate_v18_contract,
        v18_integration.V18_CONTRACT,
        v18_integration.V18_POLICY,
        v18_integration.validate_v18_contract,
        v16_run.V16_POLICY,
        v16_run._restart_composition_audit,
        v18_integration.GuardedCandidateFitter,
        v16_run.engine.evolve_universal_lane_expression,
        v18_run.install_v18_candidate,
        v17_run._v17_prompt_audit,
        os.environ.get("LLM_TEMPERATURE"),
        expression_rules.MAX_COEFFICIENTS,
        v16_prompt.build_init_prompt,
        v16_prompt.build_mutation_prompt,
        v16_integration.V16_MAX_PROMPT_EXCLUSIONS,
    )
    old_temperature = previous[12]
    old_init_prompt = previous[14]
    old_mutation_prompt = previous[15]
    try:
        _install_prompt_treatment(instruction_on)
        if str(SENSITIVITY_DIR) not in sys.path:
            sys.path.insert(0, str(SENSITIVITY_DIR))
        from experiment import audit_prompts as sensitivity_audit_prompts  # noqa: WPS433
        v18_run.V18_CONTRACT = run_contract
        v18_run.V18_POLICY = run_policy
        v18_run.validate_v18_contract = lambda: None
        v18_integration.V18_CONTRACT = run_contract
        v18_integration.V18_POLICY = run_policy
        v18_integration.validate_v18_contract = lambda: None
        v16_run.V16_POLICY = run_policy
        os.environ["LLM_TEMPERATURE"] = str(args.temperature)
        expression_rules.MAX_COEFFICIENTS = expression_rules.MAX_PYTHON_AST_NODES
        v16_integration.V16_MAX_PROMPT_EXCLUSIONS = 0
        v17_run._v17_prompt_audit = lambda attempts: sensitivity_audit_prompts(
            attempts,
            instruction_on=instruction_on,
        )
        v18_run.validate_fit_guard_audit = lambda audit: original_audit(
            audit, expected_candidates=SENSITIVITY_BUDGET
        )
        v16_run._restart_composition_audit = lambda history: None
        v18_integration.GuardedCandidateFitter = CoSyDelayGuardedCandidateFitter
        v18_run.install_v18_candidate = install_cosydelay_sensitivity_candidate
        v16_run.engine.evolve_universal_lane_expression = force_treatment_evolution
        argv = [
            "--intersection",
            str(args.intersection),
            "--output",
            str(args.output),
        ]
        if args.data_dir is not None:
            argv += ["--data-dir", str(args.data_dir)]
        engine = v16_run.engine
        previous_engine = (
            engine.PILOT_POPULATION,
            engine.PILOT_GENERATIONS,
            engine.install_v10_candidate,
            engine.V10_POLICY,
            engine._prompt_audit,
            engine.select_epsilon_parsimonious,
            sys.argv[:],
        )
        try:
            engine.PILOT_POPULATION = SENSITIVITY_POPULATION
            engine.PILOT_GENERATIONS = SENSITIVITY_GENERATIONS
            engine.install_v10_candidate = v18_run.install_v18_candidate
            engine.V10_POLICY = v16_run.V16_POLICY
            engine._prompt_audit = v17_run._v17_prompt_audit
            original_epsilon = engine.select_epsilon_parsimonious
            engine.select_epsilon_parsimonious = lambda evaluations: (
                _diagnostic_epsilon_parsimony(evaluations, original_epsilon)
            )
            sys.argv = [sys.argv[0], *argv]
            code = engine.main()
        finally:
            (
                engine.PILOT_POPULATION,
                engine.PILOT_GENERATIONS,
                engine.install_v10_candidate,
                engine.V10_POLICY,
                engine._prompt_audit,
                engine.select_epsilon_parsimonious,
                sys.argv[:],
            ) = previous_engine
    finally:
        (
            v18_run.V18_CONTRACT,
            v18_run.V18_POLICY,
            v18_run.validate_v18_contract,
            v18_integration.V18_CONTRACT,
            v18_integration.V18_POLICY,
            v18_integration.validate_v18_contract,
            v16_run.V16_POLICY,
            v16_run._restart_composition_audit,
            v18_integration.GuardedCandidateFitter,
            v16_run.engine.evolve_universal_lane_expression,
            v18_run.install_v18_candidate,
            v17_run._v17_prompt_audit,
            old_temperature,
            expression_rules.MAX_COEFFICIENTS,
            old_init_prompt,
            old_mutation_prompt,
            v16_integration.V16_MAX_PROMPT_EXCLUSIONS,
        ) = previous
        if old_temperature is None:
            os.environ.pop("LLM_TEMPERATURE", None)
        else:
            os.environ["LLM_TEMPERATURE"] = old_temperature
        v16_prompt.build_init_prompt = old_init_prompt
        v16_prompt.build_mutation_prompt = old_mutation_prompt

    if code:
        return int(code)

    result_path = args.output / "result.json"
    history_path = args.output / "history.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    fitted_ids = {
        int(item["candidate_id"])
        for item in history
        if isinstance(item, dict) and item.get("candidate_id") is not None
    }
    if len(fitted_ids) != SENSITIVITY_BUDGET:
        raise RuntimeError(
            f"sensitivity budget mismatch: ids={len(fitted_ids)}, "
            f"required={SENSITIVITY_BUDGET}"
        )
    result.update(
        {
            "status": "complete_cosydelay_instruction_temperature_cell",
            "not_a_main_performance_result": True,
            "reviewer_experiment": "cosydelay_instruction_by_temperature",
            "intersection_id": args.intersection,
            "run_id": args.run_id,
            "base_seed": args.base_seed,
            "paired_seed": args.paired_seed,
            "traffic_principle_instruction": instruction_on,
            "temperature": float(args.temperature),
            "population": SENSITIVITY_POPULATION,
            "generations": SENSITIVITY_GENERATIONS,
            "candidate_evaluations_required": SENSITIVITY_BUDGET,
            "fitness_definition": "training_r2_plus_strict_joint_pass_binary",
            "selected_strict_joint_pass": bool(
                (result.get("selected_enhanced_physics") or {}).get("joint_pass", False)
            ),
            "validation_or_test_used": False,
            "test_file_opened": False,
        }
    )
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"CoSyDelay-sensitivity I{args.intersection} run={args.run_id} "
        f"instruction={args.instruction} T={args.temperature} complete",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
