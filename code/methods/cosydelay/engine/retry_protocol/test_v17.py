"""Regression tests for the isolated V17 cross-batch retry repair."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from methods.cosydelay.engine.training_protocol.contract import V16_CONTRACT
from methods.cosydelay.engine.training_protocol.policy import V16_POLICY
from methods.cosydelay.engine.training_protocol.prompt import OUTPUT_SCHEMA

from .contract import V17_CONTRACT, validate_v17_contract
from .integration import install_cross_batch_legality_repair
from .policy import V17_POLICY
from .prompt import (
    append_cross_batch_correction,
    correction_contains_supervision_metrics,
    format_cross_batch_legality_correction,
)
from .run_p10g10_training import validate_retry_audit


def test_contract_diff_is_retry_only():
    validate_v17_contract()
    assert V17_CONTRACT.population == V16_CONTRACT.population == 10
    assert V17_CONTRACT.generations == V16_CONTRACT.generations == 10
    assert V17_CONTRACT.optimizer_restarts == V16_CONTRACT.optimizer_restarts == 10
    assert V17_CONTRACT.optimizer == V16_CONTRACT.optimizer
    assert V17_POLICY.fitness_definition == V16_POLICY.fitness_definition
    assert V17_POLICY.coefficient_bounds == V16_POLICY.coefficient_bounds
    assert V17_POLICY.llm_sampling == V16_POLICY.llm_sampling
    assert not V17_CONTRACT.generation_budget_changed_from_v16
    assert not V17_CONTRACT.optimizer_contract_changed_from_v16
    assert not V17_CONTRACT.fitness_contract_changed_from_v16
    assert not V17_CONTRACT.physical_contract_changed_from_v16


def test_contract_validator_rejects_algorithm_drift(monkeypatch):
    import methods.cosydelay.engine.retry_protocol.contract as module

    monkeypatch.setattr(
        module,
        "V17_POLICY",
        replace(V17_POLICY, optimizer_restarts=9),
    )
    with pytest.raises(RuntimeError, match="outside the declared retry repair"):
        module.validate_v17_contract()


def test_prompt_correction_is_compact_legality_only_and_schema_last():
    base_prompt = "ROLE AND TASK\n\n" + OUTPUT_SCHEMA + "\n"
    repaired = append_cross_batch_correction(
        base_prompt,
        reason="coefficient-role ambiguity: use different coefficients",
        expression="Cycle_Time * exp(a3 / GR_phase) * flow_lane**a3",
        sequence=12,
    )
    assert repaired.count(OUTPUT_SCHEMA) == 1
    assert repaired.rstrip().endswith(OUTPUT_SCHEMA)
    assert repaired.count("CROSS-BATCH LEGALITY CORRECTION:") == 1
    assert "Correction sequence: 12." in repaired
    correction = repaired.split("CROSS-BATCH LEGALITY CORRECTION:", 1)[1]
    correction = correction.rsplit(OUTPUT_SCHEMA, 1)[0]
    assert not correction_contains_supervision_metrics(correction)


@pytest.mark.parametrize(
    "reason",
    [
        "Training R2 was low",
        "validation RMSE changed",
        "test MAE changed",
        "Fitness tie",
    ],
)
def test_prompt_rejects_metric_or_split_supervision(reason):
    with pytest.raises(ValueError, match="contains supervision"):
        format_cross_batch_legality_correction(
            reason=reason,
            expression="Cycle_Time * a1",
            sequence=2,
        )


def test_cross_batch_failure_is_injected_once_then_cleared():
    prompts: list[str] = []
    call = {"number": 0}

    def parse_llm_response(response):
        return response, None, ""

    def validator(expression):
        if expression == "BAD":
            return False, "coefficient-role ambiguity: use separate coefficients"
        return True, "Expression passed grammar and coefficient-role checks"

    def run_llm(prompt):
        prompts.append(prompt)
        call["number"] += 1
        return "GOOD" if "CROSS-BATCH LEGALITY CORRECTION:" in prompt else "BAD"

    def generator(*args, **kwargs):
        del args, kwargs
        response = adaptation.run_llm("BASE\n\n" + OUTPUT_SCHEMA + "\n")
        passed, reason = adaptation.validate_candidate_expression(response)
        if not passed:
            adaptation._write_expression_attempt_audit(
                {
                    "status": "validator_rejected",
                    "expression": response,
                    "reason": reason,
                }
            )
            raise RuntimeError(reason)
        return response, "", ""

    adaptation = SimpleNamespace(
        run_llm=run_llm,
        validate_candidate_expression=validator,
        parse_llm_response=parse_llm_response,
        _write_expression_attempt_audit=lambda record: None,
    )
    population = SimpleNamespace(
        safe_generate_universal_lane_expression=generator,
    )
    with install_cross_batch_legality_repair(
        population_module=population,
        adaptation_module=adaptation,
    ) as audit:
        with pytest.raises(RuntimeError):
            population.safe_generate_universal_lane_expression()
        assert population.safe_generate_universal_lane_expression()[0] == "GOOD"
    assert len(prompts) == 2
    assert "CROSS-BATCH LEGALITY CORRECTION:" not in prompts[0]
    assert prompts[1].count("CROSS-BATCH LEGALITY CORRECTION:") == 1
    assert prompts[1].rstrip().endswith(OUTPUT_SCHEMA)
    summary = validate_retry_audit(audit)
    assert summary["failed_generation_batches"] == 1
    assert summary["correction_injections"] == 1
    assert summary["successful_correction_recoveries"] == 1
    assert not summary["trailing_failure_pending"]


def test_real_generator_repairs_exhausted_role_conflict_next_batch(monkeypatch):
    import expression_adaptation_lane as adaptation

    bad = "Cycle_Time * flow_lane**a1 * exp(a1 / GR_phase)"
    good = "Cycle_Time * (a1 + a2 * flow_lane / GR_phase)"
    prompts: list[str] = []

    def response(expression):
        return (
            "### Expression\n"
            f"y = {expression}\n"
            "### Explanation\ncompact traffic-delay response"
        )

    def run_llm(prompt):
        prompts.append(prompt)
        if "CROSS-BATCH LEGALITY CORRECTION:" in prompt:
            return response(good)
        return response(bad)

    def validator(expression):
        if expression == bad:
            return False, "coefficient-role ambiguity: a1 is power and exp"
        return True, "Expression passed grammar and coefficient-role checks"

    monkeypatch.setattr(
        adaptation,
        "build_universal_lane_init_prompt",
        lambda *args, **kwargs: "REAL GENERATOR BASE\n\n" + OUTPUT_SCHEMA + "\n",
    )
    monkeypatch.setattr(adaptation, "run_llm", run_llm)
    monkeypatch.setattr(adaptation, "validate_candidate_expression", validator)
    monkeypatch.setattr(
        adaptation, "_write_expression_attempt_audit", lambda record: None
    )
    population = SimpleNamespace(
        safe_generate_universal_lane_expression=(
            adaptation.safe_generate_universal_lane_expression
        )
    )
    kwargs = {
        "feature_explanations": {},
        "universal_features": ["flow_lane", "GR_phase", "Cycle_Time"],
        "mutation_type": "initial",
        "intersection_id": 2,
        "max_retries": 2,
        "include_physical_knowledge": True,
        "enforce_physical_prefilter": True,
        "excluded_expressions": [],
    }
    with install_cross_batch_legality_repair(
        population_module=population,
        adaptation_module=adaptation,
    ) as audit:
        with pytest.raises(RuntimeError):
            population.safe_generate_universal_lane_expression(**kwargs)
        generated = population.safe_generate_universal_lane_expression(**kwargs)
    assert generated[0] == good
    assert len(prompts) == 3
    assert all(
        "CROSS-BATCH LEGALITY CORRECTION:" not in prompt
        for prompt in prompts[:2]
    )
    assert prompts[2].count("CROSS-BATCH LEGALITY CORRECTION:") == 1
    assert prompts[2].rstrip().endswith(OUTPUT_SCHEMA)
    summary = validate_retry_audit(audit)
    assert summary["failed_generation_batches"] == 1
    assert summary["successful_correction_recoveries"] == 1


def test_retry_audit_fails_when_correction_state_does_not_follow_failure():
    records = [
        {
            "event": "generation_batch_failed",
            "sequence": 1,
            "correction_injected": False,
            "last_reason": "coefficient-role ambiguity",
        },
        {
            "event": "generation_batch_succeeded",
            "sequence": 2,
            "correction_injected": False,
        },
    ]
    with pytest.raises(RuntimeError, match="state-machine audit failed"):
        validate_retry_audit(records)
