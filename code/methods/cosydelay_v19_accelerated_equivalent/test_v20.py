from __future__ import annotations

from contextlib import contextmanager
import os
from types import SimpleNamespace

from methods.cosydelay_v16_manuscript_principlewise.prompt import OUTPUT_SCHEMA

from .paper_prompt import build_init_prompt, build_regeneration_prompt
from .prompt_audit import audit_v20_prompts
from .llm_seed import install_reproducible_seed_schedule
from .normalization import (
    install_expression_normalization,
    normalize_python_power_operator,
)
from .pairwise_population import (
    PROGRAM_SIDE_REGENERATION_NOVELTY_SCOPE,
    PROMPT_HISTORY_EXPRESSION_COUNT,
    _physical_principles_fraction,
)
from .binary_fitness import binary_joint_fitness, install_binary_joint_fitness
from .prefit_r4 import install_prefit_r4_gate, validate_exact_time_dimension
from .regeneration import (
    REGENERATION_CHECK_ATTEMPTS,
    install_after_compatibility_layers,
    with_initialization_fallback,
)


def test_v20_prompt_audit_accepts_initialization_and_regeneration() -> None:
    attempts = [
        {
            "status": "success",
            "prompt": build_init_prompt({}, [], intersection_id=1),
        },
        {
            "status": "success",
            "prompt": build_regeneration_prompt(
                "Cycle_Time * a1 / GR_phase",
                "",
                "",
                (True, "passed"),
                "large",
                [],
                intersection_id=1,
            ),
        },
    ]
    audit = audit_v20_prompts(attempts)
    assert audit["initialization_prompt_calls"] == 1
    assert audit["regeneration_prompt_calls"] == 1
    assert audit["output_schema_is_last_in_every_prompt"]
    assert all(item["prompt"].rstrip().endswith(OUTPUT_SCHEMA) for item in attempts)


def test_outer_generic_generator_uses_initialization_fallback() -> None:
    calls: list[tuple[str, object, object]] = []

    def installed_compatibility_generator(*args, **kwargs):
        operation = kwargs.get(
            "mutation_type", args[2] if len(args) > 2 else "initial"
        )
        parent = kwargs.get("base_expr", args[3] if len(args) > 3 else None)
        feedback = kwargs.get(
            "search_feedback", args[13] if len(args) > 13 else None
        )
        calls.append((operation, parent, feedback))
        if operation != "initial":
            raise RuntimeError("Failed to generate valid expression after all retries")
        return "fresh", "", "ok"

    generate = with_initialization_fallback(installed_compatibility_generator)
    result = generate(
        {},
        [],
        "large",
        "parent expression",
        "",
        "",
        (False, "failed"),
        1,
        search_feedback="parent feedback",
    )
    assert result == ("fresh", "", "ok")
    assert calls == [
        ("large", "parent expression", "parent feedback"),
        ("initial", None, None),
    ]


def test_provider_runtime_error_does_not_trigger_initialization() -> None:
    calls = 0

    def unavailable(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider unavailable")

    generate = with_initialization_fallback(unavailable)
    try:
        generate({}, [], "large")
    except RuntimeError as exc:
        assert str(exc) == "provider unavailable"
    else:
        raise AssertionError("provider error was swallowed")
    assert calls == 1


def test_every_regeneration_uses_the_same_global_exploration_contract() -> None:
    prompts = [
        build_regeneration_prompt(
            "Cycle_Time * (a1 + a2 * flow_lane / GR_phase)",
            "",
            "",
            (True, "passed"),
            kind,
            [],
            intersection_id=1,
        )
        for kind in ("small", "large")
    ]
    assert prompts[0] == prompts[1]
    assert "REGENERATION TASK:" in prompts[0]
    assert "Replace or fundamentally reorganize" in prompts[0]
    assert "algebraically equivalent rewrite is insufficient" in prompts[0]
    assert "No functional family is prescribed" in prompts[0]
    assert REGENERATION_CHECK_ATTEMPTS == 3


def test_fallback_is_installed_after_compatibility_wrapper_and_restored() -> None:
    def original_generator(*args, **kwargs):
        return "original", "", ""

    def final_compatibility_generator(*args, **kwargs):
        return "compatibility", "", ""

    population = SimpleNamespace(
        safe_generate_universal_lane_expression=original_generator
    )

    @contextmanager
    def inherited_install(**kwargs):
        assert kwargs["population_module"] is population
        population.safe_generate_universal_lane_expression = (
            final_compatibility_generator
        )
        try:
            yield "runtime"
        finally:
            population.safe_generate_universal_lane_expression = original_generator

    with install_after_compatibility_layers(
        inherited_install, population
    ) as runtime:
        assert runtime == "runtime"
        installed = population.safe_generate_universal_lane_expression
        assert installed.v20_initialization_fallback is True
        assert installed({}, [], "initial")[0] == "compatibility"

    assert population.safe_generate_universal_lane_expression is original_generator


def test_llm_seed_is_distinct_reproducible_and_restored() -> None:
    observed: list[int] = []
    module = SimpleNamespace(
        run_llm=lambda prompt: observed.append(int(os.environ["LLM_SEED"])) or prompt
    )
    original_environment_seed = os.environ.get("LLM_SEED")
    os.environ["LLM_SEED"] = "77"
    try:
        with install_reproducible_seed_schedule(
            module, base_seed=1000, intersection_id=2
        ) as state:
            assert module.run_llm("one") == "one"
            assert module.run_llm("two") == "two"
            assert state["calls"] == 2
        assert observed == [201001, 201002]
        assert os.environ["LLM_SEED"] == "77"
    finally:
        if original_environment_seed is None:
            os.environ.pop("LLM_SEED", None)
        else:
            os.environ["LLM_SEED"] = original_environment_seed


def test_caret_power_is_normalized_before_symbolic_parse_and_restored() -> None:
    original_parser = lambda response: (
        "Cycle_Time * flow_lane ^ a1 / GR_phase ^ a2",
        "",
        "physical explanation",
    )
    module = SimpleNamespace(parse_llm_response=original_parser)
    with install_expression_normalization(module) as state:
        expression, _, _ = module.parse_llm_response("raw archived response")
        assert expression == "Cycle_Time * flow_lane ** a1 / GR_phase ** a2"
        assert state["caret_power_conversions"] == 1
    assert module.parse_llm_response is original_parser
    assert normalize_python_power_operator("a1 ** flow_lane") == "a1 ** flow_lane"


def test_physical_compliance_is_displayed_as_fully_satisfied_principles() -> None:
    individual = {
        "physical_joint_pass": False,
        "rule_scores": {
            "R1": 1.0,
            "R2": 1.0,
            "R3": 1.0,
            "R4": 0.0,
            "R5": 1.0,
            "R6": 1.0,
            "R7": 1.0,
        },
    }
    assert _physical_principles_fraction(individual) == "6/7"


def test_full_history_duplicate_gate_does_not_disclose_formulas_to_prompt() -> None:
    assert PROGRAM_SIDE_REGENERATION_NOVELTY_SCOPE == "full_evaluated_history"
    assert PROMPT_HISTORY_EXPRESSION_COUNT == 0


def test_strict_joint_pass_is_the_only_physical_fitness_component() -> None:
    records = {"expr": {}}

    def fractional_evaluator(*args, **kwargs):
        details = {
            "verifier": {"score": 6 / 7, "joint_pass": False},
        }
        records["expr"].update({"fitness": 0.6934 + 6 / 7, "details": details})
        return 6 / 7, 0.6934, 0.6934 + 6 / 7, {}, "failed R4", details

    module = SimpleNamespace(evaluate_expression_with_fitting=fractional_evaluator)
    runtime = SimpleNamespace(evaluations=records)
    with install_binary_joint_fitness(module, runtime):
        physical, accuracy, fitness, *_ = module.evaluate_expression_with_fitting(
            "expr"
        )
    assert physical == 0.0
    assert accuracy == 0.6934
    assert fitness == 0.6934
    assert records["expr"]["fitness"] == 0.6934
    assert records["expr"]["details"]["principlewise_diagnostic_fraction"] == 6 / 7
    assert binary_joint_fitness(0.6934, True) == 1.6934


def test_r4_rejects_mixed_time_dimension_before_fitting() -> None:
    bad = "a1 + a2 * flow_lane / GR_phase + a3 * Cycle_Time / GR_phase"
    good = "Cycle_Time * (a1 + a2 * flow_lane / GR_phase)"
    assert validate_exact_time_dimension(bad)[0] is False
    assert validate_exact_time_dimension(good)[0] is True

    calls: list[str] = []
    module = SimpleNamespace(
        validate_candidate_expression=lambda expression: (True, "legal")
    )
    original = module.validate_candidate_expression
    with install_prefit_r4_gate(module):
        passed, reason = module.validate_candidate_expression(bad)
        calls.append(reason)
        assert not passed
        assert reason.startswith("R4 pre-fit rejection:")
    assert module.validate_candidate_expression is original
    assert len(calls) == 1
