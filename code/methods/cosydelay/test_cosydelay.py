from __future__ import annotations

from pathlib import Path

import pytest

from .regeneration import REGENERATION_ATTEMPTS, with_uniform_regeneration


HERE = Path(__file__).resolve().parent


def test_regeneration_uses_one_five_attempt_policy() -> None:
    seen = []

    def source(*args, **kwargs):
        seen.append((args, kwargs))
        return "ok"

    wrapped = with_uniform_regeneration(source)
    assert wrapped("x", "y", "regeneration", None, None, None, None, None, 3) == "ok"
    assert seen[0][0][8] == REGENERATION_ATTEMPTS
    assert seen[0][1] == {}


def test_initialization_is_not_rewritten() -> None:
    seen = []

    def source(*args, **kwargs):
        seen.append((args, kwargs))
        return "ok"

    wrapped = with_uniform_regeneration(source)
    assert wrapped("x", "y", "initial", None, None, None, None, None, 2) == "ok"
    assert seen[0][0][8] == 2


def test_keyword_regeneration_also_uses_the_same_budget() -> None:
    seen = []

    def source(*args, **kwargs):
        seen.append((args, kwargs))
        return "ok"

    wrapped = with_uniform_regeneration(source)
    wrapped("features", mutation_type="regeneration", max_retries=99)
    assert seen[0][1]["max_retries"] == REGENERATION_ATTEMPTS == 5


def test_uniform_wrapper_has_no_second_prompt_or_fallback_mode() -> None:
    source = (HERE / "regeneration.py").read_text(encoding="utf-8").lower()
    assert "with_initialization_fallback" not in source
    assert "initialization_prompt" not in source


def test_cosydelay_runner_embeds_uniform_policy() -> None:
    from .run_p10g10_100 import _source

    source = _source()
    assert "cosydelay.regeneration" in source
    assert "cosydelay.fitter" in source
    compile(source, "<cosydelay-runner-test>", "exec")


def test_expanded_range_profile_is_declared_without_optimizer_import() -> None:
    from .range_policy import (
        COSYDELAY_RANGE_PROFILE,
        COSYDELAY_RANGE_PROFILE_ID,
        COSYDELAY_RANGE_SELECTION_MANIFEST,
    )

    assert COSYDELAY_RANGE_PROFILE_ID == "expanded_nonlinear_training_screen_v1"
    assert COSYDELAY_RANGE_PROFILE["scale"] == (0.001, 1000.0)
    assert COSYDELAY_RANGE_PROFILE["power_exponent"] == (0.01, 8.0)
    assert COSYDELAY_RANGE_PROFILE["exp_coefficient"] == (0.00001, 2.0)
    assert COSYDELAY_RANGE_SELECTION_MANIFEST["r2_wins"] == COSYDELAY_RANGE_SELECTION_MANIFEST["pairs"] == 6
    assert COSYDELAY_RANGE_SELECTION_MANIFEST["test_used"] is False


def test_regeneration_prompt_requests_structural_exploration() -> None:
    from .prompt import build_regeneration_prompt

    value = build_regeneration_prompt(
        "a1 * Cycle_Time / GR_phase", "", "", (False, "duplicate"),
        "large", [], 1,
    )
    assert "structurally distinct candidate" in value
    assert "not a minimal modification" in value


def test_regeneration_does_not_fallback_to_initialization() -> None:
    source = (HERE / "regeneration.py").read_text(encoding="utf-8")
    assert "initialization prompt" not in source.lower()


def test_cosydelay_runner_installs_structural_family_novelty_overlay() -> None:
    from .run_p10g10_100 import _source

    source = _source()
    assert "structural_diversity_population_v1 as pairwise_population" in source
    assert "cosydelay_v21_global_selection import global_population as pairwise_population" not in source


def test_cosydelay_embedded_population_overlay_runtime() -> None:
    import inspect

    from .run_p10g10_100 import _source

    namespace = {"__name__": "cosydelay_overlay_probe"}
    exec(compile(_source(), "<cosydelay-overlay-probe>", "exec"), namespace)
    population = namespace["pairwise_population"]
    assert population.STRUCTURAL_DIVERSITY_MODE == "family_unique"
    assert inspect.signature(
        population.evolve_universal_lane_expression
    ).parameters["structural_diversity_mode"].default == "family_unique"


def test_full_history_is_checked_without_prompt_history() -> None:
    from methods.structural_diversity_population_v1 import (
        wrap_without_history_prompt,
    )

    observed = []

    def source(*args, **kwargs):
        observed.append(kwargs)
        return "ok"

    wrapped = wrap_without_history_prompt(source)
    assert wrapped(excluded_expressions=["old-1", "old-2"]) == "ok"
    assert observed[0]["excluded_expressions"] == ["old-1", "old-2"]
    assert observed[0]["max_prompt_exclusions"] == 0


def test_history_is_checked_program_side_but_not_written_into_prompt() -> None:
    import expression_adaptation_lane as lane
    from methods.structural_diversity_population_v1 import (
        wrap_without_history_prompt,
    )

    captured = []
    original_run_llm = lane.run_llm
    try:
        lane.run_llm = lambda prompt: (
            captured.append(prompt)
            or "### Expression\ny = Cycle_Time * (a1 + a2 * flow_lane / (GR_phase + 1))\n"
            "### Explanation\nA compact traffic-physical structure."
        )
        wrapped = wrap_without_history_prompt(
            lane.safe_generate_universal_lane_expression
        )
        expression, _, _ = wrapped(
            {},
            [],
            mutation_type="initial",
            excluded_expressions=[
                "Cycle_Time * (a1 + a2 * flow_lane / (GR_phase + 1)) + a3"
            ],
            enforce_physical_prefilter=False,
        )
    finally:
        lane.run_llm = original_run_llm
    assert "GR_phase + 1" in expression
    assert captured
    assert "GR_phase / (GR_phase + 1)" not in captured[0]
    assert "Cycle_Time * (a1 + a2 * flow_lane / (GR_phase + 1)) + a3" not in captured[0]
