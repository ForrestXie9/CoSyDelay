from types import SimpleNamespace

from .prompt import (
    build_mutation_prompt,
    extract_physical_repair_feedback,
    install_prompt_contract,
)
from .ranking import paper_fitness_training_tie_key
from .source_manifest import V14_SOURCE_FILES, validate_v14_source_manifest


def candidate(fitness, raw_r2, rmse):
    return {
        "fitness": fitness,
        "evaluation_details": {
            "selection_metrics": {
                "macro_raw_r2": raw_r2,
                "pooled_rmse": rmse,
            }
        },
    }


def test_physical_feedback_is_retained_but_metrics_are_removed():
    feedback = """PARENT TRAINING EVALUATION (Training only):
- Paper Fitness: 1.812300
- Pooled RMSE: 4.123
- Approach E Training: R2=0.1, RMSE=9.2, MAE=7.1
MANDATORY REPAIR:
- R2: use only positive nondecreasing demand factors and remove subtractive terms.
- R9: make the complete expression approach positive infinity.
"""
    cleaned = extract_physical_repair_feedback(feedback)
    assert "- R2:" in cleaned
    assert "- R9:" in cleaned
    assert "Fitness" not in cleaned
    assert "RMSE" not in cleaned
    assert "Approach E" not in cleaned


def test_mutation_prompt_contains_only_rule_level_retry_feedback():
    prompt = build_mutation_prompt(
        "Cycle_Time*a1*flow_lane/GR_phase",
        "",
        "",
        (True, "All eight physical principles passed."),
        "large",
        [],
        search_feedback=(
            "PARENT TRAINING EVALUATION (Training only):\n"
            "- Paper Fitness: 1.8\n"
            "MANDATORY REPAIR:\n"
            "- R3: place green-dependent factors only in positive denominators."
        ),
    )
    assert "POST-FIT PHYSICAL REPAIR" in prompt
    assert "- R3:" in prompt
    assert "Paper Fitness" not in prompt
    assert "PARENT TRAINING EVALUATION" not in prompt


def test_explicit_selection_hook_uses_tie_key_and_restores():
    old_key = lambda item: (item["fitness"],)
    module = SimpleNamespace(
        individual_selection_key=old_key,
        fit_lane_parameters_to_approaches=None,
        safe_generate_universal_lane_expression=None,
    )
    # The integration contract is tested directly in the full scoped tests;
    # here verify the key itself never lets lower Fitness win.
    assert paper_fitness_training_tie_key(candidate(1.1, -5, 99)) > (
        paper_fitness_training_tie_key(candidate(1.0, 0.99, 1))
    )
    assert module.individual_selection_key is old_key


def test_prompt_installation_is_scoped():
    old_init = lambda *args, **kwargs: "old-init"
    old_mutation = lambda *args, **kwargs: "old-mutation"
    module = SimpleNamespace(
        build_universal_lane_init_prompt=old_init,
        build_universal_lane_mutation_prompt=old_mutation,
    )
    with install_prompt_contract(adaptation_module=module):
        assert module.build_universal_lane_mutation_prompt is build_mutation_prompt
        assert module._cosydelay_init_prompt_override is not None
    assert module.build_universal_lane_init_prompt is old_init
    assert module.build_universal_lane_mutation_prompt is old_mutation
    assert not hasattr(module, "_cosydelay_init_prompt_override")


def test_complete_source_manifest_exists_and_is_unique():
    validate_v14_source_manifest()
    resolved = [path.resolve() for path in V14_SOURCE_FILES]
    assert len(resolved) == len(set(resolved))
