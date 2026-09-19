from types import SimpleNamespace

from .prompt import (
    PHYSICAL_REQUIREMENTS,
    build_mutation_prompt,
    install_prompt_contract,
)
from .ranking import paper_fitness_training_tie_key
from .integration import structure_neutral_repair_guidance
from .run_training_pilot import _v13_prompt_audit


def candidate(fitness, raw_r2, rmse):
    return {
        "fitness": fitness,
        "train_rmse": 999.0,
        "evaluation_details": {
            "selection_metrics": {
                "macro_raw_r2": raw_r2,
                "pooled_rmse": rmse,
            }
        },
    }


def test_lower_fitness_can_never_win_via_tie_break():
    higher_fitness = candidate(1.1, -100.0, 1000.0)
    lower_fitness = candidate(1.0, 0.99, 0.1)
    assert paper_fitness_training_tie_key(higher_fitness) > paper_fitness_training_tie_key(lower_fitness)


def test_raw_r2_then_rmse_break_exact_fitness_ties():
    assert paper_fitness_training_tie_key(candidate(1.0, -2.0, 20.0)) > paper_fitness_training_tie_key(candidate(1.0, -3.0, 1.0))
    assert paper_fitness_training_tie_key(candidate(1.0, -2.0, 10.0)) > paper_fitness_training_tie_key(candidate(1.0, -2.0, 20.0))


def test_prompt_has_no_mandatory_global_template_or_fixed_amplitude():
    assert "y = Cycle_Time * flow_lane / GR_phase" not in PHYSICAL_REQUIREMENTS
    assert "not as an unavoidable delay amplitude" in PHYSICAL_REQUIREMENTS
    assert "positive fitted coefficients" in PHYSICAL_REQUIREMENTS


def test_scoped_prompt_installation_uses_v13_requirements_and_restores():
    calls = {}

    def old_init(*args, **kwargs):
        return "old-init"

    def old_mutation(*args, **kwargs):
        return "old-mutation"

    module = SimpleNamespace(
        build_universal_lane_init_prompt=old_init,
        build_universal_lane_mutation_prompt=old_mutation,
    )
    with install_prompt_contract(adaptation_module=module):
        prompt = module.build_universal_lane_init_prompt({}, [], intersection_id=2)
        assert "not as an unavoidable delay amplitude" in prompt
        assert "y = Cycle_Time * flow_lane / GR_phase" not in prompt
    assert module.build_universal_lane_init_prompt is old_init
    assert module.build_universal_lane_mutation_prompt is old_mutation


def test_mutation_prompt_exposes_no_training_metrics_or_feedback():
    prompt = build_mutation_prompt(
        "Cycle_Time*a1*flow_lane/GR_phase",
        "ignored",
        "ignored",
        (True, "All physical rules passed."),
        "large",
        [],
        intersection_id=2,
        search_feedback=(
            "Paper Fitness: 1.8; R2=0.8; RMSE=4; MAE=3; Weakest Training approach E"
        ),
    )
    assert "PARENT EXPRESSION:" in prompt
    assert "PARENT PHYSICAL AUDIT: PASS" in prompt
    assert "All physical rules passed." in prompt
    assert "PARENT TRAINING EVALUATION" not in prompt
    assert "Paper Fitness: 1.8" not in prompt
    assert "Weakest Training approach" not in prompt
    for metric_name in ("Fitness", "R2", "RMSE", "MAE", "Validation", "Test"):
        assert metric_name not in prompt


def test_r9_retry_feedback_is_structure_neutral():
    feedback = structure_neutral_repair_guidance("R9 zero-green limit failed")
    assert "positive infinity" in feedback
    assert "mandatory global factorization" in feedback
    assert "fixed /GR_phase factor outside" not in feedback


def test_r9_retry_feedback_does_not_recurse_when_installed(monkeypatch):
    from methods.cosydelay_v9_clean_from_scratch import integration as clean

    monkeypatch.setattr(clean, "_compact_physical_repair_guidance", structure_neutral_repair_guidance)
    feedback = clean._compact_physical_repair_guidance("R9 failed")
    assert "positive infinity" in feedback


def test_audit_allows_parent_formula_without_treating_it_as_prompt_bias():
    prompt = build_mutation_prompt(
        "Cycle_Time * flow_lane / GR_phase * a1",
        "",
        "",
        (True, "All physical rules passed."),
        "large",
        [],
    )
    audit = _v13_prompt_audit([{"status": "success", "prompt": prompt}])
    assert audit["fixed_global_template_absent"]
    assert audit["mutation_training_feedback_absent"]
