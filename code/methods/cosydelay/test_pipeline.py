"""Tests of split use, exact selection, final coefficients and Test isolation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from . import pipeline as p


def physics(passed):
    return {"joint_pass": passed, "rule_scores": {k: float(passed) for k in p.MANUSCRIPT_RULE_NAMES}}


def fixture_data(tmp_path):
    data, search = tmp_path / "data", tmp_path / "search"
    data.mkdir()
    search.mkdir()
    declarations = {}
    for split in ("Train", "Validation", "Test"):
        source = p.DEFAULT_DATA / f"Intersection_1_{split}.jsonl"
        target = data / source.name
        target.write_text("\n".join(source.read_text(encoding="utf-8").splitlines()[:6]) + "\n", encoding="utf-8")
        declarations[split.lower()] = {"rows": 6, "sha256": p.digest(target), "file": target.name}
    p.write_json(data / "FROZEN_SPLIT_MANIFEST.json", {"split_seed": 20260712,
                 "intersections": {"1": {"files": declarations}}})
    p.write_json(search / "result.json", {"intersection_id": 1, "accessed_splits": ["train"],
                 "test_file_opened": False, "train_file_sha256": declarations["train"]["sha256"],
                 "candidate_evaluations": 2})
    records = [{"event": "evaluated", "candidate_id": index, "expression": "Cycle_Time * a1 / GR_phase",
                "lane_parameters": {"S_L": {"a1": value}},
                "evaluation_details": {"verifier": physics(passed)}}
               for index, value, passed in [(1, .1, False), (2, .2, True)]]
    p.write_json(search / "history.json", records)
    return data, search


class FakeRunner:
    """Inject cheap, deliberately changed refit coefficients; no LLM is used."""
    approaches = ("S",)
    refit_calls = 0

    def __init__(self, train, intersection, output, settings):
        self.output = Path(output)

    def predict(self, frame, expression, parameters):
        return {"S": frame.Delay_S.to_numpy() + parameters["S_L"]["a1"]}

    def refit(self, frame, expression, seed):
        type(self).refit_calls += 1
        assert len(frame) == 12
        assert (self.output / "selection_frozen.json").exists()
        assert not (self.output / "model.json").exists()
        return {"S_L": {"a1": 9.}}, {"rows": len(frame), "seed": seed}

    def check_physics(self, expression, parameters):
        assert parameters["S_L"]["a1"] == 9.
        return physics(False)

    def close(self):
        pass


def test_validation_uses_fitness_and_test_uses_refitted_model(tmp_path):
    data, search = fixture_data(tmp_path)
    output = tmp_path / "output"
    FakeRunner.refit_calls = 0
    result = p.complete_pipeline(search=search, data=data, output=output, intersection=1,
                                 settings=p.load_config(), version={"git_commit": "test"}, runner_factory=FakeRunner)
    selection = json.loads((output / "selection_frozen.json").read_text(encoding="utf-8"))
    model = json.loads((output / "model.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "split_access_audit.json").read_text(encoding="utf-8"))
    assert selection["candidate"]["candidate_id"] == 2  # physical pass wins, not lower RMSE alone
    assert selection["candidate"]["lane_parameters"]["S_L"]["a1"] == .2
    assert result["lane_parameters"] == model["lane_parameters"] == {"S_L": {"a1": 9.}}
    assert model["parameters_sha256"] == model["physical_parameters_sha256"]
    assert selection["validation"]["training_physical_binary"] == 1
    assert model["physical_binary"] == result["test"]["physical_binary"] == 0
    assert result["test"]["diravg"]["mae"] == pytest.approx(9.)
    assert result["test"]["fitness"] == result["test"]["fitness_accuracy"]
    assert FakeRunner.refit_calls == 1
    assert not audit["test_opened_before_freeze"]
    assert all(e["phase"] == "test" for e in audit["events"] if e["split"] == "test")
    assert p.digest(output / "model.json") == result["model_sha256"]
    with pytest.raises(FileExistsError):
        p.complete_pipeline(search=search, data=data, output=output, intersection=1, settings=p.load_config())


def test_test_access_during_refit_is_denied(tmp_path):
    data, search = fixture_data(tmp_path)

    class BadRunner(FakeRunner):
        def refit(self, *args):
            (data / "Intersection_1_Test.jsonl").read_text()
            raise AssertionError("Test must never open during refit")

    output = tmp_path / "blocked"
    with pytest.raises(PermissionError):
        p.complete_pipeline(search=search, data=data, output=output, intersection=1,
                            settings=p.load_config(), version={}, runner_factory=BadRunner)
    assert not (output / "model.json").exists()
    assert not (output / "result.json").exists()
    audit = json.loads((output / "split_access_audit.json").read_text())
    assert audit["blocked_test_access_attempts"] == 1
    assert not audit["test_opened_before_freeze"]


@pytest.mark.parametrize("defect", ["wrong_training", "missing_coefficients", "missing_candidate"])
def test_unsafe_or_incomplete_search_cannot_be_relabelled_three_way(tmp_path, defect):
    data, search = fixture_data(tmp_path)
    if defect == "wrong_training":
        path = search / "result.json"
        value = json.loads(path.read_text())
        value["train_file_sha256"] = "different_original_training_file"
    else:
        path = search / "history.json"
        value = json.loads(path.read_text())
        if defect == "missing_coefficients":
            value[0].pop("lane_parameters")
        else:
            value.pop()
    p.write_json(path, value)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        p.complete_pipeline(search=search, data=data, output=output, intersection=1,
                            settings=p.load_config(), version={}, runner_factory=FakeRunner)
    audit = json.loads((output / "split_access_audit.json").read_text())
    assert not any(e["split"] in ("validation", "test") for e in audit["events"])


def test_nonfinite_test_prediction_fails_without_fallback(tmp_path):
    data, search = fixture_data(tmp_path)

    class NonfiniteRunner(FakeRunner):
        def predict(self, frame, expression, parameters):
            if parameters["S_L"]["a1"] == 9.:
                return {"S": np.full(len(frame), np.nan)}
            return super().predict(frame, expression, parameters)

    output = tmp_path / "nonfinite"
    with pytest.raises(ValueError, match="no rows may be dropped"):
        p.complete_pipeline(search=search, data=data, output=output, intersection=1,
                            settings=p.load_config(), version={}, runner_factory=NonfiniteRunner)
    assert (output / "model.json").exists()
    assert not (output / "result.json").exists()
    assert json.loads((output / "status.json").read_text())["phase"] == "test"


def test_metrics_clip_each_direction_only_for_fitness():
    frame = pd.DataFrame({"Delay_N": [1., 2., 3.], "Delay_S": [1., 2., 3.]})
    values = p.prediction_metrics(frame, {"N": np.array([1., 2., 3.]), "S": np.array([4., 4., 4.])}, ("N", "S"))
    assert values["diravg"]["r2"] < 0
    assert values["fitness_accuracy"] == .5
    with pytest.raises(ValueError):
        p.prediction_metrics(frame, {"N": np.array([1., np.nan, 3.])}, ("N",))


def test_validation_ties_use_rmse_then_candidate_id():
    records = [{"candidate_id": i, "validation_fitness": 1.8, "metrics": {"diravg": {"rmse": error}}}
               for i, error in [(3, 2.), (2, 1.), (1, 1.)]]
    assert min(records, key=p.validation_rank)["candidate_id"] == 1


def test_partial_or_incomplete_physics_is_never_a_pass():
    partial = physics(True)
    partial["rule_scores"][p.MANUSCRIPT_RULE_NAMES[0]] = .99
    with pytest.raises(ValueError):
        p.physical_component(partial)
    assert p.physical_component(physics(False)) == 0
    incomplete = physics(True)
    incomplete["diagnostics"] = {"incomplete_resource_limited_check": True}
    with pytest.raises(ValueError):
        p.physical_component(incomplete)
    nonfinite = physics(True)
    nonfinite["rule_scores"][p.MANUSCRIPT_RULE_NAMES[0]] = float("nan")
    with pytest.raises(ValueError):
        p.physical_component(nonfinite)


def test_training_history_preserves_coefficients_for_all_candidates():
    source = (Path(__file__).parent / "support" / "population_evolution_lane.py").read_text(encoding="utf-8")
    assert '"lane_parameters": individual.get("lane_parameters")' in source


def test_stage_guard_requires_persisted_freeze(tmp_path):
    guard = p.SplitStages(tmp_path, 1)
    with pytest.raises(PermissionError):
        guard.audit("open", (str(tmp_path / "Intersection_1_Validation.jsonl"), "r"))
    guard.phase = "final_checks"
    with pytest.raises(PermissionError):
        guard.audit("open", (str(tmp_path / "Intersection_1_Test.jsonl"), "r"))
    model = tmp_path / "model.json"
    p.write_json(model, {"frozen": True})
    guard.freeze(model)
    guard.audit("open", (str(tmp_path / "Intersection_1_Test.jsonl"), "r"))


def test_initialization_only_budget_does_not_require_regeneration(monkeypatch):
    from .engine.global_selection.prompt_audit import audit_v21_prompts
    from .prompt import build_init_prompt
    features = ["flow_lane", "GR_phase", "Cycle_Time"]
    prompt = build_init_prompt({name: name for name in features}, features, 1)
    records = [{"status": "success", "prompt": prompt}]
    monkeypatch.setenv("COSYDELAY_GENERATIONS", "1")
    audit = audit_v21_prompts(records)
    assert audit["regeneration_prompt_calls"] == 0
    assert audit["regeneration_required"] is False
    monkeypatch.setenv("COSYDELAY_GENERATIONS", "2")
    with pytest.raises(RuntimeError):
        audit_v21_prompts(records)


def test_finite_but_overflowing_predictions_do_not_produce_missing_metrics():
    frame = pd.DataFrame({"Delay_N": [1., 2., 3.]})
    with pytest.raises(FloatingPointError):
        p.prediction_metrics(frame, {"N": np.full(3, 1e300)}, ("N",))


def test_credential_or_unsupported_config_fields_are_rejected(tmp_path):
    settings = p.load_config()
    settings["api_key"] = "not-a-real-key"
    config = tmp_path / "unsafe_config.json"
    p.write_json(config, settings)
    with pytest.raises(ValueError, match="credentials belong in the environment"):
        p.load_config(config)


def test_reused_search_cannot_be_reported_with_a_different_budget_or_seed():
    archive = {"search_budget": {"population": 2, "generations": 1}, "llm_seed_schedule": {"base_seed": 123}}
    settings = p.archive_settings(p.load_config(), archive, {})
    assert (settings["population"], settings["generations"], settings["seed_base"]) == (2, 1, 123)
    for overrides in ({"population": 10}, {"generations": 10}, {"seed_base": 999}):
        with pytest.raises(ValueError):
            p.archive_settings(p.load_config(), archive, overrides)
