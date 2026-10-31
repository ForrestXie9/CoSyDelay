"""Validation selection, coefficient-only refit, and frozen Test evaluation."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, parse_dialogue_flexible, preprocess_data_flexible
from expression_rules import canonical_expression
from optimization_lane import calculate_approach_delays_from_universal, prepare_optimization_context
from .engine.data_protocol.run_formal_training_search import lanes_for
from .engine.fit_guard.fit_timeout import GuardedCandidateFitter
from .engine.training_protocol.physics import MANUSCRIPT_RULE_NAMES, ManuscriptVerifierConfig
from .fitter import accelerated_fit_supervisor_main
from .physics import GuardedSymbolicChecks, SymbolicCheckError


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPOSITORY / "configs" / "cosydelay.json"
DEFAULT_DATA = REPOSITORY / "data" / "locked_splits"
INVALID_APPROACHES = {(4, "E"), (6, "W")}
DATASET_NAME = re.compile(r"^Intersection_\d+_(Train|Validation|Test)\.jsonl$", re.I)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(json_ready(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(12):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(.03 * (attempt + 1))


def load_config(path=DEFAULT_CONFIG):
    settings = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    allowed = {"protocol_id", "population", "generations", "seed_base", "validation_metric",
               "validation_tie_breakers", "candidate_pool", "refit_split", "refit_restarts",
               "refit_maxiter", "refit_maxfun", "refit_timeout_seconds", "final_physics_timeout_seconds",
               "physical_failure_policy", "prediction_failure_policy"}
    if set(settings) != allowed:
        raise ValueError("config contains missing or unsupported fields; credentials belong in the environment")
    if settings.get("protocol_id") != "cosydelay_train_validation_refit_v1":
        raise ValueError("unsupported CoSyDelay protocol")
    if settings.get("validation_metric") != "mean_nonnegative_approach_r2_plus_training_binary_physics":
        raise ValueError("validation metric must match the released Fitness definition")
    if settings.get("validation_tie_breakers") != ["lower_direction_average_rmse", "lower_candidate_id"]:
        raise ValueError("unsupported validation tie-breaking policy")
    if settings.get("candidate_pool") != "all_completed_training_candidates":
        raise ValueError("all archived Training candidates must be considered")
    if settings.get("refit_split") != "Training+Validation":
        raise ValueError("final coefficient refit must use Training+Validation")
    if (settings.get("refit_restarts"), settings.get("refit_maxiter"), settings.get("refit_maxfun")) != (10, 200, 20000):
        raise ValueError("search and final refit retain the same ten-start optimizer settings")
    if settings.get("physical_failure_policy") != "retain_refitted_model_with_physical_component_zero_no_test_reselection":
        raise ValueError("unsupported physical failure policy")
    if settings.get("prediction_failure_policy") != "fail_run_without_fallback_or_row_dropping":
        raise ValueError("unsupported prediction failure policy")
    for key in ("population", "generations"):
        if not isinstance(settings.get(key), int) or isinstance(settings[key], bool) or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("refit_timeout_seconds", "final_physics_timeout_seconds"):
        if not isinstance(settings.get(key), (int, float)) or not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not isinstance(settings.get("seed_base"), int) or isinstance(settings["seed_base"], bool):
        raise ValueError("seed_base must be an integer")
    return settings


def source_version():
    def git(*arguments):
        try:
            return subprocess.check_output(["git", *arguments], cwd=REPOSITORY,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    paths = sorted((REPOSITORY / "code" / "methods" / "cosydelay").rglob("*.py"))
    # Never hash credential files or include environment variables in provenance.
    paths.append(DEFAULT_CONFIG)
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_worktree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "source_sha256": {p.relative_to(REPOSITORY).as_posix(): digest(p) for p in paths}}


def archive_settings(settings, archive_result, overrides):
    """A reused search must not acquire a fictional new budget or seed."""
    settings = dict(settings)
    budget = archive_result.get("search_budget", {})
    for key in ("population", "generations"):
        observed = budget.get(key)
        if not isinstance(observed, int) or observed < 1:
            raise ValueError("reused archive lacks an explicit public search budget")
        if overrides.get(key) is not None and overrides[key] != observed:
            raise ValueError(f"--{key} differs from the completed search archive")
        settings[key] = observed
    seed = archive_result.get("llm_seed_schedule", {}).get("base_seed")
    if not isinstance(seed, int):
        raise ValueError("reused archive lacks an explicit search seed")
    if overrides.get("seed_base") is not None and overrides["seed_base"] != seed:
        raise ValueError("--seed-base differs from the completed search archive")
    settings["seed_base"] = seed
    return settings


class SplitStages:
    """Deny Test access until a persisted final model has been frozen."""
    def __init__(self, data, intersection):
        self.data = Path(data).resolve()
        self.intersection = intersection
        self.phase = "training"
        self.active = True
        self.events = []
        self.frozen_model_sha256 = None

    def audit(self, event, arguments):
        if not self.active or event != "open" or not arguments:
            return
        if not isinstance(arguments[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(arguments[0]))
        match = DATASET_NAME.match(path.name)
        if not match:
            return
        split = match.group(1).lower()
        permitted = {"train"}
        if self.phase in ("validation_selection", "refit", "final_checks", "test"):
            permitted.add("validation")
        if self.phase == "test" and self.frozen_model_sha256 is not None:
            permitted.add("test")
        expected = self.data / f"Intersection_{self.intersection}_{split.title()}.jsonl"
        allowed = split in permitted and path.resolve() == expected
        self.events.append({"split": split, "phase": self.phase, "allowed": allowed,
                            "path": str(path.resolve()), "utc": utc_now()})
        if not allowed:
            raise PermissionError(f"{split} access denied in pipeline phase {self.phase}")

    def freeze(self, model_path):
        self.frozen_model_sha256 = digest(model_path)
        self.phase = "test"


def prediction_metrics(frame, predictions, approaches):
    by_approach = {}
    for approach in approaches:
        truth = frame[f"Delay_{approach}"].to_numpy(dtype=float)
        estimate = np.asarray(predictions[approach], dtype=float)
        if truth.shape != estimate.shape or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(estimate)):
            raise ValueError(f"invalid {approach} predictions; no rows may be dropped")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            denominator = float(np.sum((truth - truth.mean()) ** 2))
            if len(truth) < 2 or not math.isfinite(denominator) or denominator <= 0:
                raise ValueError(f"nonpositive target variance for declared valid approach {approach}")
            error = truth - estimate
            by_approach[approach] = {"r2": float(1 - np.sum(error**2) / denominator),
                                     "rmse": float(np.sqrt(np.mean(error**2))),
                                     "mae": float(np.mean(np.abs(error))), "rows": len(truth)}
        if any(not math.isfinite(by_approach[approach][key]) for key in ("r2", "rmse", "mae")):
            raise ValueError(f"nonfinite {approach} metric; no rows may be dropped")
    if not by_approach:
        raise ValueError("no valid approaches")
    return {"diravg": {key: float(np.mean([v[key] for v in by_approach.values()]))
                       for key in ("r2", "rmse", "mae")},
            "fitness_accuracy": float(np.mean([max(0., v["r2"]) for v in by_approach.values()])),
            "by_approach": by_approach}


def physical_component(physical):
    rules = physical.get("rule_scores", {})
    if set(rules) != set(MANUSCRIPT_RULE_NAMES) or any(
            not math.isfinite(float(rules[k])) or not 0 <= float(rules[k]) <= 1 for k in MANUSCRIPT_RULE_NAMES):
        raise ValueError("missing or invalid seven-rule physical audit")
    if not physical.get("joint_pass", False):
        return 0
    if any(float(rules[k]) < 1 - 1e-12 for k in MANUSCRIPT_RULE_NAMES):
        raise ValueError("a physical pass requires a complete seven-rule audit")
    if physical.get("diagnostics", {}).get("incomplete_resource_limited_check"):
        raise ValueError("an incomplete check cannot claim physical compliance")
    return 1


def validation_rank(record):
    return (-record["validation_fitness"], record["metrics"]["diravg"]["rmse"], record["candidate_id"])


def read_split(data, intersection, split, manifest):
    path = Path(data) / f"Intersection_{intersection}_{split}.jsonl"
    declaration = manifest["intersections"][str(intersection)]["files"][split.lower()]
    if digest(path) != declaration["sha256"]:
        raise ValueError(f"{split} data hash differs from the frozen manifest")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(records) != declaration["rows"]:
        raise ValueError(f"{split} row count differs from the frozen manifest")
    signatures = [object_digest(parse_dialogue_flexible(row["dialogue"], intersection)) for row in records]
    if len(set(signatures)) != len(signatures):
        raise ValueError(f"duplicate numeric input configurations in {split}")
    frame = preprocess_data_flexible(load_dataset_flexible(str(path), intersection), intersection).reset_index(drop=True)
    return frame, set(signatures)


def archived_candidates(search, expected_train_sha, intersection):
    search = Path(search)
    result = json.loads((search / "result.json").read_text(encoding="utf-8"))
    if result.get("train_file_sha256") != expected_train_sha:
        raise ValueError("search used a different Training file; retrospective re-fitting cannot fix this")
    if result.get("accessed_splits") != ["train"] or result.get("test_file_opened") is not False:
        raise ValueError("search archive does not establish Training-only access")
    if result.get("intersection_id") != intersection:
        raise ValueError("search archive intersection mismatch")
    history = json.loads((search / "history.json").read_text(encoding="utf-8"))
    candidates = [item for item in history if item.get("event") == "evaluated"]
    ids = [item["candidate_id"] for item in candidates]
    if not candidates or len(ids) != len(set(ids)) or len(candidates) != result.get("candidate_evaluations"):
        raise ValueError("Training candidate archive is incomplete or contains duplicate candidate IDs")
    for candidate in candidates:
        if not candidate.get("lane_parameters"):
            raise ValueError("missing Training coefficients; do not silently refit candidates before Validation selection")
        physical = candidate.get("evaluation_details", {}).get("verifier")
        if physical is None:
            raise ValueError("missing Training physical audit")
        physical_component(physical)
    return candidates


class NumericalRunner:
    def __init__(self, train, intersection, output, settings):
        self.intersection, self.settings = intersection, settings
        config = INTERSECTION_CONFIGS[intersection]
        self.all_approaches = tuple(config["approaches"])
        self.approaches = tuple(a for a in self.all_approaches if (intersection, a) not in INVALID_APPROACHES)
        self.lanes, self.mapping = lanes_for(config)
        self.symbolic = GuardedSymbolicChecks(Path(output) / "checks")

    def predict(self, frame, expression, parameters):
        return calculate_approach_delays_from_universal(
            df=frame, universal_expr=expression, lane_parameters=parameters,
            lanes=self.lanes, lane_to_approach=self.mapping,
            intersection_id=self.intersection, strict=True)

    def refit(self, frame, expression, seed):
        context = prepare_optimization_context(frame, self.lanes, self.mapping, self.intersection)
        static = {"df": frame, "lanes": self.lanes, "lane_to_approach": self.mapping,
                  "intersection_id": self.intersection, "prepared_context": context,
                  "approach_targets": {a: frame[f"Delay_{a}"] for a in self.all_approaches}}
        fitter = GuardedCandidateFitter(
            static_fit_kwargs=static, parallel_workers=4,
            maxiter=self.settings["refit_maxiter"], maxfun=self.settings["refit_maxfun"],
            fit_timeout_seconds=self.settings["refit_timeout_seconds"],
            startup_timeout_seconds=30, close_timeout_seconds=5,
            supervisor_target=accelerated_fit_supervisor_main)
        diagnostics = {}
        try:
            parameters = fitter.fit(universal_expr=expression, warm_parameters=None,
                                    rng=np.random.default_rng(seed), n_restarts=self.settings["refit_restarts"],
                                    diagnostics=diagnostics, **static)
            if not parameters or any(not math.isfinite(float(v)) or float(v) <= 0
                                     for lane in parameters.values() for v in lane.values()):
                raise ValueError("invalid final coefficients")
            diagnostics["native_restart_selection_label"] = diagnostics.get("restart_selection")
            diagnostics["restart_selection"] = "minimum_training_plus_validation_mse"
            diagnostics["coefficient_fit_split"] = "Training+Validation"
            return parameters, diagnostics
        finally:
            fitter.close()

    def check_physics(self, expression, parameters):
        timeout = self.settings["final_physics_timeout_seconds"]
        try:
            return self.symbolic.call("physical", expression, time.monotonic() + timeout + 20,
                                      timeout=timeout, parameters=parameters, lanes=self.lanes,
                                      features=["flow_lane", "GR_phase", "Cycle_Time"])
        except (SymbolicCheckError, ValueError) as exc:
            return {"joint_pass": False, "score": 0., "rule_order": list(MANUSCRIPT_RULE_NAMES),
                    "rule_scores": {k: 0. for k in MANUSCRIPT_RULE_NAMES},
                    "diagnostics": {"incomplete_resource_limited_check": True, "reason": str(exc)},
                    "config": asdict(ManuscriptVerifierConfig())}

    def close(self):
        self.symbolic.close()


def complete_pipeline(*, search, data, output, intersection, settings, version=None, runner_factory=NumericalRunner):
    """Select using Validation, refit once, and evaluate only the frozen model."""
    output, data, search = Path(output).resolve(), Path(data).resolve(), Path(search).resolve()
    if (output / "model.json").exists() or (output / "result.json").exists():
        raise FileExistsError("refusing to overwrite an already frozen model")
    manifest = json.loads((data / "FROZEN_SPLIT_MANIFEST.json").read_text(encoding="utf-8-sig"))
    declarations = manifest["intersections"][str(intersection)]["files"]
    version = source_version() if version is None else version
    guard = SplitStages(data, intersection)
    sys.addaudithook(guard.audit)
    runner = None
    started = time.monotonic()
    try:
        train, train_inputs = read_split(data, intersection, "Train", manifest)
        candidates = archived_candidates(search, declarations["train"]["sha256"], intersection)
        write_json(output / "search_frozen.json", {"utc": utc_now(), "phase": "training_search_complete",
                   "validation_or_test_opened": False, "candidate_count": len(candidates),
                   "training_history_sha256": digest(search / "history.json"),
                   "training_result_sha256": digest(search / "result.json")})
        guard.phase = "validation_selection"
        validation, validation_inputs = read_split(data, intersection, "Validation", manifest)
        if train_inputs & validation_inputs:
            raise ValueError("Training and Validation input configurations overlap")
        runner = runner_factory(train, intersection, output, settings)
        selection_started = time.monotonic()
        records = []
        viable = []
        for candidate in candidates:
            physical = candidate["evaluation_details"]["verifier"]
            record = {"candidate_id": candidate["candidate_id"], "expression": candidate["expression"],
                      "training_parameters_sha256": object_digest(candidate["lane_parameters"]),
                      "training_physical_binary": physical_component(physical)}
            try:
                prediction = runner.predict(validation, candidate["expression"], candidate["lane_parameters"])
                values = prediction_metrics(validation, prediction, runner.approaches)
                record.update(status="valid", metrics=values,
                              validation_fitness=values["fitness_accuracy"] + record["training_physical_binary"])
                viable.append((record, candidate))
            except (ValueError, ArithmeticError, RuntimeError) as exc:
                record.update(status="invalid_validation_prediction", reason=f"{type(exc).__name__}: {exc}")
            records.append(record)
        write_json(output / "validation_candidates.json", records)
        if not viable:
            raise RuntimeError("no candidate has complete finite Validation predictions")
        selected, candidate = min(viable, key=lambda pair: validation_rank(pair[0]))
        selection = {"utc": utc_now(), "selected_by": settings["validation_metric"],
                     "tie_breakers": settings["validation_tie_breakers"], "candidate_pool_size": len(candidates),
                     "candidate": candidate, "validation": selected, "test_opened": False,
                     "configuration": settings, "configuration_sha256": object_digest(settings)}
        write_json(output / "selection_frozen.json", selection)
        selection_wall = time.monotonic() - selection_started
        guard.phase = "refit"
        combined = pd.concat([train, validation], ignore_index=True)
        expression = candidate["expression"]
        refit_seed = int.from_bytes(hashlib.sha256(
            f"refit|{settings['seed_base']}|{canonical_expression(expression)}".encode()).digest()[:8], "little")
        refit_started = time.monotonic()
        parameters, diagnostics = runner.refit(combined, expression, refit_seed)
        refit_wall = time.monotonic() - refit_started
        guard.phase = "final_checks"
        physics_started = time.monotonic()
        final_physical = runner.check_physics(expression, parameters)
        final_binary = physical_component(final_physical)
        physics_wall = time.monotonic() - physics_started
        write_json(output / "final_physics.json", final_physical)
        model = {"method": "CoSyDelay", "protocol_id": settings["protocol_id"], "intersection_id": intersection,
                 "frozen_utc": utc_now(), "expression": expression, "lane_parameters": parameters,
                 "parameters_sha256": object_digest(parameters), "physical_binary": final_binary,
                 "physical": final_physical, "physical_parameters_sha256": object_digest(parameters),
                 "refit_split": "Training+Validation", "refit_rows": len(combined), "refit_seed": refit_seed,
                 "refit_diagnostics": diagnostics, "new_structure_search_during_refit": False,
                 "selection_frozen_sha256": digest(output / "selection_frozen.json"),
                 "training_search_frozen_sha256": digest(output / "search_frozen.json"),
                 "configuration": settings, "configuration_sha256": object_digest(settings),
                 "source_version": version, "split_seed": manifest["split_seed"],
                 "split_manifest_sha256": digest(data / "FROZEN_SPLIT_MANIFEST.json"),
                 "split_files": declarations, "test_opened": False}
        write_json(output / "model.json", model)
        guard.freeze(output / "model.json")
        write_json(output / "freeze.json", {"utc": utc_now(), "model_sha256": guard.frozen_model_sha256,
                   "final_physics_sha256": digest(output / "final_physics.json"), "test_opened": False})
        # Reload the persisted model. No Training coefficients or stale physical
        # scores are used for Test, and no Test-driven fallback is permitted.
        model = json.loads((output / "model.json").read_text(encoding="utf-8"))
        if object_digest(model["lane_parameters"]) != model["physical_parameters_sha256"]:
            raise ValueError("frozen model and final physics use different coefficients")
        test_started = time.monotonic()
        test, test_inputs = read_split(data, intersection, "Test", manifest)
        if test_inputs & (train_inputs | validation_inputs):
            raise ValueError("Test input configurations overlap Training/Validation")
        prediction = runner.predict(test, model["expression"], model["lane_parameters"])
        test_metrics = prediction_metrics(test, prediction, runner.approaches)
        test_metrics["physical_binary"] = model["physical_binary"]
        test_metrics["fitness"] = test_metrics["fitness_accuracy"] + model["physical_binary"]
        if digest(output / "model.json") != guard.frozen_model_sha256:
            raise ValueError("frozen model changed during Test evaluation")
        rows = [{"approach": a, "row": i, "truth": float(y), "prediction": float(p)}
                for a in runner.approaches for i, (y, p) in enumerate(zip(test[f"Delay_{a}"], prediction[a]))]
        pd.DataFrame(rows).to_csv(output / "test_predictions.csv", index=False)
        result = {"status": "complete", "method": "CoSyDelay", "protocol_id": settings["protocol_id"],
                  "intersection_id": intersection, "expression": model["expression"],
                  "lane_parameters": model["lane_parameters"], "physical_binary": model["physical_binary"],
                  "physical": model["physical"], "validation_selection": selected, "test": test_metrics,
                  "fitness": test_metrics["fitness"], "source_version": version,
                  "model_sha256": guard.frozen_model_sha256,
                  "test_used_for_fit_or_selection": False, "structure_search_during_refit": False,
                  "timing": {"validation_selection_seconds": selection_wall, "refit_seconds": refit_wall,
                             "final_physics_seconds": physics_wall,
                             "test_evaluation_seconds": time.monotonic() - test_started,
                             "post_search_seconds": time.monotonic() - started}, "finished_utc": utc_now()}
        write_json(output / "result.json", result)
        write_json(output / "status.json", {"status": "complete", "test_diravg": test_metrics["diravg"],
                   "test_fitness": test_metrics["fitness"], "final_physical_binary": final_binary, "utc": utc_now()})
        return result
    except Exception as exc:
        write_json(output / "status.json", {"status": "failed", "phase": guard.phase,
                   "reason": f"{type(exc).__name__}: {exc}", "utc": utc_now()})
        raise
    finally:
        write_json(output / "split_access_audit.json", {"events": guard.events,
                   "frozen_model_sha256": guard.frozen_model_sha256,
                   "test_opened_before_freeze": any(e["split"] == "test" and e["allowed"] and e["phase"] != "test" for e in guard.events),
                   "blocked_test_access_attempts": sum(e["split"] == "test" and not e["allowed"] for e in guard.events)})
        guard.active = False
        if runner is not None:
            runner.close()
