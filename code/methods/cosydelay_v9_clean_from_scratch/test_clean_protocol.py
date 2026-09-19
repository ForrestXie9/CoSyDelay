from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import expression_adaptation_lane as adaptation_pipeline
from expression_rules import parse_symbolic_expression
from methods.cosydelay_lbfgsb_r10_parallel4.prefit_gate import (
    evaluate_structural_prefit_gate,
)
from methods.cosydelay_lbfgsb_r10_parallel4_v3.physics_audit import (
    audit_fitted_physics,
)
from methods.cosydelay_v9_clean_from_scratch import clean_fitter
from methods.cosydelay_v9_clean_from_scratch import integration
from methods.cosydelay_v9_clean_from_scratch import launch_formal_i1_i6 as launcher
from methods.cosydelay_v9_clean_from_scratch import run_formal_training_search as runner
from methods.cosydelay_v9_clean_from_scratch.diagnostics import (
    audit_parameter_quality,
    summarize_compute_efficiency,
)
from methods.cosydelay_v9_clean_from_scratch.policy import CLEAN_POLICY
from methods.cosydelay_v9_clean_from_scratch.metrics import score_predictions
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import (
    SplitAccessGuard,
    accessed_dataset_splits,
)
from methods.cosydelay_v9_clean_from_scratch.source_manifest import (
    SOURCE_FILES,
    validate_source_manifest,
)


class FakeFitter:
    instances = []

    def __init__(self, *, parallel_workers, maxiter=200, maxfun=20000):
        self.parallel_workers = parallel_workers
        self.maxiter = maxiter
        self.maxfun = maxfun
        self.parents = []
        self.parameters = {}
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def register_parent(self, expression, parent):
        self.parents.append((expression, parent))

    def remember_parameters(self, expression, parameters):
        self.parameters[str(expression)] = parameters


@contextmanager
def fake_prefit_gate(*, population_module, adaptation_module):
    yield []


def fake_modules():
    calls = []

    def generator(*args, **kwargs):
        calls.append((args, dict(kwargs)))
        if kwargs.get("mutation_type") == "initial":
            return "Cycle_Time*a1*flow_lane/GR_phase", "", "initial"
        return "Cycle_Time*a1*flow_lane/(GR_phase**a2)", "", "offspring"

    def evaluator(*args, **kwargs):
        raise AssertionError("base evaluator should not be reached in this test")

    population = SimpleNamespace(
        fit_lane_parameters_to_approaches=object(),
        evaluate_expression_with_fitting=evaluator,
        safe_generate_universal_lane_expression=generator,
        MAX_INITIALIZATION_BATCHES_PER_SLOT=10,
        MAX_OFFSPRING_BATCHES_PER_SLOT=10,
    )
    adaptation = SimpleNamespace(
        PHYSICAL_REQUIREMENTS_STANDARD="standard",
        PHYSICAL_REQUIREMENTS_COMPACT="compact",
        build_universal_lane_init_prompt=lambda *args, **kwargs: "prompt",
        validate_candidate_expression=lambda expression: (True, "base pass"),
    )
    return population, adaptation, calls


class CleanProtocolTests(unittest.TestCase):
    def setUp(self):
        FakeFitter.instances.clear()
        self.frame = pd.DataFrame({"x": [1.0]})
        self.targets = {"E": pd.Series([1.0])}

    def install(self, population, adaptation):
        return integration.install_clean_single_evolution(
            df_train=self.frame,
            targets=self.targets,
            lanes=["E_T"],
            lane_to_approach={"E_T": "E"},
            intersection_id=1,
            population_module=population,
            adaptation_module=adaptation,
        )

    def test_incumbent_is_absent_from_interface_and_policy(self):
        signature = inspect.signature(integration.install_clean_single_evolution)
        self.assertNotIn("incumbent", signature.parameters)
        self.assertFalse(CLEAN_POLICY.external_incumbent_allowed)
        self.assertEqual(CLEAN_POLICY.population, 10)
        self.assertEqual(CLEAN_POLICY.generations, 10)
        self.assertEqual(CLEAN_POLICY.optimizer_restarts, 10)
        self.assertEqual(CLEAN_POLICY.optimizer_maxiter, 200)
        self.assertEqual(CLEAN_POLICY.optimizer_maxfun, 20_000)
        self.assertEqual(CLEAN_POLICY.approach_workers_cap, 4)
        self.assertEqual(CLEAN_POLICY.physical_consistency_bonus, 1.0)
        self.assertEqual(CLEAN_POLICY.fitness_lower_bound, 0.0)
        self.assertEqual(CLEAN_POLICY.fitness_upper_bound, 2.0)
        self.assertEqual(
            CLEAN_POLICY.prompt_contract_version,
            "v9_global_flow_inverse_green_v3",
        )

    def test_formal_cli_does_not_expose_method_or_seed_overrides(self):
        with patch.object(
            sys,
            "argv",
            ["runner", "--intersection", "1", "--output", "new-run"],
        ):
            args = runner.arguments()
        for prohibited in (
            "seed",
            "population",
            "generations",
            "optimizer_restarts",
            "incumbent_result",
            "validation",
            "test",
        ):
            self.assertFalse(hasattr(args, prohibited), prohibited)

    def test_method_config_matches_frozen_policy(self):
        config_path = Path(__file__).resolve().parent / "method_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["method_id"], CLEAN_POLICY.method_id)
        self.assertEqual(config["search"]["population"], CLEAN_POLICY.population)
        self.assertEqual(config["search"]["generations"], CLEAN_POLICY.generations)
        self.assertEqual(
            config["search"]["optimizer_restarts"],
            CLEAN_POLICY.optimizer_restarts,
        )
        self.assertEqual(
            config["search"]["optimizer_maxiter"], CLEAN_POLICY.optimizer_maxiter
        )
        self.assertEqual(
            config["search"]["optimizer_maxfun"], CLEAN_POLICY.optimizer_maxfun
        )
        self.assertEqual(
            config["search"]["approach_workers_cap"],
            CLEAN_POLICY.approach_workers_cap,
        )
        self.assertEqual(
            config["llm"]["model_requested"],
            CLEAN_POLICY.llm_model_requested,
        )
        self.assertEqual(
            config["llm"]["temperature"], CLEAN_POLICY.llm_temperature
        )
        self.assertIsNone(config["llm"]["top_p"])
        self.assertIsNone(config["llm"]["max_tokens"])
        self.assertEqual(
            config["llm"]["sampling_defaults"],
            "original provider defaults; temperature, top_p, and max_tokens are not sent",
        )
        self.assertEqual(
            config["coefficient_bounds"]["profile"].split(";")[0],
            CLEAN_POLICY.coefficient_bounds_profile,
        )
        for role, bounds in CLEAN_POLICY.coefficient_bounds.items():
            self.assertEqual(config["coefficient_bounds"][role], list(bounds))
        self.assertEqual(config["selection"]["fitness_source"], "paper implementation")
        self.assertEqual(
            config["selection"][
                "binary_physical_consistency_for_admitted_candidates"
            ],
            CLEAN_POLICY.physical_consistency_bonus,
        )

    def test_source_manifest_covers_optimizer_and_physics_dependencies(self):
        validate_source_manifest()
        relative = {
            str(path.relative_to(runner.GMINI)).replace("\\", "/")
            for path in SOURCE_FILES
        }
        required = {
            "methods/cosydelay_v9_clean_from_scratch/integration.py",
            "methods/cosydelay_v9_clean_from_scratch/metrics.py",
            "methods/cosydelay_v9_clean_from_scratch/clean_fitter.py",
            "methods/cosydelay_v9_clean_from_scratch/diagnostics.py",
            "methods/cosydelay_v9_clean_from_scratch/source_manifest.py",
            "methods/cosydelay_lbfgsb_r10_parallel4_v3/fitter.py",
            "methods/cosydelay_lbfgsb_r10_parallel4_v3/physics_audit.py",
            "methods/cosydelay_lbfgsb_r10_parallel3/parallel_fitter.py",
            "methods/prospective_lbfgsb_jacobian_v1/jacobian_fitter.py",
        }
        self.assertTrue(required.issubset(relative))
        self.assertNotIn(
            "methods/cosydelay_lbfgsb_r10_parallel4_v3/training_selection.py",
            relative,
        )
        self.assertEqual(len(SOURCE_FILES), len(set(path.resolve() for path in SOURCE_FILES)))

    def test_clean_fitter_exposes_no_polish_and_relabels_diagnostics(self):
        fitter_type = clean_fitter.CleanMixedRestartJacobianFitter
        self.assertFalse(hasattr(fitter_type, "fit_final_with_polish"))
        fitter = fitter_type.__new__(fitter_type)
        fitter.parallel_workers = 4
        fitter.maxiter = 1000
        fitter.maxfun = 10000
        fitter._executor = object()
        diagnostics = {}
        with patch.object(
            clean_fitter,
            "fit_lane_parameters_mixed_jacobian_parallel",
            return_value={"E_T": {"a1": 1.0}},
        ):
            result = fitter.fit(
                universal_expr="Cycle_Time*a1*flow_lane/GR_phase",
                diagnostics=diagnostics,
            )
        self.assertEqual(result, {"E_T": {"a1": 1.0}})
        self.assertEqual(
            diagnostics["start_strategy_id"],
            clean_fitter.CLEAN_START_STRATEGY_ID,
        )

    def test_batch_integrity_rejects_post_freeze_source_or_data_change(self):
        source_hashes = {"method.py": "source-sha"}
        protocol = {
            "method_id": CLEAN_POLICY.method_id,
            "policy": CLEAN_POLICY.to_dict(),
            "source_sha256": source_hashes,
            "training_sha256": {"1": {"sha256": "train-sha"}},
        }
        result = {
            "formal_protocol_complete": True,
            "method_id": CLEAN_POLICY.method_id,
            "intersection_id": 1,
            "policy": CLEAN_POLICY.to_dict(),
            "source_sha256": source_hashes,
            "train_file_sha256": "train-sha",
            "selection_source": "full_training_evolution_only",
            "accessed_splits": ["train"],
            "project_artifact_read_policy": "allowlist_only",
            "outer_validation_accessed": False,
            "test_file_opened": False,
            "post_evolution_cv_reranking": False,
            "post_evolution_refit": False,
            "incumbent_argument_supported": False,
            "incumbent_injected_into_initial_population": False,
            "initial_population_all_generated_in_current_run": True,
            "initial_population_size_verified": CLEAN_POLICY.population,
            "completed_generations": CLEAN_POLICY.generations,
            "legacy_validation_labels_removed_from_formal_history": True,
            "selected_enhanced_physics_from_evolution": {"joint_pass": True},
        }
        launcher.validate_formal_result(result, protocol, intersection=1)
        result["source_sha256"] = {"method.py": "changed"}
        with self.assertRaisesRegex(RuntimeError, "integrity"):
            launcher.validate_formal_result(result, protocol, intersection=1)

    def test_interpretability_and_compute_diagnostics_are_nonselective(self):
        expression = "Cycle_Time*a1*flow_lane/(GR_phase**a2)"
        quality = audit_parameter_quality(
            expression,
            {"E_T": {"a1": 999.99, "a2": 0.05}},
            CLEAN_POLICY.coefficient_bounds,
        )
        self.assertTrue(quality["diagnostic_only_not_used_for_selection"])
        self.assertEqual(quality["near_boundary_values"], 2)
        self.assertFalse(quality["parameter_identifiability_proven"])
        compute = summarize_compute_efficiency(
            [
                {
                    "event": "evaluated",
                    "evaluation_details": {
                        "fit_wall_seconds": 2.0,
                        "physical_wall_seconds": 0.5,
                        "fit": {
                            "total_function_evaluations": 100,
                            "total_gradient_evaluations": 90,
                            "start_strategy_id": clean_fitter.CLEAN_START_STRATEGY_ID,
                            "formal_method_adapter": "clean",
                        },
                    },
                }
            ]
        )
        self.assertEqual(compute["optimizer_function_evaluations_sum"], 100)
        self.assertEqual(compute["fit_wall_seconds_sum"], 2.0)

    def test_initial_and_offspring_are_both_generated_in_current_run(self):
        population, adaptation, calls = fake_modules()
        with patch.object(
            integration, "install_structural_prefit_gate", fake_prefit_gate
        ), patch.object(
            integration, "MixedRestartJacobianFitter", FakeFitter
        ):
            with self.install(population, adaptation) as runtime:
                self.assertIn(
                    "Cycle_Time * flow_lane / GR_phase",
                    adaptation.PHYSICAL_REQUIREMENTS_STANDARD,
                )
                self.assertIn(
                    "Do not use subtraction",
                    adaptation.PHYSICAL_REQUIREMENTS_STANDARD,
                )
                self.assertIn(
                    "a9, a10, and every higher identifier",
                    adaptation.PHYSICAL_REQUIREMENTS_STANDARD,
                )
                initial = population.safe_generate_universal_lane_expression(
                    {}, [], mutation_type="initial"
                )
                offspring = population.safe_generate_universal_lane_expression(
                    {},
                    [],
                    "large",
                    initial[0],
                    initial[1],
                    initial[2],
                    (True, "pass"),
                    1,
                )
                self.assertEqual(len(calls), 2)
                self.assertFalse(runtime.incumbent_injected)
                self.assertEqual(
                    [item["source"] for item in runtime.generation_audit],
                    ["llm_generated_in_current_run"] * 2,
                )
                self.assertIsNone(runtime.generation_audit[0]["parent_expression"])
                self.assertEqual(
                    runtime.generation_audit[1]["parent_expression"], initial[0]
                )
                self.assertFalse(
                    any(item["external_incumbent"] for item in runtime.generation_audit)
                )
                self.assertEqual(
                    FakeFitter.instances[0].parents,
                    [(initial[0], None), (offspring[0], initial[0])],
                )
            self.assertEqual(adaptation.PHYSICAL_REQUIREMENTS_STANDARD, "standard")

    def test_compact_repair_feedback_is_actionable_and_hides_sympy_detail(self):
        guidance = integration._compact_physical_repair_guidance(
            "R9 pre-fit: NotImplementedError: Not sure of sign of 2 - a8"
        )
        self.assertIn("explicit uncancellable fixed /GR_phase", guidance)
        self.assertIn("do not use fitted green exponents", guidance)
        self.assertNotIn("NotImplementedError", guidance)

    def test_evaluator_rejects_validation_and_wrong_training_objects(self):
        population, adaptation, _ = fake_modules()
        with patch.object(
            integration, "install_structural_prefit_gate", fake_prefit_gate
        ), patch.object(
            integration, "MixedRestartJacobianFitter", FakeFitter
        ):
            with self.install(population, adaptation):
                with self.assertRaisesRegex(RuntimeError, "does not accept Validation"):
                    population.evaluate_expression_with_fitting(
                        "Cycle_Time*a1*flow_lane/GR_phase",
                        self.frame,
                        ["E_T"],
                        {"E_T": "E"},
                        self.targets,
                        1,
                        None,
                        optimizer_restarts=10,
                        df_validation=pd.DataFrame({"x": [2.0]}),
                    )
                with self.assertRaisesRegex(RuntimeError, "installed Training frame"):
                    population.evaluate_expression_with_fitting(
                        "Cycle_Time*a1*flow_lane/GR_phase",
                        self.frame.copy(),
                        ["E_T"],
                        {"E_T": "E"},
                        self.targets,
                        1,
                        None,
                        optimizer_restarts=10,
                    )
                with self.assertRaisesRegex(ValueError, "exactly 10"):
                    population.evaluate_expression_with_fitting(
                        "Cycle_Time*a1*flow_lane/GR_phase",
                        self.frame,
                        ["E_T"],
                        {"E_T": "E"},
                        self.targets,
                        1,
                        None,
                        optimizer_restarts=9,
                    )

    def test_data_guard_allows_exact_train_and_blocks_other_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "Intersection_1_Train.jsonl"
            guard = SplitAccessGuard(train)
            guard.audit_event("open", (str(train), "r", 0))
            self.assertEqual(guard.records[-1]["split"], "train")
            self.assertTrue(guard.records[-1]["allowed"])
            for name in (
                "Intersection_1_Validation.jsonl",
                "Intersection_1_Test.jsonl",
                "Intersection_2_Train.jsonl",
            ):
                with self.assertRaises(PermissionError, msg=name):
                    guard.audit_event("open", (str(root / name), "r", 0))

    def test_data_guard_blocks_nonstandard_historical_artifact_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "data" / "Intersection_1_Train.jsonl"
            current_audit = root / "current" / "llm_audit.jsonl"
            guard = SplitAccessGuard(
                train,
                guarded_roots=(root,),
                allowed_artifact_paths=(current_audit,),
            )
            guard.audit_event("open", (str(train), "r", 0))
            guard.audit_event("open", (str(current_audit), "r", 0))
            self.assertEqual(
                {item["split"] for item in guard.records},
                {"train", "artifact"},
            )
            self.assertEqual(accessed_dataset_splits(guard.records), ["train"])
            for name in (
                "result.json",
                "renamed_old_winner.json",
                "cached_expression.pkl",
                "unlabelled_data.csv",
            ):
                with self.assertRaises(PermissionError, msg=name):
                    guard.audit_event("open", (str(root / "archive" / name), "r", 0))
            before = len(guard.records)
            guard.audit_event(
                "open", (str(root / "archive" / "result.json"), "w", 0)
            )
            self.assertEqual(len(guard.records), before)

    def test_formal_history_removes_legacy_validation_labels(self):
        raw = {
            "validation_r2": None,
            "validation_feedback": "physical rules passed",
            "nested": {
                "validation_rmse": None,
                "outer_validation_accessed": False,
                "train_r2": 0.8,
            },
        }
        cleaned = runner.sanitize_formal_history(raw)
        self.assertEqual(cleaned["physics_feedback"], "physical rules passed")
        self.assertEqual(cleaned["nested"], {"train_r2": 0.8})
        self.assertNotIn("validation_r2", cleaned)

    def test_physical_prompt_does_not_call_a_rule_check_validation(self):
        prompt = adaptation_pipeline.build_universal_lane_mutation_prompt(
            "Cycle_Time*a1*flow_lane/GR_phase",
            "",
            "physical parent",
            (True, "All physical rules passed"),
            "large",
            ["flow_lane", "GR_phase", "Cycle_Time"],
            include_physical_knowledge=True,
        )
        self.assertIn("Physical-audit status", prompt)
        self.assertNotIn("Validation status", prompt)

    def test_strict_parser_rejects_code_and_undocumented_operators(self):
        valid = "Cycle_Time*a1*flow_lane/(GR_phase**a2) + Cycle_Time*a3*log(1+flow_lane)"
        parsed, _ = parse_symbolic_expression(valid)
        self.assertIsNotNone(parsed)
        rejected = (
            "__import__('os').system('echo unsafe')",
            "open('secret.txt').read()",
            "flow_lane.__class__",
            "sin(flow_lane)",
            "Cycle_Time*a9*flow_lane/GR_phase",
            "[flow_lane][0]",
            "lambda: flow_lane",
        )
        for expression in rejected:
            with self.assertRaises(ValueError, msg=expression):
                parse_symbolic_expression(expression)

    def test_formal_reporting_metrics_match_data_driven_definition(self):
        targets = {
            "A": pd.Series([1.0, 2.0, 4.0]),
            "B": pd.Series([2.0, 5.0, 8.0]),
        }
        predictions = {
            "A": np.asarray([1.2, 1.7, 3.5]),
            "B": np.asarray([2.3, 4.0, 9.0]),
        }
        metrics = score_predictions(targets, predictions)
        expected_r2 = np.mean(
            [
                r2_score(targets[name], predictions[name])
                for name in targets
            ]
        )
        truth = np.concatenate([targets[name].to_numpy() for name in targets])
        predicted = np.concatenate([predictions[name] for name in targets])
        self.assertAlmostEqual(metrics["macro_raw_r2"], expected_r2)
        expected_paper_r2 = np.mean(
            [
                max(0.0, r2_score(targets[name], predictions[name]))
                for name in targets
            ]
        )
        self.assertAlmostEqual(
            metrics["macro_nonnegative_r2"], expected_paper_r2
        )
        self.assertAlmostEqual(
            metrics["pooled_rmse"],
            np.sqrt(mean_squared_error(truth, predicted)),
        )
        self.assertAlmostEqual(
            metrics["pooled_mae"],
            mean_absolute_error(truth, predicted),
        )

    def test_evolution_fitness_exactly_matches_paper_definition(self):
        metrics = {"macro_nonnegative_r2": 0.37}
        self.assertAlmostEqual(
            integration._fitness(metrics, 1.0, CLEAN_POLICY), 1.37
        )
        self.assertAlmostEqual(
            integration._fitness(metrics, 0.0, CLEAN_POLICY), 0.37
        )
        self.assertEqual(
            integration._fitness(
                {"macro_nonnegative_r2": 1.5}, 1.0, CLEAN_POLICY
            ),
            2.0,
        )

    def test_expression_attempt_audit_records_invalid_duplicate_and_accept(self):
        responses = iter(
            (
                "not the required output format",
                "### Expression\ny = Cycle_Time*a1*flow_lane/GR_phase\n"
                "### Explanation\nduplicate",
                "### Expression\ny = Cycle_Time*a1*flow_lane*exp(a2/GR_phase)\n"
                "### Explanation\naccepted",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / "attempts.jsonl"
            with patch.object(
                adaptation_pipeline, "run_llm", side_effect=lambda prompt: next(responses)
            ), patch.dict(
                os.environ,
                {"EXPRESSION_ATTEMPT_AUDIT_LOG": str(audit_path)},
            ):
                expression, _, _ = (
                    adaptation_pipeline.safe_generate_universal_lane_expression(
                        {},
                        ["flow_lane", "GR_phase", "Cycle_Time"],
                        mutation_type="initial",
                        max_retries=3,
                        excluded_expressions=[
                            "Cycle_Time*a1*flow_lane/GR_phase"
                        ],
                    )
                )
            self.assertEqual(
                expression,
                "Cycle_Time*a1*flow_lane*exp(a2/GR_phase)",
            )
            records = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [item["status"] for item in records],
                ["format_invalid", "canonical_duplicate", "accepted"],
            )

    def test_coefficient_budget_retry_names_forbidden_identifiers(self):
        prompts = []
        responses = iter(
            (
                "### Expression\ny = Cycle_Time*a1*flow_lane*"
                "(1+a9*flow_lane)/GR_phase\n### Explanation\ntoo many",
                "### Expression\ny = Cycle_Time*a1*flow_lane/GR_phase\n"
                "### Explanation\ncorrected",
            )
        )

        def respond(prompt):
            prompts.append(prompt)
            return next(responses)

        with patch.object(adaptation_pipeline, "run_llm", side_effect=respond):
            expression, _, _ = (
                adaptation_pipeline.safe_generate_universal_lane_expression(
                    {},
                    ["flow_lane", "GR_phase", "Cycle_Time"],
                    mutation_type="initial",
                    max_retries=2,
                    include_physical_knowledge=True,
                )
            )
        self.assertEqual(expression, "Cycle_Time*a1*flow_lane/GR_phase")
        self.assertEqual(len(prompts), 2)
        self.assertIn("a9, a10, and all higher identifiers are forbidden", prompts[1])

    def test_prefit_gate_rejects_units_direction_and_zero_flow_failures(self):
        self.assertTrue(
            evaluate_structural_prefit_gate(
                "Cycle_Time*a1*flow_lane/GR_phase"
            ).passed
        )
        invalid = {
            "zero_flow": "Cycle_Time*(a1+a2*flow_lane)/GR_phase",
            "units": "a1*flow_lane/GR_phase",
            "green_direction": "Cycle_Time*a1*flow_lane*GR_phase",
            "unsupported_function": "Cycle_Time*a1*flow_lane/sin(GR_phase)",
        }
        for label, expression in invalid.items():
            self.assertFalse(
                evaluate_structural_prefit_gate(expression).passed,
                msg=label,
            )

    def test_clean_prefit_moves_structurally_impossible_r9_before_fitting(self):
        population, adaptation, _ = fake_modules()
        audit = []

        @contextmanager
        def recording_gate(*, population_module, adaptation_module):
            def base_validator(expression):
                audit.append({"expression": expression, "passed": True})
                return True, "base pass"

            adaptation_module.validate_candidate_expression = base_validator
            yield audit

        with patch.object(
            integration, "install_structural_prefit_gate", recording_gate
        ), patch.object(
            integration, "MixedRestartJacobianFitter", FakeFitter
        ):
            with self.install(population, adaptation):
                finite_at_zero_green = (
                    "Cycle_Time*a1*flow_lane/(1+GR_phase)"
                )
                passed, reason = adaptation.validate_candidate_expression(
                    finite_at_zero_green
                )
                self.assertFalse(passed)
                self.assertIn("explicit uncancellable fixed /GR_phase", reason)
                self.assertFalse(audit[-1]["passed"])
                self.assertIn(
                    "coefficient_robust_endpoint_audit", audit[-1]
                )

    def test_enhanced_fitted_physics_passes_simple_physical_expression(self):
        result = audit_fitted_physics(
            "Cycle_Time*a1*flow_lane/GR_phase",
            {"E_T": {"a1": 1.0}},
            ["E_T"],
        )
        self.assertTrue(result["joint_pass"], result["errors"])
        self.assertTrue(result["symbolic_r8_exact"])
        self.assertTrue(result["symbolic_r9_positive_infinity"])
        self.assertTrue(result["dense_flow_nondecreasing"])
        self.assertTrue(result["dense_green_nonincreasing"])

    def test_fitted_failure_family_is_rejected_before_another_fit(self):
        population, adaptation, _ = fake_modules()

        def evaluated_base(*args, **kwargs):
            return (
                1.0,
                0.5,
                1.5,
                {"E_T": {"a1": 1.0}},
                "",
                {
                    "status": "evaluated",
                    "verifier": {
                        "joint_pass": True,
                        "rule_scores": {"R2_nondecreasing_flow": 1.0},
                    },
                },
            )

        population.evaluate_expression_with_fitting = evaluated_base
        policy = replace(
            CLEAN_POLICY,
            structural_diversity_mode="family_unique",
            reject_fitted_structural_families=True,
        )
        failed_audit = {
            "audit_id": "test",
            "joint_pass": False,
            "standard_rule_scores": {"R2_nondecreasing_flow": 0.0},
            "errors": ["failed dense_flow_nondecreasing"],
        }
        metrics = {
            "macro_nonnegative_r2": 0.5,
            "macro_rmse": 1.0,
            "macro_mae": 1.0,
        }
        with patch.object(
            integration, "install_structural_prefit_gate", fake_prefit_gate
        ), patch.object(
            integration, "MixedRestartJacobianFitter", FakeFitter
        ), patch.object(
            integration,
            "calculate_approach_delays_from_universal",
            return_value={"E": np.asarray([1.0])},
        ), patch.object(
            integration, "score_predictions", return_value=metrics
        ), patch.object(
            integration, "audit_fitted_physics", return_value=failed_audit
        ):
            with integration.install_clean_single_evolution(
                df_train=self.frame,
                targets=self.targets,
                lanes=["E_T"],
                lane_to_approach={"E_T": "E"},
                intersection_id=1,
                policy=policy,
                population_module=population,
                adaptation_module=adaptation,
            ) as runtime:
                failed_expression = "Cycle_Time*a1*flow_lane/GR_phase"
                with self.assertRaisesRegex(RuntimeError, "R2"):
                    population.evaluate_expression_with_fitting(
                        failed_expression,
                        self.frame,
                        ["E_T"],
                        {"E_T": "E"},
                        self.targets,
                        1,
                        None,
                        optimizer_restarts=10,
                    )
                renamed_same_family = "Cycle_Time*a2*flow_lane/GR_phase"
                passed, reason = adaptation.validate_candidate_expression(
                    renamed_same_family
                )
                self.assertFalse(passed)
                self.assertIn("FITTED STRUCTURAL FAMILY REJECTION", reason)
                self.assertIn("Renaming coefficients", reason)
                self.assertEqual(len(runtime.rejected_family_feedback), 1)
                self.assertEqual(
                    runtime.prefit_audit[-1]["gate_id"],
                    "fitted_rejected_family_cache_2026_08_11_v1",
                )

    def test_formal_runner_emits_machine_checkable_clean_invariants(self):
        expression = "Cycle_Time*a1*flow_lane/GR_phase"
        parameters = {"E_T": {"a1": 1.0}}
        metrics = {
            "macro_r2": 0.8,
            "macro_raw_r2": 0.8,
            "macro_rmse": 4.0,
            "macro_mae": 3.0,
            "pooled_rmse": 4.1,
            "pooled_mae": 3.0,
            "normalized_rmse": 0.4,
            "normalized_mae": 0.3,
            "by_approach": {},
        }
        selected = {
            "parameters": parameters,
            "metrics": metrics,
            "fitness": 1.8,
            "enhanced_physics": {"joint_pass": True, "audit_id": "test"},
        }
        history = [
            {
                "event": "evaluated",
                "generation": 0,
                "individual": index,
                "expression": f"{expression}*{index}",
            }
            for index in range(1, 11)
        ]
        history.append(
            {
                "event": "search_budget_summary",
                "completed_generations": 10,
                "candidate_evaluations": 110,
                "final_selection_policy": (
                    "best_training_fitness_from_surviving_population"
                ),
            }
        )
        generation_audit = [
            {
                "sequence": index,
                "expression": f"{expression}*{index}",
                "canonical_expression": f"test-{index}",
                "mutation_type": "initial",
                "parent_expression": None,
                "source": "llm_generated_in_current_run",
                "external_incumbent": False,
            }
            for index in range(1, 11)
        ]
        runtime = SimpleNamespace(
            evaluations={expression: selected},
            incumbent_injected=False,
            prefit_audit=[],
            fitted_rejections=[],
            generation_audit=generation_audit,
        )

        @contextmanager
        def fake_install(**kwargs):
            yield runtime

        def fake_evolve(**kwargs):
            self.assertIsNone(kwargs["max_wall_seconds"])
            self.assertEqual(kwargs["pop_size"], 10)
            self.assertEqual(kwargs["generations"], 10)
            self.assertEqual(kwargs["optimizer_restarts"], 10)
            self.assertNotIn("df_validation", kwargs)
            return expression, "", "test", parameters, history

        class FakeGuard:
            def __init__(self, path, **kwargs):
                self.records = [
                    {
                        "path": str(Path(path).resolve()),
                        "file_name": Path(path).name,
                        "split": "train",
                        "allowed": True,
                    }
                ]

            def install(self):
                return None

            def disable(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            train_path = data / "Intersection_1_Train.jsonl"
            train_path.write_text("{}\n", encoding="utf-8")
            output = root / "run"
            args = SimpleNamespace(
                intersection=1,
                seed=20260830,
                data_dir=data,
                output=output,
            )
            frame = pd.DataFrame({"Delay_E": [1.0, 2.0]})
            with patch.object(runner, "arguments", return_value=args), patch.object(
                runner,
                "INTERSECTION_CONFIGS",
                {1: {"approaches": ["E"], "movements": {"E": ["T"]}}},
            ), patch.object(
                runner, "load_dataset_flexible", return_value=frame
            ), patch.object(
                runner, "preprocess_data_flexible", return_value=frame
            ), patch.object(
                runner, "SplitAccessGuard", FakeGuard
            ), patch.object(
                runner, "install_clean_single_evolution", fake_install
            ), patch.object(
                runner, "evolve_universal_lane_expression", fake_evolve
            ), patch.object(
                runner,
                "read_jsonl",
                side_effect=lambda path: (
                    [{"status": "accepted"}] * 10
                    if Path(path).name == "expression_attempt_audit.jsonl"
                    else []
                ),
            ), patch.dict(
                os.environ, {"LLM_API_KEY": "unit-test-placeholder"}
            ):
                self.assertEqual(runner.main(), 0)
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertTrue(result["formal_protocol_complete"])
            self.assertEqual(result["accessed_splits"], ["train"])
            self.assertFalse(result["outer_validation_accessed"])
            self.assertFalse(result["test_file_opened"])
            self.assertIsNone(result["incumbent_result"])
            self.assertFalse(result["incumbent_argument_supported"])
            self.assertTrue(result["initial_population_all_generated_in_current_run"])
            self.assertEqual(result["initial_population_size_verified"], 10)
            self.assertFalse(result["post_evolution_cv_reranking"])
            self.assertFalse(result["post_evolution_refit"])
            self.assertTrue(result["fresh_unseen_test_still_required"])


if __name__ == "__main__":
    unittest.main()
