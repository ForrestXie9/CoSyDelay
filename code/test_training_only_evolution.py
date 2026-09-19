"""Regression tests that frozen validation cannot steer evolution."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import population_evolution_lane as evolution


class TrainingOnlyFitnessTest(unittest.TestCase):
    @staticmethod
    def _mock_evaluation(train_r2, physical_pass, label):
        return (
            1.0 if physical_pass else 0.0,
            train_r2,
            train_r2,
            {"L": {"a1": 1.0}},
            label,
            {
                "status": "evaluated",
                "verifier": {
                    "joint_pass": physical_pass,
                    "score": 1.0 if physical_pass else 0.0,
                    "rule_scores": {
                        "R8_zero_flow_boundary": 1.0,
                        "R9_zero_green_limit": 1.0 if physical_pass else 0.0,
                    },
                    "lane_errors": (
                        {}
                        if physical_pass
                        else {
                            "L": [
                                "R9: delay at near-zero green is not severe enough"
                            ]
                        }
                    ),
                },
                "validation_r2": None,
                "validation_rmse": None,
                "train_rmse": 0.1,
                "selection_accuracy": train_r2,
            },
        )

    def test_validation_accuracy_is_diagnostic_not_fitness(self):
        train = pd.DataFrame({"split": [0, 0, 0]})
        validation = pd.DataFrame({"split": [1, 1, 1]})
        train_targets = {"A": pd.Series([0.0, 1.0, 2.0])}
        validation_targets = {"A": pd.Series([0.0, 1.0, 2.0])}
        physical = SimpleNamespace(
            joint_pass=True,
            score=1.0,
            lane_errors={},
            to_dict=lambda: {
                "joint_pass": True,
                "score": 1.0,
                "rule_scores": {},
            },
        )

        def predictions(*, df, **_kwargs):
            if int(df["split"].iloc[0]) == 0:
                return {"A": np.asarray([0.0, 1.0, 2.0])}
            return {"A": np.asarray([2.0, 1.0, 0.0])}

        with (
            patch.object(evolution, "validate_expression_basic", return_value=(True, "ok")),
            patch.object(
                evolution,
                "fit_lane_parameters_to_approaches",
                return_value={"L": {"a1": 1.0}},
            ),
            patch.object(
                evolution,
                "score_fitted_lanes_principlewise",
                return_value=physical,
            ),
            patch.object(
                evolution,
                "calculate_approach_delays_from_universal",
                side_effect=predictions,
            ),
            patch.object(
                evolution,
                "format_physical_validation_feedback",
                return_value="ok",
            ),
        ):
            _, train_r2, fitness, _, _, details = (
                evolution.evaluate_expression_with_fitting(
                    "Cycle_Time * a1 * flow_lane / GR_phase",
                    train,
                    ["L"],
                    {"L": "A"},
                    train_targets,
                    1,
                    {"split": "train"},
                    physics_weight=1.0,
                    df_validation=validation,
                    validation_targets=validation_targets,
                    prepared_validation_context={"split": "validation"},
                )
            )

        self.assertAlmostEqual(train_r2, 1.0)
        self.assertAlmostEqual(details["validation_r2"], 0.0)
        self.assertAlmostEqual(details["selection_accuracy"], 1.0)
        self.assertAlmostEqual(fitness, 2.0)
        self.assertEqual(details["selection_source"], "training")

    def test_evolution_scores_validation_only_after_best_is_selected(self):
        train = pd.DataFrame({"split": [0, 0, 0]})
        validation = pd.DataFrame({"split": [1, 1, 1]})
        targets = {"A": pd.Series([0.0, 1.0, 2.0])}
        evaluation_calls = []

        def evaluate_candidate(*args, **kwargs):
            evaluation_calls.append((args, kwargs))
            return (
                1.0,
                0.8,
                1.8,
                {"L": {"a1": 1.0}},
                "ok",
                {
                    "status": "evaluated",
                    "verifier": {
                        "joint_pass": True,
                        "score": 1.0,
                        "rule_scores": {},
                    },
                    "validation_r2": None,
                    "validation_rmse": None,
                    "train_rmse": 0.1,
                    "selection_accuracy": 0.8,
                },
            )

        with (
            patch.object(
                evolution,
                "safe_generate_universal_lane_expression",
                side_effect=[
                    (
                        "Cycle_Time * a1 * flow_lane / GR_phase",
                        "",
                        "initial",
                    ),
                    (
                        "Cycle_Time * a2 * flow_lane / GR_phase",
                        "",
                        "offspring",
                    ),
                ],
            ),
            patch.object(
                evolution,
                "prepare_optimization_context",
                side_effect=lambda df, *_args: {
                    "split": int(df["split"].iloc[0])
                },
            ),
            patch.object(
                evolution,
                "evaluate_expression_with_fitting",
                side_effect=evaluate_candidate,
            ),
            patch.object(
                evolution,
                "_score_fitted_expression_on_dataset",
                return_value=(
                    0.25,
                    {"A": 0.25},
                    0.5,
                    {"A": 0.5},
                ),
            ) as final_validation_score,
        ):
            result = evolution.evolve_universal_lane_expression(
                df_train=train,
                lanes=["L"],
                lane_to_approach={"L": "A"},
                approach_targets=targets,
                universal_features=["flow_lane", "GR_phase", "Cycle_Time"],
                generations=1,
                pop_size=1,
                df_validation=validation,
                validation_targets=targets,
            )

        history = result[-1]
        self.assertEqual(len(evaluation_calls), 2)
        self.assertTrue(
            all("df_validation" not in kwargs for _, kwargs in evaluation_calls)
        )
        final_validation_score.assert_called_once()
        final_event = [
            item for item in history
            if item.get("event") == "final_validation_evaluation"
        ]
        evaluated_events = [
            item for item in history
            if item.get("event") == "evaluated"
        ]
        self.assertEqual(len(final_event), 1)
        self.assertTrue(
            all(
                item["evaluation_details"].get("validation_r2") is None
                for item in evaluated_events
            )
        )
        self.assertEqual(final_event[0]["validation_r2"], 0.25)
        self.assertTrue(final_event[0]["fitness_unchanged_by_validation"])

    def test_feasible_archive_controls_final_selection_using_training_only(self):
        train = pd.DataFrame({"split": [0, 0, 0]})
        targets = {"A": pd.Series([0.0, 1.0, 2.0])}
        feasible = "Cycle_Time * a1 * flow_lane / GR_phase"
        inaccurate = "Cycle_Time * a2 * flow_lane / GR_phase"
        with (
            patch.object(
                evolution,
                "safe_generate_universal_lane_expression",
                side_effect=[
                    (feasible, "", "feasible"),
                    (inaccurate, "", "inaccurate"),
                ],
            ),
            patch.object(
                evolution,
                "prepare_optimization_context",
                return_value={"split": 0},
            ),
            patch.object(
                evolution,
                "evaluate_expression_with_fitting",
                side_effect=[
                    self._mock_evaluation(0.30, True, "passed"),
                    self._mock_evaluation(0.90, False, "failed R9"),
                ],
            ),
        ):
            result = evolution.evolve_universal_lane_expression(
                df_train=train,
                lanes=["L"],
                lane_to_approach={"L": "A"},
                approach_targets=targets,
                universal_features=["flow_lane", "GR_phase", "Cycle_Time"],
                generations=1,
                pop_size=1,
                use_feasible_archive=True,
            )

        self.assertEqual(result[0], feasible)
        summary = next(
            item for item in result[-1] if item.get("event") == "search_budget_summary"
        )
        self.assertEqual(summary["feasible_archive_size"], 1)
        self.assertEqual(
            summary["final_selection_policy"],
            "best_training_r2_from_physical_feasible_archive",
        )

    def test_training_side_early_stop_never_scores_validation(self):
        train = pd.DataFrame({"split": [0, 0, 0]})
        validation = pd.DataFrame({"split": [1, 1, 1]})
        targets = {"A": pd.Series([0.0, 1.0, 2.0])}
        generator = patch.object(
            evolution,
            "safe_generate_universal_lane_expression",
            side_effect=[
                ("Cycle_Time * a1 * flow_lane / GR_phase", "", "first"),
                ("Cycle_Time * a2 * flow_lane / GR_phase", "", "second"),
            ],
        )
        with (
            generator as mocked_generator,
            patch.object(
                evolution,
                "prepare_optimization_context",
                side_effect=lambda df, *_args: {"split": int(df["split"].iloc[0])},
            ),
            patch.object(
                evolution,
                "evaluate_expression_with_fitting",
                side_effect=[
                    self._mock_evaluation(0.80, True, "passed"),
                    self._mock_evaluation(0.79, True, "passed"),
                ],
            ),
            patch.object(
                evolution,
                "_score_fitted_expression_on_dataset",
                return_value=(0.25, {"A": 0.25}, 0.5, {"A": 0.5}),
            ) as final_validation_score,
        ):
            result = evolution.evolve_universal_lane_expression(
                df_train=train,
                lanes=["L"],
                lane_to_approach={"L": "A"},
                approach_targets=targets,
                universal_features=["flow_lane", "GR_phase", "Cycle_Time"],
                generations=2,
                pop_size=2,
                df_validation=validation,
                validation_targets=targets,
                use_feasible_archive=True,
                early_stop_feasible_count=2,
                early_stop_train_gap=0.02,
            )

        self.assertEqual(mocked_generator.call_count, 2)
        final_validation_score.assert_called_once()
        summary = next(
            item for item in result[-1] if item.get("event") == "search_budget_summary"
        )
        self.assertTrue(summary["early_stop_triggered"])
        self.assertEqual(summary["completed_generations"], 0)
        self.assertFalse(summary["validation_accessed_during_evolution"])

    def test_targeted_feedback_uses_fitted_failure_without_lane_labels(self):
        feedback, failed = evolution.augment_targeted_physical_feedback(
            "Failed joint rules: R9",
            {
                "rule_scores": {
                    "R2_nondecreasing_flow": 1.0,
                    "R7_operational_responsiveness": 1.0,
                    "R9_zero_green_limit": 0.0,
                },
                "lane_errors": {
                    "E_L": [
                        "R9: delay at near-zero green is not severe enough (191s)"
                    ]
                },
            },
        )
        self.assertEqual(failed, ["R9_zero_green_limit"])
        self.assertIn("coefficient-robust positive singular", feedback)
        self.assertIn("191s", feedback)
        self.assertNotIn("E_L", feedback)


if __name__ == "__main__":
    unittest.main()
