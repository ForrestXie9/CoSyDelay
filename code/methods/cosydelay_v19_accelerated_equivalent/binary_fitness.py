"""V20 strict binary physical-compliance Fitness overlay."""

from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType
from typing import Iterator


def binary_joint_fitness(training_accuracy: float, joint_pass: bool) -> float:
    return float(
        min(max(float(training_accuracy) + float(bool(joint_pass)), 0.0), 2.0)
    )


@contextmanager
def install_binary_joint_fitness(
    population_module: ModuleType,
    runtime,
) -> Iterator[None]:
    """Use strict joint pass (0/1) as the V20 physical Fitness component."""

    previous_evaluator = population_module.evaluate_expression_with_fitting

    def evaluate_with_binary_joint_pass(*args, **kwargs):
        result = previous_evaluator(*args, **kwargs)
        _, accuracy, _, parameters, feedback, details = result
        verifier = details.get("verifier", {}) or {}
        diagnostic_fraction = float(verifier.get("score", 0.0))
        joint_pass = bool(verifier.get("joint_pass", False))
        binary_component = float(joint_pass)
        fitness = binary_joint_fitness(float(accuracy), joint_pass)
        details.update(
            {
                "score_mode": "strict_joint_pass_binary",
                "selected_physical_component": binary_component,
                "principlewise_diagnostic_fraction": diagnostic_fraction,
                "strict_joint_pass_binary_component": binary_component,
                "fitness_definition": (
                    "training_mean_nonnegative_approach_r2_plus_"
                    "strict_joint_pass_binary"
                ),
                "strict_joint_pass_is_diagnostic_only": False,
                "physical_hard_gate": False,
                "paper_fitness_components": {
                    "mean_nonnegative_approach_r2": float(accuracy),
                    "strict_joint_pass_binary": binary_component,
                    "principlewise_diagnostic_fraction": diagnostic_fraction,
                    "lower_bound": 0.0,
                    "upper_bound": 2.0,
                },
            }
        )
        expression = str(args[0] if args else kwargs["expr"])
        record = runtime.evaluations.get(expression)
        if record is not None:
            record.update({"fitness": fitness, "details": details})
        return (
            binary_component,
            float(accuracy),
            fitness,
            parameters,
            feedback,
            details,
        )

    population_module.evaluate_expression_with_fitting = (
        evaluate_with_binary_joint_pass
    )
    try:
        yield
    finally:
        population_module.evaluate_expression_with_fitting = previous_evaluator
