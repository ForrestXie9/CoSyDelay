"""V22 search with principlewise physical fitness instead of strict joint-pass binary.

Identical to ``run_p10g10_100`` except the V19 binary-joint overlay is not
installed.  Fitness becomes ``train_r2 + mean(seven_rule_scores)`` through the
retained V16 manuscript principlewise evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path

from methods.cosydelay_v22_stable_regeneration.run_p10g10_100 import _source as _v22_binary_source


def _source() -> str:
    source = _v22_binary_source()
    replacements = (
        (
            "from methods.cosydelay_v19_accelerated_equivalent.binary_fitness import install_binary_joint_fitness\n",
            "",
        ),
        (
            """                    with install_prefit_r4_gate(expression_adaptation_lane):
                        with install_binary_joint_fitness(
                            pairwise_population,
                            runtime,
                        ):
                            yield runtime""",
            """                    with install_prefit_r4_gate(expression_adaptation_lane):
                        yield runtime""",
        ),
        (
            "'fitness_definition':'training_r2_plus_strict_joint_pass_binary'",
            "'fitness_definition':'training_r2_plus_principlewise_physical_mean'",
        ),
        (
            "'strict_joint_pass_binary_component':1.0 if bool(result.get('selected_enhanced_physics',{}).get('joint_pass')) else 0.0,'principlewise_fraction_role':'diagnostic_only_not_used_for_fitness'",
            "'principlewise_fraction_role':'used_for_fitness','strict_joint_pass_role':'diagnostic_only_not_used_for_fitness'",
        ),
        (
            "'strict_joint_pass_binary_fitness'",
            "'principlewise_physical_mean_fitness'",
        ),
        (
            "cosydelay_v22_uniform_restart_broad_regeneration_p10g10_100_t1",
            "cosydelay_v22_principlewise_physical_mean_p10g10_100_t1",
        ),
        (
            "v22_uniform_restart_broad_regeneration_training_only_p10g10_100_t1",
            "v22_principlewise_physical_mean_training_only_p10g10_100_t1",
        ),
        ("V22 I{args.intersection}", "V22-principlewise I{args.intersection}"),
    )
    for old, new in replacements:
        if old not in source:
            raise RuntimeError(f"V22 principlewise runner could not locate fragment: {old!r}")
        source = source.replace(old, new)
    return source


def main() -> int:
    namespace = {"__name__": "cosydelay_v22_principlewise_embedded"}
    exec(compile(_source(), "<cosydelay-v22-principlewise>", "exec"), namespace)
    code = int(namespace["main"]())
    if code:
        return code
    import sys

    output = Path(sys.argv[sys.argv.index("--output") + 1])
    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    selected_physics = result.get("selected_enhanced_physics", {})
    result.update(
        {
            "formal_version": "V22-principlewise",
            "fitness_definition": "training_r2_plus_principlewise_physical_mean",
            "principlewise_fraction_role": "used_for_fitness",
            "strict_joint_pass_role": "diagnostic_only_not_used_for_fitness",
            "selected_principlewise_physical_mean": float(
                selected_physics.get("score", 0.0)
            ),
            "selected_strict_joint_pass": bool(selected_physics.get("joint_pass")),
            "physical_scoring_contract": {
                "mode": "principlewise",
                "formula": "train_r2 + mean(seven_rule_scores)",
                "binary_joint_pass_used_for_fitness": False,
            },
        }
    )
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
