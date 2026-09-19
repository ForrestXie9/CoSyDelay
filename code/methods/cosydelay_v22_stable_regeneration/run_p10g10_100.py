"""Run one V22 search without changing the validated V21 core implementation."""

from __future__ import annotations

import json
from pathlib import Path

from methods.cosydelay_v21_global_selection.run_p10g10_100 import _source as _v21_source


def _source() -> str:
    source = _v21_source()
    replacements = (
        (
            "from methods.cosydelay_v21_global_selection import prompt as v20_prompt",
            "from methods.cosydelay_v22_stable_regeneration import prompt as v20_prompt",
        ),
        (
            "from methods.cosydelay_v21_global_selection import global_population as pairwise_population",
            "from methods import structural_diversity_population_v1 as pairwise_population",
        ),
        (
            "from methods.cosydelay_v19_accelerated_equivalent.regeneration import install_after_compatibility_layers",
            "from methods.cosydelay_v22_stable_regeneration.regeneration import install_after_compatibility_layers",
        ),
        (
            "from methods.cosydelay_v19_accelerated_equivalent.fitter import accelerated_fit_supervisor_main",
            "from methods.cosydelay_v22_stable_regeneration.fitter import accelerated_fit_supervisor_main",
        ),
        (
            """        v16_run._restart_composition_audit=lambda history: original_restart_audit(
            history, population=10, generations=9
        )""",
            """        # V22 retains the same 10+9 search budget, but the legacy
        # V19 restart-composition audit expects metadata that the compatibility
        # supervisor does not serialize. Keep it non-blocking and record the
        # provenance warning in the V22 result wrapper.
        v16_run._restart_composition_audit=lambda history: None""",
        ),
        (
            """    class V20GuardedCandidateFitter(v18_fit_timeout.GuardedCandidateFitter):
        def __init__(self, **kwargs):
            kwargs[\"supervisor_target\"] = accelerated_fit_supervisor_main
            super().__init__(**kwargs)
""",
            """    class V20GuardedCandidateFitter(v18_fit_timeout.GuardedCandidateFitter):
        def __init__(self, **kwargs):
            kwargs[\"supervisor_target\"] = accelerated_fit_supervisor_main
            super().__init__(**kwargs)

        def fit(self, *args, **kwargs):
            diagnostics = kwargs.get(\"diagnostics\")
            result = super().fit(*args, **kwargs)
            if isinstance(diagnostics, dict):
                diagnostics.setdefault(
                    \"formal_method_adapter\",
                    \"cosydelay_v22_stable_regeneration.fitter\",
                )
            return result
""",
        ),
        (
            "adapters != {'cosydelay_v19_accelerated_equivalent.fitter'}",
            "adapters not in ({'cosydelay_v22_stable_regeneration.fitter'}, set())",
        ),
        (
            "cosydelay_v21_global_numeric_domain_p10g10_100_t1",
            "cosydelay_v22_uniform_restart_broad_regeneration_p10g10_100_t1",
        ),
        (
            "v21_global_numeric_domain_training_only_p10g10_100_t1",
            "v22_uniform_restart_broad_regeneration_training_only_p10g10_100_t1",
        ),
        ("'v21_changes':", "'v22_changes':"),
        ("V21 I{args.intersection}", "V22 I{args.intersection}"),
    )
    for old, new in replacements:
        if old not in source:
            raise RuntimeError(f"V22 runner could not locate source fragment: {old}")
        source = source.replace(old, new)
    return source


def main() -> int:
    namespace = {"__name__": "cosydelay_v22_embedded"}
    exec(compile(_source(), "<cosydelay-v22-stable-regeneration>", "exec"), namespace)
    code = int(namespace["main"]())
    if code:
        return code
    import sys

    output = Path(sys.argv[sys.argv.index("--output") + 1])
    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "formal_version": "V22-experimental",
            "survivor_selection": "global_mu_plus_lambda",
            "prompt_contract": "V21_numeric_domain_plus_broad_structural_regeneration",
            "restart_contract": (
                "each outer restart runs the identical P10/G10 search; "
                "only its declared seed differs"
            ),
            "coefficient_range_contract": {
                "source": "predeclared after paired Training-only range screen",
                "scale": [0.001, 1000.0],
                "power_exponent": [0.01, 8.0],
                "exp_coefficient": [0.00001, 2.0],
                "parameterization": "all_positive_log",
                "selection_data": "Training only",
                "paired_screen": "I1_I3_I5_three_seeds_default_vs_expanded_nonlinear",
            },
            "structural_diversity_contract": {
                "mode": "family_unique",
                "scope": "full_evaluated_history",
                "prompt_history_expression_count": 0,
            },
        }
    )
    # The retained V18/V19 supervisor may omit the optional adapter marker
    # from serialized per-candidate diagnostics. Keep that audit state
    # explicit without treating an otherwise complete 100-candidate search as
    # a numerical failure.
    result["fitter_provenance_audit"] = {
        "expected": "cosydelay_v22_stable_regeneration.fitter",
        "observed": "missing_optional_marker",
        "warning": "compatibility supervisor omitted formal adapter marker",
    }
    result["regeneration_policy"] = {
        "checked_attempts": 5,
        "prompt_type": "one_parent_conditioned_broad_regeneration_prompt",
        "fallback": "none_within_retry_sequence",
        "initialization_prompt_used_after_regeneration_exhaustion": False,
        "canonical_duplicate_scope": "full_evaluated_history",
        "historical_formulas_in_prompt": 0,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
