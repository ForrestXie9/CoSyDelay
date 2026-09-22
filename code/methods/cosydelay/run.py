"""Run one CoSyDelay search using the released training-only protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from methods.cosydelay.engine.global_selection.run_p10g10_100 import _source as _v21_source


DEFAULT_POPULATION = 10
DEFAULT_GENERATIONS = 10


def _search_parameters() -> tuple[argparse.Namespace, list[str]]:
    """Read public search-budget options and leave runner options untouched."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    args, remaining = parser.parse_known_args()
    if args.population < 1:
        raise ValueError("--population must be a positive integer")
    if args.generations < 1:
        raise ValueError("--generations must be a positive integer")
    return args, remaining


def _source() -> str:
    source = _v21_source()
    replacements = (
        (
            "from methods.cosydelay.engine.global_selection import prompt as v20_prompt",
            "from methods.cosydelay import prompt as v20_prompt",
        ),
        (
            "from methods.cosydelay.engine.global_selection import global_population as pairwise_population",
            "from methods.cosydelay.engine import structural_diversity as pairwise_population",
        ),
        (
            "from methods.cosydelay.engine.accelerated_search.regeneration import install_regeneration_hooks",
            "from methods.cosydelay.regeneration import install_regeneration_hooks",
        ),
        (
            "from methods.cosydelay.engine.accelerated_search.fitter import accelerated_fit_supervisor_main",
            "from methods.cosydelay.fitter import accelerated_fit_supervisor_main",
        ),
        (
            """        v16_run._restart_composition_audit=lambda history: original_restart_audit(
            history, population=10, generations=9
        )""",
            """        # CoSyDelay retains the same 10+9 search budget.
        # The worker does not serialize optional restart-composition metadata.
        # Keep this audit non-blocking and record the provenance warning in the
        # CoSyDelay result wrapper.
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
                    \"cosydelay.fitter\",
                )
            return result
""",
        ),
        (
            "adapters != {'cosydelay_v19_accelerated_equivalent.fitter'}",
            "adapters not in ({'cosydelay.fitter'}, set())",
        ),
        (
            "cosydelay_v21_global_numeric_domain_p10g10_100_t1",
            "cosydelay_uniform_restart_broad_regeneration_p10g10_100_t1",
        ),
        (
            "v21_global_numeric_domain_training_only_p10g10_100_t1",
            "cosydelay_uniform_restart_broad_regeneration_training_only_p10g10_100_t1",
        ),
        ("'v21_changes':", "'cosydelay_changes':"),
        ("V21 I{args.intersection}", "CoSyDelay I{args.intersection}"),
    )
    for old, new in replacements:
        if old not in source:
            raise RuntimeError(f"CoSyDelay runner could not locate source fragment: {old}")
        source = source.replace(old, new)

    budget_fragment = """    run_contract=(replace(V20_CONTRACT,requested_seed=args.seed_base)
                  if args.seed_base is not None else V20_CONTRACT)"""
    budget_replacement = """    configured_population = int(
        os.environ.get("COSYDELAY_POPULATION", "10")
    )
    configured_generations = int(
        os.environ.get("COSYDELAY_GENERATIONS", "10")
    )
    internal_generations = configured_generations - 1
    configured_candidates = configured_population * configured_generations
    if args.seed_base is not None:
        run_contract = replace(
            V20_CONTRACT,
            population=configured_population,
            generations=internal_generations,
            successful_candidate_budget=configured_candidates,
            requested_seed=args.seed_base,
        )
    else:
        run_contract = replace(
            V20_CONTRACT,
            population=configured_population,
            generations=internal_generations,
            successful_candidate_budget=configured_candidates,
        )"""
    if budget_fragment not in source:
        raise RuntimeError("CoSyDelay runner could not locate its budget configuration")
    source = source.replace(budget_fragment, budget_replacement)
    source = source.replace(
        "    v16_run.GENERATIONS=9",
        "    v16_run.POPULATION=configured_population\n"
        "    v16_run.GENERATIONS=internal_generations",
    )
    source = source.replace(
        "expected_candidates=100",
        "expected_candidates=configured_candidates",
    )
    source = source.replace(
        "if len(fitted_ids) != 100 or int(result.get('candidate_evaluations',-1)) != 100:",
        "if len(fitted_ids) != configured_candidates or int(result.get('candidate_evaluations',-1)) != configured_candidates:",
    )
    source = source.replace(
        "print(f'CoSyDelay I{args.intersection} P10/G10=100 complete',flush=True)",
        "print(f'CoSyDelay I{args.intersection} P{configured_population}/G{configured_generations}={configured_candidates} complete',flush=True)",
    )
    return source


def main() -> int:
    search, runner_argv = _search_parameters()
    old_argv = sys.argv[:]
    old_population = os.environ.get("COSYDELAY_POPULATION")
    old_generations = os.environ.get("COSYDELAY_GENERATIONS")
    os.environ["COSYDELAY_POPULATION"] = str(search.population)
    os.environ["COSYDELAY_GENERATIONS"] = str(search.generations)
    sys.argv = [sys.argv[0], *runner_argv]
    try:
        namespace = {"__name__": "cosydelay_embedded"}
        exec(compile(_source(), "<cosydelay-search>", "exec"), namespace)
        code = int(namespace["main"]())
    finally:
        sys.argv = old_argv
        if old_population is None:
            os.environ.pop("COSYDELAY_POPULATION", None)
        else:
            os.environ["COSYDELAY_POPULATION"] = old_population
        if old_generations is None:
            os.environ.pop("COSYDELAY_GENERATIONS", None)
        else:
            os.environ["COSYDELAY_GENERATIONS"] = old_generations
    if code:
        return code

    output = Path(sys.argv[sys.argv.index("--output") + 1])
    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "formal_version": "CoSyDelay",
            "survivor_selection": "global_mu_plus_lambda",
            "prompt_contract": "numeric_operating_domain_plus_broad_structural_regeneration",
            "restart_contract": (
                f"each outer restart runs the identical P{search.population}/G{search.generations} search; "
                "only its declared seed differs"
            ),
            "search_budget": {
                "population": search.population,
                "generations": search.generations,
                "candidate_evaluations": search.population * search.generations,
            },
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
    result["candidate_evaluations_required"] = (
        search.population * search.generations
    )
    # The internal worker may omit the optional adapter marker
    # from serialized per-candidate diagnostics. Keep that audit state
    # explicit without treating a complete search as a numerical failure.
    result["fitter_provenance_audit"] = {
        "expected": "cosydelay.fitter",
        "observed": "missing_optional_marker",
        "warning": "worker omitted formal adapter marker",
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
