"""Source manifest for prospective V16 runs."""

from pathlib import Path

from methods.cosydelay_v15_symbolic_r9.source_manifest import V15_SOURCE_FILES
from methods.cosydelay_v15_symbolic_r9.source_manifest import GMINI


HERE = Path(__file__).resolve().parent
V16_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *V15_SOURCE_FILES,
            # Complete local import closure for the formal Training launcher,
            # its inherited adapters, and the frozen Test-only evaluator.
            # V15's manifest predates several package-level imports; list the
            # missing modules here without changing an active V15 run.
            GMINI / "llm" / "__init__.py",
            GMINI / "methods" / "__init__.py",
            GMINI / "methods" / "cosydelay_lbfgsb_r10_parallel3" / "__init__.py",
            GMINI / "methods" / "cosydelay_lbfgsb_r10_parallel3" / "integration.py",
            GMINI / "methods" / "cosydelay_lbfgsb_r10_parallel4" / "__init__.py",
            GMINI / "methods" / "cosydelay_lbfgsb_r10_parallel4" / "integration.py",
            GMINI / "methods" / "cosydelay_lbfgsb_r10_parallel4" / "search_policy.py",
            GMINI / "methods" / "cosydelay_lbfgsb_r10_parallel4_v3" / "__init__.py",
            GMINI / "methods" / "cosydelay_lbfgsb_r10_parallel4_v3" / "policy.py",
            GMINI / "methods" / "cosydelay_lbfgsb_r10_parallel4_v3" / "training_selection.py",
            GMINI / "methods" / "cosydelay_v9_clean_from_scratch" / "__init__.py",
            GMINI / "methods" / "cosydelay_v9_clean_from_scratch" / "operations" / "run04_pythonw_supervisor.py",
            GMINI / "methods" / "cosydelay_v10_accuracy_first_parsimony_alllog" / "__init__.py",
            GMINI / "methods" / "cosydelay_v10_accuracy_first_parsimony_alllog" / "final_prompt.py",
            GMINI / "methods" / "cosydelay_v11_total_response_evolution" / "__init__.py",
            GMINI / "methods" / "cosydelay_v13_unbiased_prompt_tie_break" / "__init__.py",
            GMINI / "methods" / "cosydelay_v13_unbiased_prompt_tie_break" / "prompt.py",
            GMINI / "methods" / "prospective_lbfgsb_jacobian_v1" / "__init__.py",
            GMINI / "methods" / "prospective_lbfgsb_jacobian_v1" / "integration.py",
            GMINI / "methods" / "prospective_optimizer_conditioning_v1" / "__init__.py",
            GMINI / "methods" / "prospective_optimizer_conditioning_v1" / "complexity_prompt.py",
            HERE / "__init__.py",
            HERE / "physics.py",
            HERE / "contract.py",
            HERE / "prompt.py",
            HERE / "policy.py",
            HERE / "integration.py",
            HERE / "fitter.py",
            HERE / "diagnostics.py",
            HERE / "source_manifest.py",
            HERE / "run_training_pilot.py",
            HERE / "launch_authorized_pilot.py",
            HERE / "run_offline_training_smoke.py",
            HERE / "benchmark_persistent_limit_worker.py",
            HERE / "benchmark_maxiter_i1_i2.py",
            HERE / "benchmark_bounds_i1_i2.py",
            HERE / "benchmark_bound_components_i2.py",
            HERE / "run_p10g10_training.py",
            HERE / "launch_p10g10_i1_i6.py",
            HERE / "evaluate_frozen_test.py",
        )
    )
)


def validate_v16_source_manifest() -> None:
    missing = [str(path) for path in V16_SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing V16 source files: " + ", ".join(missing))
    if len(V16_SOURCE_FILES) != len(set(V16_SOURCE_FILES)):
        raise RuntimeError("V16 source manifest contains duplicate paths")
