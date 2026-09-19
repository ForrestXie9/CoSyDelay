"""Authoritative source manifest for the frozen clean implementation."""

from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]

# Include every local module imported on the formal execution path, including
# modules that provide optimizer constants or helper functions. Third-party
# code is frozen separately through package versions.
SOURCE_FILES = (
    HERE / "policy.py",
    HERE / "integration.py",
    HERE / "metrics.py",
    HERE / "clean_fitter.py",
    HERE / "diagnostics.py",
    HERE / "source_manifest.py",
    HERE / "run_formal_training_search.py",
    HERE / "launch_formal_i1_i6.py",
    HERE / "method_config.json",
    GMINI / "constants.py",
    GMINI / "data_processing.py",
    GMINI / "expression_adaptation_lane.py",
    GMINI / "expression_rules.py",
    GMINI / "expression_validation_lane.py",
    GMINI / "llm_config.py",
    GMINI / "llm_integration.py",
    GMINI / "llm" / "api_general.py",
    GMINI / "llm" / "interface_LLM.py",
    GMINI / "optimization_lane.py",
    GMINI / "population_evolution_lane.py",
    GMINI / "residual_guidance_lane.py",
    GMINI / "methods" / "cosydelay_lbfgsb_r10_parallel4" / "prefit_gate.py",
    GMINI
    / "methods"
    / "cosydelay_lbfgsb_r10_parallel4"
    / "prefit_integration.py",
    GMINI
    / "methods"
    / "cosydelay_lbfgsb_r10_parallel4"
    / "parallel_fitter.py",
    GMINI
    / "methods"
    / "cosydelay_lbfgsb_r10_parallel3"
    / "parallel_fitter.py",
    GMINI
    / "methods"
    / "cosydelay_lbfgsb_r10_parallel4_v3"
    / "fitter.py",
    GMINI
    / "methods"
    / "cosydelay_lbfgsb_r10_parallel4_v3"
    / "physics_audit.py",
    GMINI
    / "methods"
    / "prospective_lbfgsb_jacobian_v1"
    / "jacobian_fitter.py",
)


def validate_source_manifest() -> None:
    missing = [str(path) for path in SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen source files: " + ", ".join(missing))
    resolved = [path.resolve() for path in SOURCE_FILES]
    if len(resolved) != len(set(resolved)):
        raise RuntimeError("frozen source manifest contains duplicate paths")
