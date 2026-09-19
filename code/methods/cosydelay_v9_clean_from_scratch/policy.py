"""Frozen algorithmic policy for the clean formal search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CleanSearchPolicy:
    method_id: str = "cosydelay_v9_clean_from_scratch"
    method_status: str = "prospective_clean_protocol_pending_formal_runs"
    execution_contract_version: str = "clean_baseline_v1"
    evolution_base_seed: int = 20260830
    population: int = 10
    generations: int = 10
    optimizer: str = "L-BFGS-B"
    optimizer_restarts: int = 10
    optimizer_maxiter: int = 200
    optimizer_maxfun: int = 20_000
    approach_workers_cap: int = 4
    fitness_definition: str = (
        "paper_mean_nonnegative_approach_r2_plus_binary_physical_consistency"
    )
    physical_consistency_bonus: float = 1.0
    fitness_lower_bound: float = 0.0
    fitness_upper_bound: float = 2.0
    coefficient_bounds_profile: str = "baseline_pre_v6"
    scale_lower: float = 0.001
    scale_upper: float = 1000.0
    power_exponent_lower: float = 0.05
    power_exponent_upper: float = 5.0
    exp_coefficient_lower: float = 0.0001
    exp_coefficient_upper: float = 1.0
    prompt_knowledge: bool = True
    prompt_style: str = "standard"
    prompt_contract_version: str = "v9_global_flow_inverse_green_v3"
    structural_diversity_mode: str = "canonical"
    residual_guidance_mode: str = "none"
    targeted_physical_feedback: bool = True
    coefficient_robust_r9_prefit_gate: bool = True
    invalid_output_retries_per_generation_batch: int = 5
    generation_batches_per_population_slot: int = 30
    llm_transport_attempts_per_output_attempt: int = 5
    llm_model_requested: str = "gpt-4.1-mini"
    llm_temperature: Optional[float] = None
    llm_top_p: Optional[float] = None
    llm_max_tokens: Optional[int] = None
    llm_sampling_defaults: str = (
        "original_provider_defaults_not_sent_for_temperature_top_p_max_tokens"
    )
    llm_seed_requested: int = 20260712
    initial_population_source: str = "llm_generated_from_scratch"
    external_incumbent_allowed: bool = False
    parent_coefficient_inheritance: bool = True
    post_evolution_cv_reranking: bool = False
    post_evolution_refit: bool = False
    fitted_enhanced_physics_is_hard_gate: bool = True
    reuse_standard_fitted_physics_audit: bool = False
    reuse_prefit_symbolic_endpoint_audit: bool = False
    reject_fitted_structural_families: bool = False
    validation_or_test_used_for_selection: bool = False
    fixed_r9_guard: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def coefficient_bounds(self) -> Dict[str, tuple[float, float]]:
        return {
            "scale": (self.scale_lower, self.scale_upper),
            "power_exponent": (
                self.power_exponent_lower,
                self.power_exponent_upper,
            ),
            "exp_coefficient": (
                self.exp_coefficient_lower,
                self.exp_coefficient_upper,
            ),
        }

    @property
    def llm_sampling(self) -> Dict[str, Any]:
        return {
            "model": self.llm_model_requested,
            "temperature": self.llm_temperature,
            "top_p": self.llm_top_p,
            "max_tokens": self.llm_max_tokens,
            "seed": self.llm_seed_requested,
        }


CLEAN_POLICY = CleanSearchPolicy()
