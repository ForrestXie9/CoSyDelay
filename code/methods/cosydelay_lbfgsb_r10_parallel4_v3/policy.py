"""Predeclared v3 search and training-only selection policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class V3SearchPolicy:
    method_id: str = "cosydelay_lbfgsb_r10_parallel4_v3_traincv_prospective"
    retention_status: str = "prospective_not_official"
    population: int = 4
    generations: int = 2
    optimizer_restarts: int = 10
    approach_workers_cap: int = 4
    inner_folds: int = 4
    search_holdout_fold: int = 0
    cross_validation_top_k: int = 3
    r2_weight: float = 1.0
    normalized_rmse_weight: float = 0.05
    normalized_mae_weight: float = 0.05
    polish_maxiter: int = 200
    polish_maxfun: int = 20_000
    fixed_r9_guard: bool = False
    validation_or_test_used_for_selection: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


V3_POLICY = V3SearchPolicy()
