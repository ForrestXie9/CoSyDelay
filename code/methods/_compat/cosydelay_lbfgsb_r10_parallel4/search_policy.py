"""Training-only adaptive search policy for the retained P4/G2 pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class AdaptiveSearchPolicy:
    policy_id: str = "p4g2_feasible_archive_training_stop_v1"
    retention_status: str = "official_after_training_only_repeated_pilot"
    population: int = 4
    generations: int = 2
    feasible_archive: bool = True
    early_stop_feasible_count: int = 2
    early_stop_train_r2_gap: float = 0.02
    targeted_physical_feedback: bool = True
    selection_source: str = "training"
    validation_accessed_during_evolution: bool = False
    fixed_r9_guard: bool = False

    def evolution_kwargs(self) -> Dict[str, Any]:
        return {
            "use_feasible_archive": self.feasible_archive,
            "early_stop_feasible_count": self.early_stop_feasible_count,
            "early_stop_train_gap": self.early_stop_train_r2_gap,
            "targeted_physical_feedback": self.targeted_physical_feedback,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


P4G2_ADAPTIVE_SEARCH = AdaptiveSearchPolicy()
