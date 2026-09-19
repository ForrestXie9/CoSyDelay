"""Complete local-source manifest for V14 reproducibility records."""

from pathlib import Path

from methods.cosydelay_v9_clean_from_scratch.source_manifest import (
    SOURCE_FILES as V9_SOURCE_FILES,
)


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]

V14_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *V9_SOURCE_FILES,
            GMINI / "population_evolution_lane.py",
            GMINI / "methods" / "cosydelay_v10_accuracy_first_parsimony_alllog" / "policy.py",
            GMINI / "methods" / "cosydelay_v10_accuracy_first_parsimony_alllog" / "integration.py",
            GMINI / "methods" / "cosydelay_v10_accuracy_first_parsimony_alllog" / "fitter.py",
            GMINI / "methods" / "cosydelay_v10_accuracy_first_parsimony_alllog" / "run_final_prompt_pilot.py",
            GMINI / "methods" / "cosydelay_v11_total_response_evolution" / "policy.py",
            GMINI / "methods" / "cosydelay_v11_total_response_evolution" / "integration.py",
            GMINI / "methods" / "cosydelay_v11_total_response_evolution" / "final_prompt.py",
            GMINI / "methods" / "cosydelay_v13_unbiased_prompt_tie_break" / "policy.py",
            GMINI / "methods" / "cosydelay_v13_unbiased_prompt_tie_break" / "ranking.py",
            GMINI / "methods" / "cosydelay_v13_unbiased_prompt_tie_break" / "integration.py",
            HERE / "__init__.py",
            HERE / "policy.py",
            HERE / "prompt.py",
            HERE / "ranking.py",
            HERE / "integration.py",
            HERE / "source_manifest.py",
            HERE / "run_training_pilot.py",
            HERE / "launch_authorized_pilot.py",
        )
    )
)


def validate_v14_source_manifest() -> None:
    missing = [str(path) for path in V14_SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing V14 source files: " + ", ".join(missing))
