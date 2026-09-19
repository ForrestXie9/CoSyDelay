"""Source manifest for reproducible V15 runs."""

from pathlib import Path

from methods.cosydelay_v14_physical_feedback_repair.source_manifest import (
    GMINI,
    V14_SOURCE_FILES,
)


HERE = Path(__file__).resolve().parent
V15_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *V14_SOURCE_FILES,
            GMINI / "methods" / "prospective_optimizer_conditioning_v1" / "role_policy.py",
            GMINI / "methods" / "prospective_optimizer_conditioning_v1" / "all_log_fitter.py",
            HERE / "__init__.py",
            HERE / "policy.py",
            HERE / "fitter.py",
            HERE / "integration.py",
            HERE / "source_manifest.py",
            HERE / "run_training_pilot.py",
            HERE / "run_p10g10_training.py",
            HERE / "launch_p10g10_i1_i6.py",
        )
    )
)


def validate_v15_source_manifest() -> None:
    missing = [str(path) for path in V15_SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing V15 source files: " + ", ".join(missing))
