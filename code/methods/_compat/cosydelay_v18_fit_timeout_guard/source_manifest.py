"""Complete source manifest for frozen V18 runs."""

from pathlib import Path

from methods.cosydelay_v17_cross_batch_retry.source_manifest import (
    GMINI,
    V17_SOURCE_FILES,
)


HERE = Path(__file__).resolve().parent
V18_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *V17_SOURCE_FILES,
            HERE / "__init__.py",
            HERE / "policy.py",
            HERE / "contract.py",
            HERE / "fit_timeout.py",
            HERE / "prefit_r7.py",
            HERE / "integration.py",
            HERE / "source_manifest.py",
            HERE / "run_p10g10_training.py",
            HERE / "launch_p10g10_i1_i6.py",
            HERE / "evaluate_frozen_test.py",
            HERE / "launch_formal_and_test.py",
            HERE / "run_real_fit_smoke.py",
            HERE / "run_offline_integration_smoke.py",
            HERE / "test_v18.py",
            HERE / "BOUNDING_AUDIT.md",
            HERE / "README.md",
        )
    )
)


def validate_v18_source_manifest() -> None:
    missing = [str(path) for path in V18_SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing V18 source files: " + ", ".join(missing))
    if len(V18_SOURCE_FILES) != len(set(V18_SOURCE_FILES)):
        raise RuntimeError("V18 source manifest contains duplicate paths")
