"""Complete source manifest for frozen V17 runs."""

from pathlib import Path

from methods.cosydelay_v16_manuscript_principlewise.source_manifest import (
    GMINI,
    V16_SOURCE_FILES,
)


HERE = Path(__file__).resolve().parent
V17_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *V16_SOURCE_FILES,
            HERE / "__init__.py",
            HERE / "policy.py",
            HERE / "prompt.py",
            HERE / "contract.py",
            HERE / "integration.py",
            HERE / "source_manifest.py",
            HERE / "run_p10g10_training.py",
            HERE / "launch_p10g10_i1_i6.py",
            HERE / "evaluate_frozen_test.py",
            HERE / "launch_formal_and_test.py",
            HERE / "test_v17.py",
            HERE / "README.md",
        )
    )
)


def validate_v17_source_manifest() -> None:
    missing = [str(path) for path in V17_SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing V17 source files: " + ", ".join(missing))
    if len(V17_SOURCE_FILES) != len(set(V17_SOURCE_FILES)):
        raise RuntimeError("V17 source manifest contains duplicate paths")
