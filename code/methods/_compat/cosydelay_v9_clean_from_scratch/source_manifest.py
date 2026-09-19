"""Source manifest for the packaged CoSyDelay runtime.

The research workspace used a fixed absolute-path manifest. The release is
relocatable, so the manifest is derived from the Python modules shipped under
code/methods.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]  # code/methods


def _packaged_python_files() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in GMINI.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    )


SOURCE_FILES = _packaged_python_files()


def validate_source_manifest() -> None:
    missing = [str(path) for path in SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing packaged source files: " + ", ".join(missing))
    if len(SOURCE_FILES) != len(set(SOURCE_FILES)):
        raise RuntimeError("packaged source manifest contains duplicate paths")
