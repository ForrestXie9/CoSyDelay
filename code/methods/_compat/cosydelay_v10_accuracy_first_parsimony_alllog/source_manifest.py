"""Authoritative source manifest for prompt-v6 efficiency-v1 formal runs."""

from __future__ import annotations

from pathlib import Path

from methods.cosydelay_v9_clean_from_scratch.source_manifest import (
    SOURCE_FILES as V9_SOURCE_FILES,
)


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]

V10_SOURCE_FILES = (
    HERE / "policy.py",
    HERE / "integration.py",
    HERE / "fitter.py",
    HERE / "final_prompt.py",
    HERE / "run_final_prompt_pilot.py",
    HERE / "run_formal_training_search.py",
    HERE / "launch_formal_i1_i6.py",
    HERE / "source_manifest.py",
    GMINI
    / "methods"
    / "prospective_optimizer_conditioning_v1"
    / "all_log_fitter.py",
    GMINI
    / "methods"
    / "prospective_optimizer_conditioning_v1"
    / "complexity_prompt.py",
)

SOURCE_FILES = tuple(dict.fromkeys((*V9_SOURCE_FILES, *V10_SOURCE_FILES)))


def validate_source_manifest() -> None:
    missing = [str(path) for path in SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing V10 source files: " + ", ".join(missing))
    resolved = [path.resolve() for path in SOURCE_FILES]
    if len(resolved) != len(set(resolved)):
        raise RuntimeError("V10 source manifest contains duplicate paths")
