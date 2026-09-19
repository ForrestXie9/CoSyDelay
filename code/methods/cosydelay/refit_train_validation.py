"""Refit frozen CoSyDelay structures on Training+Validation and then evaluate Test."""

from __future__ import annotations

from pathlib import Path


def _source() -> str:
    path = (
        Path(__file__).resolve().parents[2]
        / "reviewer_revision_experiments"
        / "08_strict_three_way_comparison"
        / "refit_frozen_cosydelay_on_train_validation.py"
    )
    source = path.read_text(encoding="utf-8")
    old = "CoSyDelay-V21-20restart-validation-refit"
    if old not in source:
        raise RuntimeError("CoSyDelay refit runner insertion point missing")
    source = source.replace(old, "CoSyDelay-uniform-restarts-validation-refit")
    old_import = (
        "from methods.cosydelay_v19_accelerated_equivalent.fitter import (\n"
        "    PersistentAcceleratedEquivalentFitter,\n"
        ")"
    )
    new_import = (
        "from methods.cosydelay.fitter import (\n"
        "    StableRangeAcceleratedFitter as PersistentAcceleratedEquivalentFitter,\n"
        ")"
    )
    if old_import not in source:
        raise RuntimeError("CoSyDelay refit fitter import insertion point missing")
    return source.replace(old_import, new_import, 1)


def main() -> int:
    namespace = {
        "__name__": "cosydelay_refit_embedded",
        "__file__": str(
            Path(__file__).resolve().parents[2]
            / "reviewer_revision_experiments"
            / "08_strict_three_way_comparison"
            / "refit_frozen_cosydelay_on_train_validation.py"
        ),
    }
    exec(compile(_source(), "<cosydelay-refit>", "exec"), namespace)
    return int(namespace["main"]())


if __name__ == "__main__":
    raise SystemExit(main())
