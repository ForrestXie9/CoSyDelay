"""P3/G2 Training-only V15 symbolic-R9 pilot."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from methods.cosydelay.engine.candidate_engine import (
    run_final_prompt_pilot as base,
)
from methods.cosydelay.engine.physics_support.run_training_pilot import (
    _v14_prompt_audit,
)

from .integration import install_v15_candidate
from .policy import V15_POLICY


def _output_argument() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError("missing --output argument") from exc


def main() -> int:
    output = _output_argument()
    previous = (base.install_v10_candidate, base.V10_POLICY, base._prompt_audit)
    base.install_v10_candidate = install_v15_candidate
    base.V10_POLICY = V15_POLICY
    base._prompt_audit = _v14_prompt_audit
    try:
        code = base.main()
    finally:
        base.install_v10_candidate, base.V10_POLICY, base._prompt_audit = previous
    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "status": "completed_v15_p3g2_training_pilot",
            "method_id": V15_POLICY.method_id,
            "method_status": V15_POLICY.method_status,
            "r9_selection_rule": (
                "coefficient_robust_symbolic_positive_infinity_only"
            ),
            "r9_numerical_probe_role": "diagnostic_only_not_a_gate",
            "optimizer_restart_selection": "minimum_training_mse",
            "paper_fitness_changed": False,
            "validation_or_test_used_for_selection": False,
        }
    )
    base.write_json(result_path, result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
