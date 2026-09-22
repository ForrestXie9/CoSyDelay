"""Run one V15 P10/G10 Training-only search from scratch."""

from __future__ import annotations

import json
from pathlib import Path

from methods.cosydelay.engine.prompt_support import run_p10g10_training as base
from methods.cosydelay.engine.physics_support.run_training_pilot import _v14_prompt_audit
from methods.cosydelay.engine.data_protocol.run_formal_training_search import sha256_file

from .integration import install_v15_candidate
from .policy import V15_POLICY
from .source_manifest import GMINI, V15_SOURCE_FILES, validate_v15_source_manifest


def main() -> int:
    validate_v15_source_manifest()
    previous = (
        base.install_v13_candidate,
        base.V13_POLICY,
        base._v13_prompt_audit,
    )
    base.install_v13_candidate = install_v15_candidate
    base.V13_POLICY = V15_POLICY
    base._v13_prompt_audit = _v14_prompt_audit
    try:
        code = base.main()
    finally:
        (
            base.install_v13_candidate,
            base.V13_POLICY,
            base._v13_prompt_audit,
        ) = previous
    if code:
        return int(code)

    args = base.arguments()
    result_path = Path(args.output).resolve() / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "status": "completed_v15_p10g10_training_only_search",
            "method_id": V15_POLICY.method_id,
            "method_status": "v15_candidate_pending_i1_i6_confirmation",
            "r9_selection_rule": "coefficient_robust_symbolic_positive_infinity_only",
            "r9_numerical_probe_role": "diagnostic_only_not_a_gate",
            "optimizer_restart_selection": "minimum_training_mse",
            "paper_fitness_changed": False,
            "validation_or_test_used_for_selection": False,
            "v15_source_sha256": {
                (
                    str(path.relative_to(GMINI)).replace("\\", "/")
                    if path.is_relative_to(GMINI)
                    else str(path)
                ): sha256_file(path)
                for path in V15_SOURCE_FILES
            },
        }
    )
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
