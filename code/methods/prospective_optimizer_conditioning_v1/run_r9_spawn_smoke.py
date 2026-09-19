"""Windows-spawn smoke test for the symbolic R9 worker."""

from __future__ import annotations

from methods.cosydelay_lbfgsb_r10_parallel4_v3.physics_audit import (
    audit_search_physics,
)


EXPRESSION = (
    "Cycle_Time * flow_lane / GR_phase * (a1 * (1 + a2 * flow_lane) / "
    "(1 + a3 * GR_phase) + a4 * log(1 + a5 * flow_lane) / "
    "(1 + a6 * GR_phase**2))"
)


def main() -> int:
    result = audit_search_physics(EXPRESSION, standard_joint_pass=True)
    if not result["symbolic_r8_exact"]:
        raise RuntimeError(result)
    if not result["symbolic_r9_positive_infinity"]:
        raise RuntimeError(result)
    print(
        f"R8={result['symbolic_r8_exact']} "
        f"R9={result['symbolic_r9_positive_infinity']} "
        f"method={result['symbolic_r9_method']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
