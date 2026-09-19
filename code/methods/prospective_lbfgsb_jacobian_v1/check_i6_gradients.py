"""Check analytic gradients on all 30 frozen I6 approach blocks."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]
REVIEW_ROOT = GMINI / "reviewer_revision_experiments"
for path in (GMINI, REVIEW_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evolution_matrix import runner  # noqa: E402
from evolution_matrix.tune_i6_validation_only import (  # noqa: E402
    SOURCE,
    build_train_bundle,
    load_selected_source,
    read_json,
)
from methods.cosydelay_lbfgsb_r10_parallel3.parallel_fitter import (  # noqa: E402
    _generate_payloads,
)
from methods.prospective_lbfgsb_jacobian_v1.jacobian_fitter import (  # noqa: E402
    _mse_and_gradient,
    _symbolic_derivative_strings,
)
from optimization_lane import prepare_optimization_context  # noqa: E402


OUTPUT = HERE / "experiments" / "i6_gradient_preflight.json"
STEP_SCALE = 1e-4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    plan = read_json(SOURCE / "plan_manifest.json")
    core = runner._import_core()
    bundle = build_train_bundle(core, plan)
    prepared = prepare_optimization_context(
        bundle["fit"],
        bundle["lanes"],
        bundle["lane_to_approach"],
        6,
    )
    records = []
    for run_id in range(1, 11):
        source = load_selected_source(run_id)
        expression = str(source["expression"])
        seed = int(source["run_seed"]) ^ 0x49A6C3D1
        names, payloads = _generate_payloads(
            universal_expr=expression,
            lanes=bundle["lanes"],
            approach_targets=bundle["fit_targets"],
            prepared=prepared,
            rng=np.random.default_rng(seed),
            n_restarts=10,
            maxiter=200,
            maxfun=20_000,
            coefficient_bounds_override=None,
        )
        derivatives = _symbolic_derivative_strings(
            expression, tuple(names)
        )
        for payload in payloads:
            payload["derivative_expressions"] = derivatives
            values = np.asarray(payload["initials"][0], dtype=float)
            objective, analytic = _mse_and_gradient(values, payload)
            finite = np.zeros_like(values)
            for index, value in enumerate(values):
                step = STEP_SCALE * max(1.0, abs(float(value)))
                left = values.copy()
                right = values.copy()
                left[index] -= step
                right[index] += step
                left_objective, _ = _mse_and_gradient(left, payload)
                right_objective, _ = _mse_and_gradient(right, payload)
                finite[index] = (
                    right_objective - left_objective
                ) / (2.0 * step)
            scale = np.maximum(
                1.0, np.maximum(np.abs(analytic), np.abs(finite))
            )
            relative_error = float(
                np.max(np.abs(analytic - finite) / scale)
            )
            records.append(
                {
                    "run_id": run_id,
                    "approach": payload["approach"],
                    "parameters": len(values),
                    "objective": objective,
                    "max_normalized_centered_difference_error": (
                        relative_error
                    ),
                    "pass_at_1e_5": relative_error <= 1e-5,
                }
            )
            print(
                f"run {run_id:02d} {payload['approach']}: "
                f"error={relative_error:.3g}",
                flush=True,
            )
    result = {
        "schema_version": 1,
        "experiment": "i6_analytic_gradient_preflight",
        "approach_blocks": len(records),
        "passes": sum(item["pass_at_1e_5"] for item in records),
        "all_pass": all(item["pass_at_1e_5"] for item in records),
        "threshold": 1e-5,
        "centered_difference_step_scale": STEP_SCALE,
        "max_normalized_centered_difference_error": max(
            item["max_normalized_centered_difference_error"]
            for item in records
        ),
        "records": records,
        "code_sha256": {
            "jacobian_fitter.py": _sha256(HERE / "jacobian_fitter.py"),
            "check_i6_gradients.py": _sha256(Path(__file__).resolve()),
            "method_config.json": _sha256(HERE / "method_config.json"),
            "full10_protocol.md": _sha256(HERE / "FULL10_PROTOCOL.md"),
        },
        "test_file_opened": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(OUTPUT)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
