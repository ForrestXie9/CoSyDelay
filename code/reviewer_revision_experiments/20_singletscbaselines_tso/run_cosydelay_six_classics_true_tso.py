"""True SUMO replay for CoSyDelay and six classical delay models.

The six classical models are the same Table-VII analytic families used in the
project: Webster/HCM/Akcelik, each with a uniform formula and a fitted
``variant1`` formula.  The fitted variants are trained only on the original
Intersection-1 training data; the SingleTSCBaselines target scenario is used
only for demand extraction and SUMO evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numexpr as ne
import numpy as np
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[2]
GMini = PACKAGE_ROOT / "code"
SIGNAL_EXPERIMENT = GMini / "signal_optimization_experiment"
RANKING_DIR = GMini / "reviewer_revision_experiments" / "16_offline_timing_plan_ranking"
GENERALIZATION_DIR = GMini / "reviewer_revision_experiments" / "17_generalization_support"
for path in (GMini, SIGNAL_EXPERIMENT, RANKING_DIR, GENERALIZATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment.controllers import (  # noqa: E402
    SymbolicModel,
    require_formal_matrix_model,
    symbolic_plan,
    webster_plan,
)
from optimization_lane import (  # noqa: E402
    MAX_DELAY,
)
from run_cosydelay_true_tso import (  # noqa: E402
    Scenario,
    _run_controller,
    _sha256,
    load_scenario,
)
from constants import SAT_FLOW  # noqa: E402
from generalization_core import TABLE_VII_MODELS, fit_table_vii_model  # noqa: E402
from ranking_core import load_train_test  # noqa: E402


LANES = (
    "S_R", "S_T", "S_L", "N_R", "N_T", "N_L",
    "E_R", "E_T", "E_L", "W_R", "W_T", "W_L",
)
PHASE_BY_LANE = {
    "S_R": "A", "N_R": "A",
    "S_T": "B", "S_L": "B", "N_T": "B", "N_L": "B",
    "E_R": "C", "W_R": "C",
    "E_T": "D", "E_L": "D", "W_T": "D", "W_L": "D",
}
PHASES = ("A", "B", "C", "D")
MIN_GREEN = 5.0
MAX_GREEN = 60.0
LOST_TIME = 24.0
MIN_CYCLE = 60.0
MAX_CYCLE = 180.0


@dataclass(frozen=True)
class DelaySpec:
    name: str
    expression: str
    lane_parameters: Dict[str, Dict[str, float]]


def _load_specs() -> Dict[str, DelaySpec]:
    specs: Dict[str, DelaySpec] = {}
    cosydelay_path = SIGNAL_EXPERIMENT / "models" / "symbolic_lane_model.json"
    cosydelay = require_formal_matrix_model(cosydelay_path)
    specs["cosydelay"] = DelaySpec(
        name="CoSyDelay",
        expression=cosydelay.expression,
        lane_parameters=cosydelay.lane_parameters,
    )

    # Fit only the optimized classical variants on the original I1 training
    # split.  No target SingleTSCBaselines labels are opened here.
    original_train, _ = load_train_test(1)
    for family in ("webster", "hcm", "akcelik"):
        uniform_name = family
        uniform_expr = TABLE_VII_MODELS[(family, "uniform")]
        specs[uniform_name] = DelaySpec(
            name={"webster": "Webster", "hcm": "HCM", "akcelik": "Akcelik"}[family],
            expression=uniform_expr,
            lane_parameters={lane: {} for lane in LANES},
        )
        fitted = fit_table_vii_model(
            1, original_train, family, "variant1", seed=20260913
        )
        specs[f"{family}-optimized"] = DelaySpec(
            name=fitted.name,
            expression=fitted.expression,
            lane_parameters=fitted.lane_parameters,
        )
    return specs


def _evaluate(spec: DelaySpec, lane: str, rate: float, green: float, cycle: float) -> float:
    context = {
        "flow_lane": float(rate) / float(SAT_FLOW),
        "GR_phase": float(green) / max(float(cycle), 1e-9),
        "Cycle_Time": float(cycle),
        **spec.lane_parameters.get(lane, {}),
    }
    try:
        value = float(ne.evaluate(spec.expression, local_dict=context))
    except Exception:
        return MAX_DELAY
    if not np.isfinite(value):
        return MAX_DELAY
    return float(np.clip(value, 0.0, MAX_DELAY))


def _classic_plan(spec: DelaySpec, rates: Dict[str, float]) -> Dict[str, float]:
    initial_dict = webster_plan(rates)
    initial = np.asarray([initial_dict[phase] for phase in PHASES], dtype=float)

    def objective(greens: np.ndarray) -> float:
        cycle = LOST_TIME + float(np.sum(greens))
        total_flow = float(sum(rates.values()))
        if cycle <= 0.0 or total_flow <= 0.0:
            return MAX_DELAY
        weighted = 0.0
        for lane in LANES:
            weighted += rates[lane] * _evaluate(
                spec, lane, rates[lane], greens[PHASES.index(PHASE_BY_LANE[lane])], cycle
            )
        return weighted / total_flow

    constraints = (
        {"type": "ineq", "fun": lambda values: LOST_TIME + np.sum(values) - MIN_CYCLE},
        {"type": "ineq", "fun": lambda values: MAX_CYCLE - LOST_TIME - np.sum(values)},
    )
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(MIN_GREEN, MAX_GREEN)] * len(PHASES),
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-8},
    )
    values = result.x if result.success and np.all(np.isfinite(result.x)) else initial
    return {phase: float(values[index]) for index, phase in enumerate(PHASES)}


def _plan(spec_key: str, spec: DelaySpec, scenario: Scenario, cosydelay_model: SymbolicModel) -> Dict[str, float]:
    if spec_key == "cosydelay":
        return symbolic_plan(scenario.rates, cosydelay_model)
    return _classic_plan(spec, scenario.rates)


def run(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.junction, args.env_name)
    specs = _load_specs()
    cosydelay_model = require_formal_matrix_model(SIGNAL_EXPERIMENT / "models" / "symbolic_lane_model.json")
    unknown = [key for key in args.controllers if key not in specs]
    if unknown:
        raise ValueError(f"Unknown controller keys: {unknown}; available={sorted(specs)}")
    plans = {key: _plan(key, specs[key], scenario, cosydelay_model) for key in args.controllers}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        for key in args.controllers:
            print(
                f"running {scenario.junction}/{scenario.env_name} "
                f"{specs[key].name} seed={seed}", flush=True,
            )
            result = _run_controller(
                scenario, specs[key].name.replace("/", "-"), plans[key], seed,
                args.output_dir / "tripinfo",
            )
            result["model_key"] = key
            rows.append(result)
            print(
                f"  avg_total_delay={result['avg_total_delay_s']:.3f}s "
                f"completion={result['completed_vehicles']}/{result['scheduled_vehicles']}",
                flush=True,
            )

    result_path = args.output_dir / "results.csv"
    fieldnames = sorted({key for row in rows for key in row if key != "applied_plan"}) + ["applied_plan_json"]
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {key: value for key, value in row.items() if key != "applied_plan"}
            record["applied_plan_json"] = json.dumps(row["applied_plan"], sort_keys=True)
            writer.writerow(record)

    model_artifacts = {
        key: {
            "display_name": spec.name,
            "expression": spec.expression,
            "lane_parameters": spec.lane_parameters,
            "plan": plans[key],
        }
        for key, spec in specs.items() if key in args.controllers
    }
    (args.output_dir / "model_artifacts.json").write_text(
        json.dumps(model_artifacts, indent=2), encoding="utf-8"
    )
    metadata = {
        "status": "complete",
        "protocol": "true_sumo_tso_fixed_plan_six_classics",
        "junction": scenario.junction,
        "env_name": scenario.env_name,
        "controllers": args.controllers,
        "seeds": args.seeds,
        "network": str(scenario.network.resolve()),
        "network_sha256": _sha256(scenario.network),
        "route": str(scenario.route.resolve()),
        "route_sha256": _sha256(scenario.route),
        "cosydelay_model": str((SIGNAL_EXPERIMENT / "models" / "symbolic_lane_model.json").resolve()),
        "cosydelay_model_sha256": _sha256(SIGNAL_EXPERIMENT / "models" / "symbolic_lane_model.json"),
        "classical_fit_source": "original Intersection-1 Train split only",
        "tls_id": scenario.tls_id,
        "green_phase_indices": scenario.green_phase_indices,
        "source_green_phase_indices": scenario.source_green_phase_indices,
        "custom_yellow_s": 3.0,
        "custom_clear_s": 3.0,
        "model_lost_time_s": LOST_TIME,
        "rates_vph": scenario.rates,
        "scheduled_vehicles": scenario.scheduled_vehicles,
        "mapping": scenario.lane_map,
        "logical_link_indices": scenario.logical_link_indices,
        "plans": plans,
        "results": str(result_path.resolve()),
        "model_artifacts": str((args.output_dir / "model_artifacts.json").resolve()),
        "test_labels_used_for_selection": False,
        "note": "Pilot/full results are SUMO replay outcomes; model plans are fixed before simulation.",
    }
    manifest_path = args.output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(result_path.resolve())
    print(manifest_path.resolve())
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junction", default="Beijing_Gaojiaoyuan")
    parser.add_argument("--env-name", default="normal_low_density")
    parser.add_argument(
        "--controllers", nargs="+",
        choices=("cosydelay", "webster", "webster-optimized", "hcm", "hcm-optimized", "akcelik", "akcelik-optimized"),
        default=["cosydelay", "webster", "webster-optimized", "hcm", "hcm-optimized", "akcelik", "akcelik-optimized"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
