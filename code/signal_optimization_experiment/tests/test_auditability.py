import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import pandas as pd

from experiment.config import LANES, MODEL_PATH
from experiment.controllers import SymbolicModel, require_formal_matrix_model
from experiment.demand import generate_route_file
from experiment.simulation import (
    ActuatedPhaseTimer,
    _trip_metrics,
    actuated_sumo_duration,
    append_result,
    build_run_identity,
)
from run_experiment import _load_existing


ROOT = Path(__file__).resolve().parents[1]


def _write_symbolic_model(path: Path, *, status="frozen", omit_lane=None, formal=False):
    expression = "Cycle_Time*a1*flow_lane/GR_phase"
    parameters = {
        lane: {"a1": 1.0}
        for lane in LANES
        if lane != omit_lane
    }
    core = {"expression": expression, "lane_parameters": parameters}
    checksum = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "status": status,
        "model_sha256": checksum,
        **core,
    }
    if formal:
        payload.update({
            "selection_protocol": {
                "source_experiment": "physics_score",
                "eligible_variant": "principlewise",
                "eligible_intersection": 1,
                "test_used_for_selection": False,
                "test_opened_after_winner_fixed": True,
            },
            "evolution": {
                "intersection_id": 1,
                "variant": "principlewise",
                "score_mode": "principlewise",
                "generations": 10,
                "population": 20,
                "independent_runs": 10,
                "optimizer_restarts": 3,
            },
            "validation": {
                "matrix_complete": True,
                "source_artifact_hashes_verified": True,
                "history_complete": True,
                "current_principlewise_physical_recheck": {"joint_pass": True},
            },
            "provenance": {
                "source_plan_sha256": "a" * 64,
                "source_config_sha256": "b" * 64,
                "source_code_sha256": {"expression_validation_lane.py": "c" * 64},
            },
        })
    path.write_text(json.dumps(payload), encoding="utf-8")


class FrozenArtifactTests(unittest.TestCase):
    def test_frozen_model_validates_schema_lane_set_coefficients_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.json"
            _write_symbolic_model(valid)
            model = SymbolicModel.load(valid)
            self.assertEqual(set(model.lane_parameters), set(LANES))
            self.assertEqual(len(model.model_sha256), 64)

            fallback = root / "fallback.json"
            _write_symbolic_model(fallback, status="fallback_not_for_formal_control")
            with self.assertRaisesRegex(ValueError, "fallback/unfrozen model rejected"):
                SymbolicModel.load(fallback)

            missing_lane = root / "missing_lane.json"
            _write_symbolic_model(missing_lane, omit_lane="S_L")
            with self.assertRaisesRegex(ValueError, "lane set mismatch"):
                SymbolicModel.load(missing_lane)

            tampered = json.loads(valid.read_text(encoding="utf-8"))
            tampered["lane_parameters"]["S_L"]["a1"] = 2.0
            valid.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                SymbolicModel.load(valid)

    def test_default_formal_model_is_frozen(self):
        model = SymbolicModel.load(MODEL_PATH)
        self.assertEqual(model.status, "frozen")

    def test_formal_entrypoint_rejects_legacy_and_accepts_matrix_provenance(self):
        with self.assertRaisesRegex(ValueError, "legacy/partial model rejected"):
            require_formal_matrix_model(MODEL_PATH)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formal.json"
            _write_symbolic_model(path, formal=True)
            model = require_formal_matrix_model(path)
            self.assertEqual(model.status, "frozen")

    def test_fallback_preparer_cannot_overwrite_formal_path(self):
        before = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "prepare_model.py"),
                "--output",
                str(MODEL_PATH),
            ],
            text=True,
            capture_output=True,
            env=environment,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("forbidden from overwriting", result.stderr)
        after = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        self.assertEqual(before, after)


class ActuatedTimerTests(unittest.TestCase):
    def test_minimum_green_is_five_not_seven_and_gap_uses_new_passages(self):
        timer = ActuatedPhaseTimer(minimum_green=5.0, passage_gap=2.0, maximum_green=60.0)
        timer.begin(0.0)
        self.assertEqual(timer.desired_remaining(0.0), 5.0)
        self.assertEqual(timer.desired_remaining(4.0), 1.0)
        self.assertEqual(timer.desired_remaining(5.0), 0.0)
        self.assertEqual(actuated_sumo_duration(timer.desired_remaining(0.0)), 4.0)

        timer.begin(0.0)
        self.assertEqual(timer.observe(4.0, {"veh-1"}), {"veh-1"})
        self.assertEqual(timer.desired_remaining(4.0), 2.0)
        # Persistent presence is not a new passage and cannot keep resetting gap.
        self.assertEqual(timer.observe(5.0, {"veh-1"}), set())
        self.assertEqual(timer.desired_remaining(5.0), 1.0)
        self.assertEqual(timer.desired_remaining(6.0), 0.0)

        self.assertEqual(timer.observe(5.0, {"veh-1", "veh-2"}), {"veh-2"})
        self.assertEqual(timer.desired_remaining(5.0), 2.0)
        self.assertEqual(timer.desired_remaining(7.0), 0.0)

    def test_passage_gap_never_exceeds_maximum_green(self):
        timer = ActuatedPhaseTimer(minimum_green=5.0, passage_gap=2.0, maximum_green=60.0)
        timer.begin(0.0)
        timer.observe(59.5, {"late-vehicle"})
        self.assertEqual(timer.desired_remaining(59.5), 0.5)
        self.assertEqual(timer.desired_remaining(60.0), 0.0)


class RunIdentityAndMetricsTests(unittest.TestCase):
    def test_content_addressed_identity_and_unique_raw_name(self):
        with tempfile.TemporaryDirectory() as directory:
            route = Path(directory) / "route.rou.xml"
            generate_route_file(route, "balanced", seed=17, duration_s=60)
            first = build_run_identity(
                "symbolic", "balanced", 17, 60, 30, route, clearance_limit_s=120
            )
            second = build_run_identity(
                "symbolic", "balanced", 17, 60, 31, route, clearance_limit_s=120
            )
            repeated = build_run_identity(
                "symbolic", "balanced", 17, 60, 30, route, clearance_limit_s=120
            )
            self.assertEqual(first["run_sha256"], repeated["run_sha256"])
            self.assertNotEqual(first["run_sha256"], second["run_sha256"])
            self.assertIn("symbolic", first["tripinfo_filename"])
            self.assertIn("s17_d60_u30", first["tripinfo_filename"])
            self.assertIn(first["symbolic_model_sha256"][:12], first["tripinfo_filename"])
            self.assertEqual(first["route_sha256"], hashlib.sha256(route.read_bytes()).hexdigest())

    def test_trip_metrics_include_depart_delay_and_demand_window_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trip.xml"
            root = ET.Element("tripinfos")
            ET.SubElement(root, "tripinfo", {
                "id": "a", "depart": "2", "departDelay": "3", "arrival": "20",
                "timeLoss": "5", "waitingTime": "4", "duration": "18",
            })
            ET.SubElement(root, "tripinfo", {
                "id": "b", "depart": "12", "departDelay": "1", "arrival": "-1",
                "timeLoss": "7", "waitingTime": "6", "duration": "10",
            })
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            metrics = _trip_metrics(path, demand_end_s=10)
            self.assertEqual(metrics["tripinfo_vehicles"], 2)
            self.assertEqual(metrics["completed"], 1)
            self.assertEqual(metrics["departed_by_demand_end"], 1)
            self.assertEqual(metrics["completed_by_demand_end"], 0)
            self.assertEqual(metrics["avg_depart_delay"], 2.0)
            self.assertEqual(metrics["avg_time_loss"], 6.0)
            self.assertEqual(metrics["avg_total_delay"], 8.0)

    def test_append_result_rejects_duplicate_key_and_schema_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            first = {"run_sha256": "run-1", "config_sha256": "config-1", "value": 1}
            append_result(path, first, key_fields=("run_sha256",))
            with self.assertRaises(FileExistsError):
                append_result(path, first, key_fields=("run_sha256",))
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                append_result(
                    path,
                    {**first, "extra": 2},
                    key_fields=("run_sha256",),
                )

    def test_runner_rejects_same_logical_key_with_multiple_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            base = {
                "scenario": "balanced",
                "controller": "symbolic",
                "seed": 1,
                "duration_s": 60,
                "update_interval_s": 30,
                "clearance_limit_s": 120,
            }
            pd.DataFrame([
                {**base, "config_sha256": "config-a", "run_sha256": "run-a"},
                {**base, "config_sha256": "config-b", "run_sha256": "run-b"},
            ]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate logical run key"):
                _load_existing(path)


class AnalyzerAuditTests(unittest.TestCase):
    def test_analyzer_excludes_unfinished_and_duplicates_without_pivot_failure(self):
        rows = []
        for seed in (1, 2, 3):
            for controller, delay in (("symbolic", 10.0), ("webster", 12.0)):
                fully = seed != 2
                row = {
                    "scenario": "balanced",
                    "controller": controller,
                    "seed": seed,
                    "duration_s": 60,
                    "update_interval_s": 30,
                    "clearance_limit_s": 120,
                    "route_sha256": f"route-{seed}",
                    "run_sha256": f"run-{seed}-{controller}",
                    "config_sha256": f"config-{seed}-{controller}",
                    "scheduled_vehicles": 20,
                    "tripinfo_vehicles": 20 if fully else 19,
                    "completed": 20 if fully else 18,
                    "route_entry_rate": 1.0 if fully else 0.95,
                    "fully_observed": fully,
                    "clearance_complete": fully,
                    "residual_queue": 0 if fully else 2,
                    "residual_expected_vehicles": 0 if fully else 2,
                    "avg_total_delay": delay,
                    "avg_time_loss": delay - 1,
                    "avg_depart_delay": 1.0,
                    "clearance_seconds": 30.0,
                }
                rows.append(row)
                if seed == 3 and controller == "webster":
                    rows.append(dict(row))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "results.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / "analyze_results.py"), str(source)],
                text=True,
                capture_output=True,
                env=environment,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            eligible = pd.read_csv(root / "results_fully_observed.csv")
            self.assertEqual(set(eligible["seed"]), {1})
            self.assertEqual(len(eligible), 2)
            excluded = pd.read_csv(root / "results_unfinished_or_invalid.csv")
            self.assertGreaterEqual(len(excluded), 5)
            duplicates = pd.read_csv(root / "results_duplicate_runs.csv")
            self.assertEqual(len(duplicates), 2)
            paired = pd.read_csv(root / "results_paired_comparisons.csv")
            self.assertEqual(int(paired.iloc[0]["n_complete_pairs"]), 1)


if __name__ == "__main__":
    unittest.main()
