from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiment.controllers import SymbolicModel
from freeze_evolved_model import (
    APPROACHES,
    HASHED_RUN_ARTIFACTS,
    LANES,
    EXPECTED_RULE_ORDER,
    EXPECTED_RULE_SCHEMA_ID,
    FormalMinimums,
    MatrixAuditError,
    _configuration_hash,
    _sha256_file,
    _split_metrics,
    audit_matrix,
    freeze_from_matrix,
    select_intersection1_principlewise,
)


SMALL_MINIMUMS = FormalMinimums(
    generations=1,
    population=1,
    runs_per_cell=2,
    optimizer_restarts=1,
    required_intersections=(1,),
)
EXPRESSION = "Cycle_Time * a1 * flow_lane / GR_phase"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _dialogue_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class MatrixFixture:
    def __init__(self, root: Path):
        self.root = root
        self.matrix = root / "matrix"
        self.data = root / "data"
        self.output = root / "symbolic_lane_model.json"
        self.run_dirs = {}
        self._build()

    def _build(self) -> None:
        self.matrix.mkdir(parents=True)
        self.data.mkdir(parents=True)
        train_path = self.data / "Intersection_1_Train.jsonl"
        test_path = self.data / "Intersection_1_Test.jsonl"
        train_path.write_text(
            "".join(json.dumps({"dialogue": f"train-{index}"}) + "\n" for index in range(4)),
            encoding="utf-8",
        )
        test_path.write_text(
            "".join(json.dumps({"dialogue": f"test-{index}"}) + "\n" for index in range(2)),
            encoding="utf-8",
        )

        split_dir = self.matrix / "_manifests" / "splits"
        train_rows = []
        train_compact = []
        for index, split in enumerate(("fit", "fit", "validation", "validation")):
            compact = {
                "source_index": index,
                "sample_id": f"train-{index}",
                "dialogue_sha256": _dialogue_hash(f"train-{index}"),
                "split": split,
            }
            train_compact.append(compact)
            train_rows.append({
                "intersection_id": 1,
                **compact,
                "source_line": index + 1,
                "split_seed": 7,
                "validation_fraction": 0.2,
            })
        test_rows = []
        test_compact = []
        for index in range(2):
            compact = {
                "source_index": index,
                "sample_id": f"test-{index}",
                "dialogue_sha256": _dialogue_hash(f"test-{index}"),
                "split": "test",
            }
            test_compact.append(compact)
            test_rows.append({"intersection_id": 1, **compact, "source_line": index + 1})
        train_manifest = split_dir / "intersection_01_train_split.csv"
        test_manifest = split_dir / "intersection_01_test_samples.csv"
        _write_csv(train_manifest, train_rows, list(train_rows[0]))
        _write_csv(test_manifest, test_rows, list(test_rows[0]))

        data_info = {
            "data_dir": str(self.data.resolve()),
            "train_file": train_path.name,
            "test_file": test_path.name,
            "train_file_sha256": _sha256_file(train_path),
            "test_file_sha256": _sha256_file(test_path),
            "train_manifest": str(train_manifest.relative_to(self.matrix)).replace("\\", "/"),
            "test_manifest": str(test_manifest.relative_to(self.matrix)).replace("\\", "/"),
            "train_manifest_sha256": _configuration_hash(train_compact),
            "test_manifest_sha256": _configuration_hash(test_compact),
            "sample_counts": {"fit": 2, "validation": 2, "test": 2},
        }
        variants = [
            {
                "variant": "binary",
                "method": "cosydelay_binary",
                "score_mode": "binary",
                "prompt_knowledge": True,
            },
            {
                "variant": "principlewise",
                "method": "cosydelay_principlewise",
                "score_mode": "principlewise",
                "prompt_knowledge": True,
            },
        ]
        physical_verifier = {
            "rule_schema_id": EXPECTED_RULE_SCHEMA_ID,
            "rule_order": list(EXPECTED_RULE_ORDER),
            "config": {},
        }
        plan = {
            "schema_version": 3,
            "runner_version": "test",
            "execution_status": "complete",
            "plan_only_is_not_an_experiment_run": False,
            "experiment": "physics_score",
            "intersections": [1],
            "variants": variants,
            "generations": 1,
            "population": 1,
            "runs_per_cell": 2,
            "base_seed": 11,
            "split_seed": 7,
            "optimizer_restarts": 1,
            "physics_weight": 1.0,
            "validation_fraction": 0.2,
            "physical_verifier": physical_verifier,
            "data": {"1": data_info},
            "run_count": 4,
            "runs": [],
        }
        for variant in variants:
            for run_id in (1, 2):
                config = {
                    "schema_version": 3,
                    "runner_version": "test",
                    "experiment": "physics_score",
                    "variant": variant["variant"],
                    "method": variant["method"],
                    "score_mode": variant["score_mode"],
                    "prompt_knowledge": True,
                    "prompt_style": "standard",
                    "llm_temperature": 0.0,
                    "intersection_id": 1,
                    "run_id": run_id,
                    "generations": 1,
                    "population": 1,
                    "runs_in_matrix": 2,
                    "base_seed": 11,
                    "run_seed": 100 + run_id,
                    "split_seed": 7,
                    "validation_fraction": 0.2,
                    "optimizer_restarts": 1,
                    "physics_weight": 1.0,
                    "selection_split": "validation",
                    "test_used_for_selection": False,
                    "universal_features": ["flow_lane", "GR_phase", "Cycle_Time"],
                    "physical_verifier": physical_verifier,
                    "data": data_info,
                    "code_sha256": {
                        "expression_validation_lane.py": "a" * 64,
                        "review_runner.py": "b" * 64,
                    },
                }
                config_hash = _configuration_hash(config)
                relative = (
                    Path("physics_score")
                    / variant["variant"]
                    / "intersection_01"
                    / f"run_{run_id:03d}"
                )
                run_dir = self.matrix / relative
                self.run_dirs[(variant["variant"], run_id)] = run_dir
                self._write_run(run_dir, config, config_hash)
                plan["runs"].append({
                    "run_dir": str(relative).replace("\\", "/"),
                    "config_hash": config_hash,
                    "config": config,
                })
        _write_json(self.matrix / "plan_manifest.json", plan)

    def _prediction_rows(self, config, split: str):
        # Run 1 is the validation winner but has deliberately worse test values.
        if split == "validation":
            offset = 0.5 if config["run_id"] == 1 else 2.0
        elif split == "test":
            offset = 5.0 if config["run_id"] == 1 else 0.1
        else:
            offset = 0.0
        rows = []
        for approach_index, approach in enumerate(APPROACHES):
            for sample_index, truth in enumerate((10.0 + approach_index, 20.0 + approach_index)):
                rows.append({
                    "method": config["method"],
                    "run_id": config["run_id"],
                    "intersection_id": 1,
                    "approach": approach,
                    "sample_id": f"{split}-{sample_index}",
                    "y_true": truth,
                    "y_pred": truth + offset,
                    "split": split,
                })
        return rows

    def _write_run(self, run_dir: Path, config, config_hash: str) -> None:
        run_dir.mkdir(parents=True)
        _write_json(run_dir / "config.json", {**config, "config_hash": config_hash})
        (run_dir / "config.sha256").write_text(config_hash + "\n", encoding="utf-8")
        history = [
            {
                "generation": generation,
                "individual": 1,
                "event": "evaluated",
                "expression": EXPRESSION,
                "physical_joint_pass": True,
                "score_mode": config["score_mode"],
            }
            for generation in (0, 1)
        ]
        _write_json(run_dir / "history.json", history)
        model = {
            "method": config["method"],
            "experiment": "physics_score",
            "variant": config["variant"],
            "intersection_id": 1,
            "run_id": config["run_id"],
            "universal_expression": {
                "template": EXPRESSION,
                "thought": "test",
                "explanation": "test",
            },
            "lane_parameters": {lane: {"a1": 1.0} for lane in LANES},
            "selection": {
                "score_mode": config["score_mode"],
                "prompt_knowledge": True,
                "physical_verifier": config["physical_verifier"],
                "selection_split": "validation",
                "test_used_for_selection": False,
            },
        }
        _write_json(run_dir / "model.json", model)
        validation = self._prediction_rows(config, "validation")
        test = self._prediction_rows(config, "test")
        fields = [
            "method", "run_id", "intersection_id", "approach",
            "sample_id", "y_true", "y_pred", "split",
        ]
        _write_csv(run_dir / "validation_predictions_long.csv", validation, fields)
        _write_csv(run_dir / "test_predictions_long.csv", test, fields)
        _write_csv(run_dir / "predictions_long.csv", validation + test, fields)
        _write_json(run_dir / "metrics.json", {
            "fit": _split_metrics(validation),
            "validation": _split_metrics(validation),
            "test": _split_metrics(test),
            "selection_statement": {
                "fit_used_for_coefficient_fitting": True,
                "validation_used_for_expression_selection": True,
                "test_used_for_selection": False,
            },
        })
        (run_dir / "llm_audit.jsonl").write_text("", encoding="utf-8")
        (run_dir / "console.log").write_text("test\n", encoding="utf-8")
        self.refresh_status(run_dir, config_hash)

    def refresh_status(self, run_dir: Path, config_hash: str | None = None) -> None:
        if config_hash is None:
            config_hash = (run_dir / "config.sha256").read_text(encoding="utf-8").strip()
        _write_json(run_dir / "status.json", {
            "execution_status": "complete",
            "config_hash": config_hash,
            "plan_only": False,
            "artifact_sha256": {
                name: _sha256_file(run_dir / name) for name in HASHED_RUN_ARTIFACTS
            },
        })

    def plan(self):
        return json.loads((self.matrix / "plan_manifest.json").read_text(encoding="utf-8"))

    def write_plan(self, plan) -> None:
        _write_json(self.matrix / "plan_manifest.json", plan)


class FormalMatrixFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = MatrixFixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_validation_only_selection_and_symbolic_schema(self):
        payload = freeze_from_matrix(
            self.fixture.matrix,
            self.fixture.output,
            minimums=SMALL_MINIMUMS,
        )
        self.assertEqual(payload["selection_protocol"]["selected_run_id"], 1)
        self.assertFalse(payload["selection_protocol"]["test_used_for_selection"])
        # Run 2 has the much better test score; selecting run 1 proves test did not rank candidates.
        self.assertLess(
            payload["metrics"]["test_report_only"]["macro_average"]["r2"],
            0.5,
        )
        model = SymbolicModel.load(self.fixture.output)
        self.assertEqual(set(model.lane_parameters), set(LANES))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["status"], "frozen")
        self.assertEqual(len(payload["model_sha256"]), 64)
        self.assertIn("source_code_bundle_sha256", payload["provenance"])
        self.assertIn("source_data_config_sha256", payload["provenance"])

    def test_default_formal_gate_rejects_pilot_matrix(self):
        with self.assertRaisesRegex(
            MatrixAuditError, "exactly match|Pilot matrix rejected"
        ):
            audit_matrix(self.fixture.matrix)

    def test_formal_gate_requires_exactly_all_nine_intersections(self):
        nine_site_minimums = FormalMinimums(
            generations=1,
            population=1,
            runs_per_cell=2,
            optimizer_restarts=1,
        )
        with self.assertRaisesRegex(MatrixAuditError, "exactly match"):
            audit_matrix(self.fixture.matrix, minimums=nine_site_minimums)

    def test_rejects_plan_only_legacy_and_binary_only_plans(self):
        plan = self.fixture.plan()
        plan["execution_status"] = "plan_only"
        plan["plan_only_is_not_an_experiment_run"] = True
        self.fixture.write_plan(plan)
        with self.assertRaisesRegex(MatrixAuditError, "not complete|Plan-only"):
            audit_matrix(self.fixture.matrix, minimums=SMALL_MINIMUMS)

        self.fixture = MatrixFixture(Path(self.temporary.name) / "legacy")
        plan = self.fixture.plan()
        plan["experiment"] = "legacy"
        self.fixture.write_plan(plan)
        with self.assertRaisesRegex(MatrixAuditError, "Legacy|non-physics_score"):
            audit_matrix(self.fixture.matrix, minimums=SMALL_MINIMUMS)

        self.fixture = MatrixFixture(Path(self.temporary.name) / "binary")
        plan = self.fixture.plan()
        plan["variants"] = [item for item in plan["variants"] if item["variant"] == "binary"]
        self.fixture.write_plan(plan)
        with self.assertRaisesRegex(MatrixAuditError, "both binary and principlewise"):
            audit_matrix(self.fixture.matrix, minimums=SMALL_MINIMUMS)

    def test_rejects_partial_and_artifact_tamper(self):
        run_dir = self.fixture.run_dirs[("binary", 1)]
        (run_dir / "console.log").unlink()
        with self.assertRaisesRegex(MatrixAuditError, "Partial run"):
            audit_matrix(self.fixture.matrix, minimums=SMALL_MINIMUMS)

        self.fixture = MatrixFixture(Path(self.temporary.name) / "tamper")
        run_dir = self.fixture.run_dirs[("principlewise", 1)]
        with (run_dir / "model.json").open("a", encoding="utf-8") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(MatrixAuditError, "Artifact hash mismatch"):
            audit_matrix(self.fixture.matrix, minimums=SMALL_MINIMUMS)

    def test_rejects_fallback_even_when_hash_is_refreshed(self):
        run_dir = self.fixture.run_dirs[("principlewise", 1)]
        model = json.loads((run_dir / "model.json").read_text(encoding="utf-8"))
        model["status"] = "fallback_not_for_formal_control"
        _write_json(run_dir / "model.json", model)
        self.fixture.refresh_status(run_dir)
        with self.assertRaisesRegex(MatrixAuditError, "Fallback/legacy"):
            audit_matrix(self.fixture.matrix, minimums=SMALL_MINIMUMS)

    def test_recomputes_validation_metric_instead_of_trusting_json(self):
        run_dir = self.fixture.run_dirs[("principlewise", 1)]
        path = run_dir / "validation_predictions_long.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["y_pred"] = "999"
        _write_csv(path, rows, list(rows[0]))
        self.fixture.refresh_status(run_dir)
        audit = audit_matrix(self.fixture.matrix, minimums=SMALL_MINIMUMS)
        winner, _, _ = select_intersection1_principlewise(audit)
        # metrics.json still claims run 1 is best, but selection uses the recomputed CSV values.
        self.assertEqual(winner.config["run_id"], 2)

    def test_does_not_fall_back_to_lower_ranked_physical_candidate(self):
        run_dir = self.fixture.run_dirs[("principlewise", 1)]
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        for record in history:
            record["physical_joint_pass"] = False
        _write_json(run_dir / "history.json", history)
        self.fixture.refresh_status(run_dir)
        with self.assertRaisesRegex(MatrixAuditError, "refusing to substitute"):
            freeze_from_matrix(
                self.fixture.matrix,
                self.fixture.output,
                minimums=SMALL_MINIMUMS,
            )

    def test_overwrite_requires_force_and_force_archives(self):
        freeze_from_matrix(
            self.fixture.matrix,
            self.fixture.output,
            minimums=SMALL_MINIMUMS,
        )
        with self.assertRaises(FileExistsError):
            freeze_from_matrix(
                self.fixture.matrix,
                self.fixture.output,
                minimums=SMALL_MINIMUMS,
            )
        payload = freeze_from_matrix(
            self.fixture.matrix,
            self.fixture.output,
            minimums=SMALL_MINIMUMS,
            force=True,
        )
        self.assertIn("_archived_previous_model", payload)
        self.assertTrue(Path(payload["_archived_previous_model"]).is_file())


if __name__ == "__main__":
    unittest.main()
