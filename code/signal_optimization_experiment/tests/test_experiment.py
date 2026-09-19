import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from experiment.config import (
    LANES,
    LOST_TIME,
    MAX_CYCLE,
    MAX_GREEN,
    MIN_CYCLE,
    MIN_GREEN,
    MODEL_PATH,
    FIXED_PLAN_PATH,
    PHASES,
)
from experiment.controllers import Controller, max_pressure_plan
from experiment.demand import generate_route_file_from_rates, scenario_rates
from experiment.real_demand import _parse_flows_only, demand_hash, load_intersection1_demands


class ExperimentTests(unittest.TestCase):
    def test_all_scenarios_define_every_lane(self):
        for scenario in ("balanced", "asymmetric", "near_saturation", "turning_surge", "switch"):
            rates = scenario_rates(scenario, 0, 3600)
            self.assertEqual(set(rates), set(LANES))
            self.assertTrue(all(rate >= 0 for rate in rates.values()))

    def test_controller_plans_obey_common_timing_bounds(self):
        rates = scenario_rates("balanced", 0, 3600)
        queues = {lane: float(index % 5) for index, lane in enumerate(LANES)}
        self.assertTrue(MODEL_PATH.exists())
        for name in (
            "webster", "akcelik", "hcm", "fixed", "actuated",
            "max_pressure", "rule", "symbolic",
        ):
            plan = Controller(name).plan(rates, queues)
            self.assertEqual(set(plan), set(PHASES), name)
            self.assertTrue(all(np.isfinite(value) for value in plan.values()), name)
            self.assertTrue(all(MIN_GREEN - 1e-6 <= value <= MAX_GREEN + 1e-6 for value in plan.values()), name)
            cycle = LOST_TIME + sum(plan.values())
            self.assertGreaterEqual(cycle, MIN_CYCLE - 1e-5, name)
            self.assertLessEqual(cycle, MAX_CYCLE + 1e-5, name)

    def test_fixed_plan_does_not_respond_to_demand_or_queues(self):
        controller = Controller("fixed")
        first = controller.plan(
            scenario_rates("balanced", 0, 3600),
            {lane: 0.0 for lane in LANES},
        )
        second = controller.plan(
            scenario_rates("asymmetric", 0, 3600),
            {lane: 100.0 for lane in LANES},
        )
        self.assertEqual(first, second)

    def test_max_pressure_uses_queue_differential(self):
        queues = {lane: 0.0 for lane in LANES}
        downstream = {lane: 0.0 for lane in LANES}
        queues["E_R"] = 20.0
        queues["W_R"] = 15.0
        plan = max_pressure_plan(queues, downstream)
        self.assertGreater(plan["C"], max(plan["A"], plan["B"], plan["D"]))

        downstream["E_R"] = 20.0
        downstream["W_R"] = 15.0
        queues["S_T"] = 5.0
        plan = max_pressure_plan(queues, downstream)
        self.assertGreater(plan["B"], plan["C"])

    def test_real_demand_exposes_only_observed_lane_flows(self):
        train = load_intersection1_demands("train")
        validation = load_intersection1_demands("validation")
        rows = load_intersection1_demands("test")
        self.assertEqual(len(train), 1001)
        self.assertEqual(len(validation), 250)
        self.assertEqual(len(rows), 503)
        self.assertEqual(train[-1].source_row_id, 1001)
        self.assertEqual(validation[0].source_row_id, 1002)
        self.assertEqual(validation[-1].source_row_id, 1251)
        train_hashes = {row.demand_sha256 for row in train}
        validation_hashes = {row.demand_sha256 for row in validation}
        test_hashes = {row.demand_sha256 for row in rows}
        self.assertEqual(len(train_hashes), len(train))
        self.assertEqual(len(validation_hashes), len(validation))
        self.assertEqual(len(test_hashes), len(rows))
        self.assertFalse(train_hashes & validation_hashes)
        self.assertFalse(train_hashes & test_hashes)
        self.assertFalse(validation_hashes & test_hashes)
        self.assertEqual(rows[0].row_id, 1)
        self.assertEqual(set(rows[0].rates), set(LANES))
        self.assertEqual(rows[0].rates["S_L"], 307.0)
        self.assertEqual(rows[0].rates["S_T"], 1018.0)
        self.assertEqual(rows[0].rates["W_R"], 62.0)
        self.assertFalse(hasattr(rows[0], "green_times"))
        self.assertFalse(hasattr(rows[0], "delays"))

    def test_real_rate_route_generation_is_paired_and_deterministic(self):
        rates = scenario_rates("balanced", 0, 60)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.rou.xml"
            second = Path(directory) / "second.rou.xml"
            first_count = generate_route_file_from_rates(first, rates, 17, 60)
            second_count = generate_route_file_from_rates(second, rates, 17, 60)
            self.assertEqual(first_count, second_count)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_real_flow_parser_ignores_green_times_and_labels(self):
        flows = """South Approach: Left Turn = 1, Through = 2, Right Turn = 3
East Approach: Left Turn = 4, Through = 5, Right Turn = 6
North Approach: Left Turn = 7, Through = 8, Right Turn = 9
West Approach: Left Turn = 10, Through = 11, Right Turn = 12"""
        first = _parse_flows_only(flows + "\nPhase A: 5\nDelay = 10")
        second = _parse_flows_only(flows + "\nPhase A: 99\nDelay = invalid")
        self.assertEqual(first, second)
        self.assertEqual(demand_hash(first), demand_hash(second))

    def test_real_route_rejects_incomplete_rates(self):
        rates = scenario_rates("balanced", 0, 60)
        rates.pop("S_L")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                generate_route_file_from_rates(
                    Path(directory) / "invalid.rou.xml", rates, 1, 10
                )

    def test_fixed_plan_is_frozen_from_train_only(self):
        data = json.loads(FIXED_PLAN_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data["provenance"]["split"], "train")
        self.assertEqual(data["provenance"]["training_rows"], 1001)
        self.assertFalse(data["provenance"]["test_data_used"])
        self.assertFalse(data["provenance"]["original_green_times_used"])
        self.assertFalse(data["provenance"]["delay_labels_used"])


if __name__ == "__main__":
    unittest.main()
