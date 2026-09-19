"""Freeze a fixed-time baseline designed from Intersection-1 training flows."""
from __future__ import annotations

import hashlib
import json

from experiment.config import FIXED_PLAN_PATH, LANES
from experiment.controllers import webster_plan
from experiment.real_demand import load_intersection1_demands


def main() -> None:
    rows = load_intersection1_demands("train")
    mean_rates = {
        lane: sum(row.rates[lane] for row in rows) / len(rows)
        for lane in LANES
    }
    plan = webster_plan(mean_rates)
    core = {"plan": plan, "training_mean_rates": mean_rates}
    checksum = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output = {
        "schema_version": 1,
        "status": "frozen",
        "plan_sha256": checksum,
        **core,
        "provenance": {
            "method": "Webster design applied to lane-flow means",
            "split": "train",
            "training_rows": len(rows),
            "source_path": str(rows[0].source_path.resolve()),
            "source_sha256": rows[0].source_sha256,
            "test_data_used": False,
            "original_green_times_used": False,
            "delay_labels_used": False,
        },
    }
    FIXED_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIXED_PLAN_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary.replace(FIXED_PLAN_PATH)
    print(FIXED_PLAN_PATH)
    print(plan)


if __name__ == "__main__":
    main()
