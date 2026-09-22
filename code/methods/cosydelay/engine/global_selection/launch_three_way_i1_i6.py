"""Run one V21 repeat on frozen Train, then evaluate Validation and Test."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    code = subprocess.call(command)
    if code:
        raise SystemExit(code)


def relabel_validation(output: Path) -> None:
    predictions = output / "predictions.csv"
    rows = list(csv.DictReader(predictions.open(encoding="utf-8")))
    for row in rows:
        row["split"] = "validation"
    with predictions.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    aggregate_path = output / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["scope"] = "fixed V21 prediction-only evaluation on frozen Validation"
    aggregate["evaluation_split"] = "validation"
    aggregate["test_used_for_selection"] = False
    aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    data_dir, root = args.data_dir.resolve(), args.output_root.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    search = root / "search"
    run([sys.executable, "-u", "-m", "methods.cosydelay.engine.global_selection.launch_i1_i6",
         "--data-dir", str(data_dir), "--output-root", str(search)])

    validation_stage = root / "validation_stage"
    validation_stage.mkdir()
    for intersection in range(1, 7):
        shutil.copy2(data_dir / f"Intersection_{intersection}_Train.jsonl",
                     validation_stage / f"Intersection_{intersection}_Train.jsonl")
        shutil.copy2(data_dir / f"Intersection_{intersection}_Validation.jsonl",
                     validation_stage / f"Intersection_{intersection}_Test.jsonl")
    validation = root / "validation_evaluation"
    run([sys.executable, "-u", "-m", "methods.cosydelay.engine.global_selection.evaluate_fixed_test",
         "--search-root", str(search), "--data-dir", str(validation_stage),
         "--output", str(validation)])
    relabel_validation(validation)

    test = root / "test_evaluation"
    run([sys.executable, "-u", "-m", "methods.cosydelay.engine.global_selection.evaluate_fixed_test",
         "--search-root", str(search), "--data-dir", str(data_dir), "--output", str(test)])
    manifest = {
        "status": "complete", "method": "V21 global selection + numeric domain",
        "repeat": 1, "training_only_search": True,
        "validation_used_for_search_or_fit": False, "test_used_for_search_or_fit": False,
        "data_dir": str(data_dir), "search_root": str(search),
        "validation_output": str(validation), "test_output": str(test),
    }
    (root / "THREE_WAY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
