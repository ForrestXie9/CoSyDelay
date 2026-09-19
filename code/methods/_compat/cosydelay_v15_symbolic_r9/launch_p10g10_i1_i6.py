"""Sequential V15 I1--I6 launcher using the authorized credential."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from methods.cosydelay_v9_clean_from_scratch.operations.run04_pythonw_supervisor import recover_api_key


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    status_path = root / "RUN_STATUS.json"
    status = {
        "status": "running",
        "started_utc": _now(),
        "method_id": "cosydelay_v15_symbolic_r9",
        "intersections": list(range(1, 7)),
        "population": 10,
        "generations": 10,
        "runs": [],
    }
    _write(status_path, status)
    previous_key = os.environ.get("LLM_API_KEY")
    try:
        os.environ["LLM_API_KEY"] = recover_api_key()
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        for intersection in range(1, 7):
            parent = root / f"intersection_{intersection:02d}"
            parent.mkdir(parents=True, exist_ok=True)
            output = parent / "run_01"
            started = _now()
            status.update(active_intersection=intersection, active_started_utc=started)
            _write(status_path, status)
            with (parent / "run_01.stdout.log").open("w", encoding="utf-8") as stdout, (
                parent / "run_01.stderr.log"
            ).open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "methods.cosydelay_v15_symbolic_r9.run_p10g10_training",
                        "--intersection",
                        str(intersection),
                        "--output",
                        str(output),
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    env=child_env,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            status["runs"].append(
                {
                    "intersection_id": intersection,
                    "started_utc": started,
                    "completed_utc": _now(),
                    "return_code": completed.returncode,
                    "result_exists": (output / "result.json").is_file(),
                }
            )
            status.pop("active_intersection", None)
            status.pop("active_started_utc", None)
            _write(status_path, status)
            if completed.returncode:
                status.update(status="failed", failed_intersection=intersection)
                _write(status_path, status)
                return int(completed.returncode)
    finally:
        if previous_key is None:
            os.environ.pop("LLM_API_KEY", None)
        else:
            os.environ["LLM_API_KEY"] = previous_key
    status.update(status="completed", completed_utc=_now())
    _write(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
