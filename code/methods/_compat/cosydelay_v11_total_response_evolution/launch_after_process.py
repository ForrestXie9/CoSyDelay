"""Wait for an existing batch, then run isolated V11 pilots sequentially."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


SYNCHRONIZE = 0x00100000
WAIT_FAILED = 0xFFFFFFFF


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def wait_for_process(pid: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
    if not handle:
        # The process already ended between scheduling and supervisor startup.
        return
    try:
        result = kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        if result == WAIT_FAILED:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(handle)


def write_status(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--intersections", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    status_path = root / "SUPERVISOR_STATUS.json"
    status = {
        "status": "waiting_for_v10",
        "wait_pid": args.wait_pid,
        "intersections": args.intersections,
        "started_utc": utc_now(),
        "runs": [],
    }
    write_status(status_path, status)
    wait_for_process(args.wait_pid)

    status["status"] = "running_v11_pilots"
    status["v10_completed_observed_utc"] = utc_now()
    write_status(status_path, status)
    for intersection in args.intersections:
        output = root / f"intersection_{intersection:02d}" / "run_01"
        output.parent.mkdir(parents=True, exist_ok=True)
        stdout_path = output.parent / "run_01.stdout.log"
        stderr_path = output.parent / "run_01.stderr.log"
        started = utc_now()
        command = [
            sys.executable,
            "-m",
            (
                "methods.cosydelay_v11_total_response_evolution."
                "launch_authorized_original_key_pilot"
            ),
            "--intersection",
            str(intersection),
            "--output",
            str(output),
        ]
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[2],
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        status["runs"].append(
            {
                "intersection_id": intersection,
                "started_utc": started,
                "completed_utc": utc_now(),
                "return_code": completed.returncode,
                "output": str(output),
                "result_exists": (output / "result.json").is_file(),
            }
        )
        write_status(status_path, status)
        if completed.returncode != 0:
            status["status"] = "failed"
            write_status(status_path, status)
            return completed.returncode

    status["status"] = "completed"
    status["completed_utc"] = utc_now()
    write_status(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
