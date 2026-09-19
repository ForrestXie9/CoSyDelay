"""Run bounded V18 I1--I6 Training searches, then open Test once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import (
    DEFAULT_DATA_DIR,
)

from .contract import V18_CONTRACT
from .fit_timeout import wait_subprocess_bounded
from .source_manifest import GMINI


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--skip-baseline-ranking", action="store_true")
    return parser.parse_args()


def _run(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    *,
    timeout_seconds: float,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=GMINI,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return_code, timed_out, cleanup = wait_subprocess_bounded(
            process, timeout_seconds=timeout_seconds
        )
    audit_path = stdout_path.with_suffix(stdout_path.suffix + ".process.json")
    audit_path.write_text(
        json.dumps(
            {
                "command_module": (
                    command[command.index("-m") + 1] if "-m" in command else None
                ),
                "return_code": int(return_code),
                "timeout_seconds": float(timeout_seconds),
                "timed_out": bool(timed_out),
                "wall_seconds": float(time.perf_counter() - started),
                "process_tree_cleanup": cleanup,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return int(return_code)


def main() -> int:
    args = arguments()
    search_root = args.search_root.resolve()
    test_output = args.test_output.resolve()
    parent = search_root.parent
    training_command = [
        sys.executable,
        "-m",
        "methods.cosydelay_v18_fit_timeout_guard.launch_p10g10_i1_i6",
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output-root",
        str(search_root),
    ]
    code = _run(
        training_command,
        parent / f"{search_root.name}.launcher.stdout.log",
        parent / f"{search_root.name}.launcher.stderr.log",
        timeout_seconds=V18_CONTRACT.full_training_launcher_wall_timeout_seconds,
    )
    if code:
        # Fail closed: the Test evaluator process is never created.
        return code
    test_command = [
        sys.executable,
        "-m",
        "methods.cosydelay_v18_fit_timeout_guard.evaluate_frozen_test",
        "--search-root",
        str(search_root),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output",
        str(test_output),
    ]
    if args.baseline_dir is not None:
        test_command.extend(["--baseline-dir", str(args.baseline_dir.resolve())])
    if args.skip_baseline_ranking:
        test_command.append("--skip-baseline-ranking")
    return _run(
        test_command,
        parent / f"{search_root.name}.test.stdout.log",
        parent / f"{search_root.name}.test.stderr.log",
        timeout_seconds=V18_CONTRACT.test_evaluator_wall_timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())

