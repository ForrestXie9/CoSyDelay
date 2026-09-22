"""Run V17 I1--I6 Training searches, then open Test exactly once."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

from methods.cosydelay._internal.data_protocol.run_formal_training_search import (
    DEFAULT_DATA_DIR,
)

from .source_manifest import GMINI


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--skip-baseline-ranking", action="store_true")
    return parser.parse_args()


def _run(command: list[str], stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=GMINI,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    return int(completed.returncode)


def main() -> int:
    args = arguments()
    search_root = args.search_root.resolve()
    test_output = args.test_output.resolve()
    parent = search_root.parent
    training_command = [
        sys.executable,
        "-m",
        "methods.cosydelay._internal.retry_protocol.launch_p10g10_i1_i6",
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output-root",
        str(search_root),
    ]
    code = _run(
        training_command,
        parent / f"{search_root.name}.launcher.stdout.log",
        parent / f"{search_root.name}.launcher.stderr.log",
    )
    if code:
        # Fail closed: the Test evaluator process is never created.
        return code
    test_command = [
        sys.executable,
        "-m",
        "methods.cosydelay._internal.retry_protocol.evaluate_frozen_test",
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
    )


if __name__ == "__main__":
    raise SystemExit(main())
