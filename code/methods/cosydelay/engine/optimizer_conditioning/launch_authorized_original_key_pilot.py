"""Launch one authorized pilot from a real file for Windows spawn safety.

The user explicitly authorized reuse of the original API credential on
2026-08-11. The credential exists only in this child process environment and
is removed in ``finally``. It is never printed or written to result artifacts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from methods.cosydelay.engine.data_protocol.operations.run04_pythonw_supervisor import (
    recover_api_key,
)
from methods.cosydelay.engine.optimizer_conditioning import (
    run_complexity_prompt_pilot as pilot,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()

    names = (
        "LLM_API_KEY",
        "LLM_TEMPERATURE",
        "LLM_TOP_P",
        "LLM_MAX_TOKENS",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ["LLM_API_KEY"] = recover_api_key()
        for name in names[1:]:
            os.environ.pop(name, None)
        forwarded = [
            str(pilot.__file__),
            "--intersection",
            str(args.intersection),
            "--output",
            str(args.output),
        ]
        if args.data_dir is not None:
            forwarded.extend(["--data-dir", str(args.data_dir)])
        sys.argv = forwarded
        return pilot.main()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
