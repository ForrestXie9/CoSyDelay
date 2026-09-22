"""Launch the authorized final-prompt pilot without exposing the API key."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from methods.cosydelay._internal.data_protocol.operations.run04_pythonw_supervisor import (
    recover_api_key,
)
from . import run_final_prompt_pilot as pilot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()

    names = ("LLM_API_KEY", "LLM_TEMPERATURE", "LLM_TOP_P", "LLM_MAX_TOKENS")
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
