"""Launch a V14 pilot with the previously authorized credential."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from methods.cosydelay_v9_clean_from_scratch.operations.run04_pythonw_supervisor import (
    recover_api_key,
)
from . import run_training_pilot as pilot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous_key = os.environ.get("LLM_API_KEY")
    try:
        os.environ["LLM_API_KEY"] = recover_api_key()
        sys.argv = [
            str(pilot.__file__),
            "--intersection", str(args.intersection),
            "--output", str(args.output),
        ]
        return pilot.main()
    finally:
        if previous_key is None:
            os.environ.pop("LLM_API_KEY", None)
        else:
            os.environ["LLM_API_KEY"] = previous_key


if __name__ == "__main__":
    raise SystemExit(main())
