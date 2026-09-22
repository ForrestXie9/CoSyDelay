"""Launch the complete run04 batch from a no-console Python scheduler task."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import traceback


GMINI = Path(
    r"Z:\CoSydelay\Paper_SM_traffic_delay\Parameters_sensitive\gmini"
)
if str(GMINI) not in sys.path:
    sys.path.insert(0, str(GMINI))

from methods.cosydelay.engine.data_protocol import (  # noqa: E402
    launch_formal_i1_i6 as launcher,
)


ROOT = GMINI / (
    "methods/cosydelay_v9_clean_from_scratch/experiments/"
    "formal_v9_paper_fitness_promptv3_providerdefaults_"
    "p10g10_i1_i6_run04_20260810"
)
HISTORY_FILE = Path(
    r"Z:\CoSydelay\Paper_SM_traffic_delay\Baseline_methods\paper_ml_reproduction"
    r"\local_history_recovery\pycharm2025_1_vfs_resolved"
    r"\20260710_153720_150_record_0000000352_change_0000_jiekou_llm_embedded.py"
)
STDOUT_PATH = ROOT.parent / f"{ROOT.name}.launcher.stdout.log"
STDERR_PATH = ROOT.parent / f"{ROOT.name}.launcher.stderr.log"
SUPERVISOR_PATH = ROOT.parent / f"{ROOT.name}.pythonw_supervisor.json"


def recover_api_key() -> str:
    text = HISTORY_FILE.read_text(encoding="utf-8")
    match = re.search(
        r"(?m)^JIEKOU_API_KEY\s*=\s*['\"](sk-[A-Za-z0-9_-]{20,})['\"]\s*$",
        text,
    )
    if not match:
        raise RuntimeError("API credential could not be recovered")
    return match.group(1)


def write_supervisor(value: dict) -> None:
    temporary = SUPERVISOR_PATH.with_suffix(SUPERVISOR_PATH.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(SUPERVISOR_PATH)


def main() -> int:
    if ROOT.exists():
        raise FileExistsError(f"refusing existing output root: {ROOT}")
    os.environ["LLM_API_KEY"] = recover_api_key()
    for name in ("LLM_TEMPERATURE", "LLM_TOP_P", "LLM_MAX_TOKENS"):
        os.environ.pop(name, None)
    sys.argv = [
        str(launcher.__file__),
        "--output-root",
        str(ROOT),
    ]
    write_supervisor(
        {
            "status": "running",
            "started_utc": launcher.utc_now(),
            "host": "pythonw_no_console",
        }
    )
    return launcher.main()


if __name__ == "__main__":
    STDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STDOUT_PATH.open("w", encoding="utf-8", buffering=1) as stdout, (
        STDERR_PATH.open("w", encoding="utf-8", buffering=1)
    ) as stderr:
        sys.stdout = stdout
        sys.stderr = stderr
        try:
            exit_code = main()
            write_supervisor(
                {
                    "status": "complete" if exit_code == 0 else "failed",
                    "completed_utc": launcher.utc_now(),
                    "return_code": int(exit_code),
                    "host": "pythonw_no_console",
                }
            )
        except Exception as exc:
            traceback.print_exc()
            write_supervisor(
                {
                    "status": "failed_exception",
                    "completed_utc": launcher.utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "host": "pythonw_no_console",
                }
            )
            raise
        finally:
            os.environ.pop("LLM_API_KEY", None)
    raise SystemExit(exit_code)
