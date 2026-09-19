from pathlib import Path
import os
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent
NETWORK = ROOT / "network"


def find_binary(name: str) -> str:
    candidate = shutil.which(name)
    if candidate:
        return candidate
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        path = Path(sumo_home) / "bin" / f"{name}.exe"
        if path.exists():
            return str(path)
    standard = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin") / f"{name}.exe"
    if standard.exists():
        return str(standard)
    raise FileNotFoundError(f"Cannot find SUMO binary {name}")


def main() -> None:
    command = [
        find_binary("netconvert"),
        "--node-files", str(NETWORK / "intersection.nod.xml"),
        "--edge-files", str(NETWORK / "intersection.edg.xml"),
        "--connection-files", str(NETWORK / "intersection.con.xml"),
        "--output-file", str(NETWORK / "intersection.net.xml"),
        "--no-turnarounds", "true",
    ]
    subprocess.run(command, check=True, cwd=NETWORK)
    print(NETWORK / "intersection.net.xml")


if __name__ == "__main__":
    main()
