"""Orchestrator: runs each selected model as a separate subprocess in its own
venv (no manual activation needed) -> results/<model>_<indicator>_forecasts.csv.

Edit MODEL_NAMES below to choose what runs.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "forecast_worker.py"
OECD_SCRIPT = ROOT / "oecd_api.py"

# Select models to run here.
MODEL_NAMES = ["historical", "qar", "timesfm", "chronos2", "moirai", "timestral", "t0gamma", "sundial"]


def venv_python(environment_name):
    """Platform-specific python path inside environments/.venv-<name>."""

    environment = ROOT / "environments" / f".venv-{environment_name}"
    if sys.platform == "win32":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


# Each model's dedicated venv python; None falls back to the current interpreter.
VENVS = {
    "historical": venv_python("benchmark"),
    "qar": venv_python("benchmark"),
    "chronos2": venv_python("chronos"),
    "sundial": venv_python("sundial"),
    "timesfm": venv_python("timesfm"),
    "moirai": venv_python("moirai"),
    "timestral": venv_python("timesfm"),
    "t0gamma": venv_python("timesfm"),
}


def child_env(python_exe):
    """A clean environment for python_exe, independent of any venv activated
    in the calling shell (an inherited PATH/VIRTUAL_ENV can make the child
    pick up a different venv's DLLs, e.g. mismatched torch builds)."""

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)

    venv_scripts = str(Path(python_exe).parent)
    path_parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    path_parts = [p for p in path_parts if "environments" not in p.lower()]
    env["PATH"] = os.pathsep.join([venv_scripts] + path_parts)

    return env


def fetch_oecd_data():
    """Run oecd_api.py in the benchmark venv; the forecasts depend on its data."""

    python_exe = venv_python("benchmark")
    print(f"[oecd] running {OECD_SCRIPT.name} ({python_exe})")
    completed = subprocess.run(
        [str(python_exe), str(OECD_SCRIPT)],
        cwd=ROOT,
        env=child_env(python_exe),
    )
    if completed.returncode != 0:
        sys.exit(f"{OECD_SCRIPT.name} FAILED (exit {completed.returncode})")


def main():

    #fetch_oecd_data()

    results = {}

    for i, name in enumerate(MODEL_NAMES, start=1):

        python_exe = VENVS.get(name) or sys.executable
        print(f"[{i}/{len(MODEL_NAMES)}] running {name} ({python_exe})")

        completed = subprocess.run(
            [str(python_exe), str(WORKER), name],
            cwd=ROOT,
            env=child_env(python_exe),
        )
        results[name] = completed.returncode

        status = "ok" if completed.returncode == 0 else f"FAILED (exit {completed.returncode})"
        print(f"[{i}/{len(MODEL_NAMES)}] {name}: {status}")

    print("\nsummary:")
    for name, code in results.items():
        print(f"  {name}: {'ok' if code == 0 else f'FAILED (exit {code})'}")


if __name__ == "__main__":
    main()
