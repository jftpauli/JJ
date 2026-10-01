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

# Select models to run here.
MODEL_NAMES = ["historical", "qar", "timesfm", "chronos2", "sundial"]

# Each model's dedicated venv python; None falls back to the current interpreter.
VENVS = {
    "historical": ROOT / "environments/.venv-benchmark/Scripts/python.exe",
    "qar": ROOT / "environments/.venv-benchmark/Scripts/python.exe",
    "chronos2": ROOT / "environments/.venv-chronos/Scripts/python.exe",
    "sundial": ROOT / "environments/.venv-sundial/Scripts/python.exe",
    "timesfm": ROOT / "environments/.venv-timesfm/Scripts/python.exe",
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


def main():

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
