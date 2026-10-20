"""Create the model virtual environments used by ``run_forecasts.py``.

Run this file with the Python installation that should own the environments:

    python requirements/setup.py

Existing environments are detected by their Python executable and are left
untouched. Models that share an environment install its requirements only
once.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None


ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = Path(__file__).resolve().parent
ENVIRONMENTS = ROOT / "environments"
INSTALLED_MARKER = ".installed"

PYTHON_HEADER = re.compile(r"(?m)^#\s*python:\s*(\d+\.\d+)\s*$")


def required_python(requirement_file: Path) -> str | None:
    """Return the ``# python: X.Y`` version declared in a requirements file."""

    match = PYTHON_HEADER.search(requirement_file.read_text(encoding="utf-8-sig"))
    return match.group(1) if match else None


def find_uv() -> list[str] | None:
    """Return a command that runs uv, installing it with pip if necessary."""

    if shutil.which("uv"):
        return ["uv"]
    command = [sys.executable, "-m", "uv"]
    if subprocess.run(command + ["--version"], capture_output=True).returncode:
        print("[install] uv (used to obtain the required Python version)")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "uv"], check=False
        )
        if subprocess.run(command + ["--version"], capture_output=True).returncode:
            return None
    return command


def load_toml_without_dependency() -> dict[str, object]:
    """Read the small config subset needed before dependencies are installed."""

    text = (ROOT / "config.toml").read_text(encoding="utf-8")
    models_match = re.search(r"(?m)^MODELS\s*=\s*(\[.*\])\s*$", text)
    environments_match = re.search(
        r"(?ms)^\[MODEL_ENVIRONMENTS\]\s*(.*?)(?=^\[|\Z)", text
    )
    if not models_match or not environments_match:
        raise ValueError(
            "config.toml must define MODELS and [MODEL_ENVIRONMENTS]"
        )

    models = ast.literal_eval(models_match.group(1))
    environment_names = {}
    for line in environments_match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, value = line.split("=", maxsplit=1)
        environment_names[name.strip()] = ast.literal_eval(value.strip())

    return {"MODELS": models, "MODEL_ENVIRONMENTS": environment_names}


def load_configuration() -> tuple[tuple[str, ...], dict[str, str]]:
    """Read and validate the model/environment configuration."""

    if tomllib is None:
        config = load_toml_without_dependency()
    else:
        with (ROOT / "config.toml").open("rb") as config_file:
            config = tomllib.load(config_file)

    models = tuple(config.get("MODELS", ()))
    environment_names = config.get("MODEL_ENVIRONMENTS", {})

    if not models:
        raise ValueError("config.toml must define at least one model in MODELS")

    missing = [model for model in models if model not in environment_names]
    if missing:
        raise ValueError(
            "config.toml has no MODEL_ENVIRONMENTS entry for: "
            + ", ".join(missing)
        )

    return models, environment_names


def python_executable(environment_name: str) -> Path:
    """Return the platform-specific Python path inside an environment."""

    environment = ENVIRONMENTS / f".venv-{environment_name}"
    if sys.platform == "win32":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def find_python(version: str) -> str:
    """Locate a Python interpreter of the given ``major.minor`` version."""

    if f"{sys.version_info.major}.{sys.version_info.minor}" == version:
        return sys.executable

    candidates = [shutil.which(f"python{version}")]
    if sys.platform == "win32" and shutil.which("py"):
        candidates.append(
            subprocess.run(
                ["py", f"-{version}", "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    if any(c and Path(c).exists() for c in candidates):
        uv = None
    else:
        uv = find_uv()
    if uv:
        subprocess.run(uv + ["python", "install", version], check=False)
        candidates.append(
            subprocess.run(
                uv + ["python", "find", version],
                capture_output=True,
                text=True,
            ).stdout.strip()
        )

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate

    raise RuntimeError(
        f"Python {version} is required but was not found. Install it (or uv) "
        "and run this script again."
    )


def setup_environment(environment_name: str, requirement_file: Path) -> None:
    executable = python_executable(environment_name)
    environment = executable.parent.parent

    if (environment / INSTALLED_MARKER).exists():
        print(f"[skip] .venv-{environment_name} already exists")
        return
    if environment.exists():
        print(f"[remove] incomplete .venv-{environment_name}")
        shutil.rmtree(environment)

    environment.parent.mkdir(parents=True, exist_ok=True)
    print(f"[create] .venv-{environment_name}")
    version = required_python(requirement_file)
    if version is None:
        venv.create(environment, with_pip=True)
    else:
        subprocess.run(
            [find_python(version), "-m", "venv", str(environment)], check=True
        )

    print(f"[install] {requirement_file.name} -> .venv-{environment_name}")
    subprocess.run(
        [str(executable), "-m", "pip", "install", "-r", str(requirement_file)],
        cwd=ROOT,
        check=True,
    )
    (environment / INSTALLED_MARKER).touch()


def main() -> None:
    models, environment_names = load_configuration()
    required_environments = dict.fromkeys(
        environment_names[model] for model in models
    )

    for environment_name in required_environments:
        requirement_file = REQUIREMENTS / f"{environment_name}.txt"
        if not requirement_file.is_file():
            raise FileNotFoundError(
                f"Missing requirements file for {environment_name}: "
                f"{requirement_file}"
            )
        setup_environment(environment_name, requirement_file)

    print(f"Ready: {', '.join(models)}")


if __name__ == "__main__":
    main()
