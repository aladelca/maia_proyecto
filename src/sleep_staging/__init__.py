"""Procesamiento y experimentación local de señales de sueño."""

import os
from pathlib import Path

from dotenv import load_dotenv

for _directory in [Path.cwd(), *Path.cwd().parents]:
    if (_directory / "pyproject.toml").exists():
        load_dotenv(_directory / ".env", override=False)
        break
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
